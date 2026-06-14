"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { getConfigDefaults, interpretDrawing, startRun } from "@/lib/api";
import type { RunConfigInput } from "@/lib/types";
import type { Article } from "@/lib/article";
import { ArticleSidebar } from "@/components/ArticleSidebar";
import { ArticleDetail } from "@/components/ArticleDetail";
import { HeroPrompt } from "@/components/HeroPrompt";

const MAX_ARTICLES = 12;

let counter = 0;
const nextId = () => `art_${Date.now().toString(36)}_${(counter++).toString(36)}`;

export function ArticleWorkspace() {
  const router = useRouter();
  const [articles, setArticles] = useState<Article[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [generatingId, setGeneratingId] = useState<string | null>(null);
  const [model, setModel] = useState<string>("");

  // Keep a ref to the latest articles so async callbacks can find a file by id
  // without going stale.
  const articlesRef = useRef<Article[]>([]);
  articlesRef.current = articles;

  useEffect(() => {
    let cancelled = false;
    getConfigDefaults()
      .then((d) => {
        if (!cancelled && d.model) setModel(d.model);
      })
      .catch(() => {
        /* backend down — leave empty so the server env default applies */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Revoke all object URLs on unmount.
  useEffect(() => {
    return () => {
      for (const a of articlesRef.current) URL.revokeObjectURL(a.thumbUrl);
    };
  }, []);

  const buildConfig = useCallback(
    (): RunConfigInput => ({ model: model.trim() || undefined }),
    [model],
  );

  const patch = useCallback((id: string, changes: Partial<Article>) => {
    setArticles((prev) => prev.map((a) => (a.id === id ? { ...a, ...changes } : a)));
  }, []);

  const readArticle = useCallback(
    async (id: string) => {
      const article = articlesRef.current.find((a) => a.id === id);
      if (!article) return;
      patch(id, { status: "reading", startedAt: Date.now(), error: null });
      try {
        const { interpretation } = await interpretDrawing("", buildConfig(), [article.file]);
        patch(id, { status: "ready", interpretation });
      } catch (e) {
        const base = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
        patch(id, {
          status: "error",
          error: `${e instanceof Error ? e.message : "read failed"} — is the backend running on ${base}?`,
        });
      }
    },
    [buildConfig, patch],
  );

  const addFiles = useCallback(
    (files: File[]) => {
      const room = MAX_ARTICLES - articlesRef.current.length;
      if (room <= 0) return;
      const incoming = files.slice(0, room).map<Article>((file) => ({
        id: nextId(),
        file,
        name: file.name,
        thumbUrl: URL.createObjectURL(file),
        status: "reading",
        startedAt: Date.now(),
        interpretation: "",
        notes: "",
        error: null,
      }));
      setArticles((prev) => [...prev, ...incoming]);
      setSelectedId((cur) => cur ?? incoming[0]?.id ?? null);
      // Kick off reads in parallel; readArticle resolves the file from the ref.
      for (const a of incoming) void readArticle(a.id);
    },
    [readArticle],
  );

  const removeArticle = useCallback((id: string) => {
    setArticles((prev) => {
      const target = prev.find((a) => a.id === id);
      if (target) URL.revokeObjectURL(target.thumbUrl);
      const next = prev.filter((a) => a.id !== id);
      setSelectedId((cur) => (cur === id ? (next[0]?.id ?? null) : cur));
      return next;
    });
  }, []);

  const generate = useCallback(
    async (id: string) => {
      const article = articlesRef.current.find((a) => a.id === id);
      if (!article || generatingId) return;
      setGeneratingId(id);
      try {
        const { run_id } = await startRun(
          article.notes.trim(),
          buildConfig(),
          [article.file],
          article.interpretation,
        );
        router.push(`/runs/${run_id}`);
      } catch (e) {
        const base = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
        patch(id, {
          status: "error",
          error: `${e instanceof Error ? e.message : "start failed"} — is the backend running on ${base}?`,
        });
        setGeneratingId(null);
      }
    },
    [buildConfig, generatingId, patch, router],
  );

  const selected = articles.find((a) => a.id === selectedId) ?? null;

  return (
    <div className="flex flex-col gap-12">
      <HeroPrompt onGenerate={addFiles} busy={false} />

      {articles.length > 0 && (
        <section className="flex flex-col gap-4">
          <div className="flex items-baseline gap-3">
            <h2 className="tech-head text-xl text-ink">Your articles</h2>
            <span className="tech-label">read in parallel</span>
          </div>
          <div className="flex flex-col gap-6 lg:flex-row lg:items-start">
            <ArticleSidebar
              articles={articles}
              selectedId={selectedId}
              onSelect={setSelectedId}
              onAddFiles={addFiles}
              onRemove={removeArticle}
            />
            <ArticleDetail
              article={selected}
              onChangeInterpretation={(v) => selected && patch(selected.id, { interpretation: v })}
              onChangeNotes={(v) => selected && patch(selected.id, { notes: v })}
              onRetry={() => selected && readArticle(selected.id)}
              onGenerate={() => selected && generate(selected.id)}
              generating={generatingId === selectedId}
            />
          </div>
        </section>
      )}
    </div>
  );
}
