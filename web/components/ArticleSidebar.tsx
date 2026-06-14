"use client";

import { useRef, useState } from "react";
import { ReadingPipeline } from "@/components/ReadingPipeline";
import type { Article } from "@/lib/article";

const ACCEPT = ["image/png", "image/jpeg"];

export function ArticleSidebar({
  articles,
  selectedId,
  onSelect,
  onAddFiles,
  onRemove,
}: {
  articles: Article[];
  selectedId: string | null;
  onSelect: (id: string) => void;
  onAddFiles: (files: File[]) => void;
  onRemove: (id: string) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function handleFiles(list: FileList | File[]) {
    const accepted = Array.from(list).filter((f) => ACCEPT.includes(f.type));
    if (accepted.length) onAddFiles(accepted);
  }

  return (
    <aside className="flex w-full flex-col gap-3 lg:w-[300px] lg:shrink-0">
      <div className="flex items-baseline justify-between">
        <span className="tech-label">articles</span>
        <span className="tech-label">{articles.length}</span>
      </div>

      <div className="flex flex-col gap-2">
        {articles.map((a) => (
          <ArticleCard
            key={a.id}
            article={a}
            selected={a.id === selectedId}
            onSelect={() => onSelect(a.id)}
            onRemove={() => onRemove(a.id)}
          />
        ))}
      </div>

      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handleFiles(e.dataTransfer.files);
        }}
        className={`cursor-pointer rounded-[var(--radius-tech)] border border-dashed px-4 py-6 text-center text-[0.78rem] transition-colors ${
          dragging ? "border-accent/70 text-ink" : "border-line text-ink-dim hover:border-accent/40"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files) handleFiles(e.target.files);
            e.target.value = "";
          }}
        />
        <div className="font-display text-2xl text-accent/70">+</div>
        drop drawings here or click to add
        <div className="tech-label mt-1">multiple at once · png · jpeg</div>
      </div>
    </aside>
  );
}

function ArticleCard({
  article,
  selected,
  onSelect,
  onRemove,
}: {
  article: Article;
  selected: boolean;
  onSelect: () => void;
  onRemove: () => void;
}) {
  return (
    <div
      onClick={onSelect}
      className={`panel group relative cursor-pointer p-2.5 transition-colors ${
        selected ? "border-accent/70" : "hover:border-line-bright"
      }`}
    >
      <div className="flex items-center gap-2.5">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={article.thumbUrl}
          alt={article.name}
          className="h-11 w-11 shrink-0 rounded-[var(--radius-tech)] border border-line bg-[#0c1016] object-cover"
        />
        <div className="min-w-0 flex-1">
          <div className="truncate text-[0.8rem] text-ink" title={article.name}>
            {article.name}
          </div>
          <div className="mt-1.5">
            <ReadingPipeline status={article.status} startedAt={article.startedAt} compact />
          </div>
        </div>
        <span className={`tech-label ${statusColor(article)}`}>{statusLabel(article)}</span>
      </div>
      <button
        onClick={(e) => {
          e.stopPropagation();
          onRemove();
        }}
        aria-label={`remove ${article.name}`}
        className="absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full border border-line bg-[#0c1016] text-[0.7rem] text-ink-dim opacity-0 transition-all hover:border-bad/60 hover:text-bad group-hover:opacity-100"
      >
        ×
      </button>
    </div>
  );
}

function statusLabel(a: Article): string {
  if (a.status === "reading") return "reading";
  if (a.status === "error") return "error";
  return "ready";
}

function statusColor(a: Article): string {
  if (a.status === "reading") return "text-accent";
  if (a.status === "error") return "text-bad";
  return "text-good";
}
