"use client";

import { motion } from "motion/react";
import { ReadingPipeline } from "@/components/ReadingPipeline";
import type { Article } from "@/lib/article";

export function ArticleDetail({
  article,
  onChangeInterpretation,
  onChangeNotes,
  onRetry,
  onGenerate,
  generating,
}: {
  article: Article | null;
  onChangeInterpretation: (v: string) => void;
  onChangeNotes: (v: string) => void;
  onRetry: () => void;
  onGenerate: () => void;
  generating: boolean;
}) {
  if (!article) {
    return (
      <div className="panel panel-ticks flex min-h-[420px] flex-1 flex-col items-center justify-center p-8 text-center">
        <div className="tech-head text-2xl text-ink-dim">No article selected</div>
        <p className="mt-2 max-w-sm text-[0.85rem] text-ink-faint">
          Drop one or more engineering drawings into the sidebar. Each becomes an article and is read
          in parallel — watch the pipeline extract its dimensions.
        </p>
      </div>
    );
  }

  return (
    <div className="flex min-w-0 flex-1 flex-col gap-4">
      <div className="panel panel-ticks p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="tech-label">article</div>
            <h2 className="tech-head mt-1 truncate text-2xl text-ink" title={article.name}>
              {article.name}
            </h2>
          </div>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={article.thumbUrl}
            alt={article.name}
            className="h-24 w-auto max-w-[200px] rounded-[var(--radius-tech)] border border-line bg-[#0c1016] object-contain"
          />
        </div>

        <div className="mt-4">
          <ReadingPipeline status={article.status} startedAt={article.startedAt} />
        </div>
      </div>

      {article.status === "error" && (
        <div className="panel p-5">
          <p className="rounded-[var(--radius-tech)] border border-bad/40 bg-bad/10 px-3 py-2 text-[0.82rem] text-bad">
            {article.error ?? "reading failed"}
          </p>
          <button
            onClick={onRetry}
            className="mt-3 rounded-[var(--radius-tech)] border border-line px-4 py-2 text-[0.8rem] text-ink-dim transition-colors hover:border-accent/50 hover:text-ink"
          >
            ↻ retry reading
          </button>
        </div>
      )}

      {article.status === "reading" && (
        <div className="panel flex items-center gap-3 p-5 text-[0.85rem] text-ink-dim">
          <span className="live-dot h-2 w-2 rounded-full bg-accent" />
          Reading the drawing — extracting dimensions and features…
        </div>
      )}

      {article.status === "ready" && (
        <div className="panel panel-ticks p-5">
          <label className="tech-label">extracted dimensions — review &amp; edit</label>
          <p className="mt-1 text-[0.82rem] text-ink-dim">
            Correct anything the reader got wrong. The drawing image is still sent as the source of
            truth, so this only guides the build.
          </p>
          <textarea
            value={article.interpretation}
            onChange={(e) => onChangeInterpretation(e.target.value)}
            rows={14}
            className="mt-3 w-full resize-y rounded-[var(--radius-tech)] border border-line bg-[#0c1016] p-3.5 font-mono text-[0.82rem] leading-relaxed text-ink outline-none transition-colors focus:border-accent/60"
          />

          <label className="tech-label mt-4 block">additional notes (optional)</label>
          <textarea
            value={article.notes}
            onChange={(e) => onChangeNotes(e.target.value)}
            rows={2}
            placeholder="extra instructions for the generator — material, tolerances, intent…"
            className="mt-2 w-full resize-y rounded-[var(--radius-tech)] border border-line bg-[#0c1016] p-3.5 text-[0.9rem] leading-relaxed text-ink outline-none transition-colors placeholder:text-ink-faint focus:border-accent/60"
          />

          <div className="mt-4 flex items-center justify-between gap-3">
            <button
              onClick={onRetry}
              disabled={generating}
              className="tech-label transition-colors hover:text-ink disabled:opacity-40"
            >
              ↻ re-read drawing
            </button>
            <motion.button
              whileTap={{ scale: 0.97 }}
              onClick={onGenerate}
              disabled={generating}
              className="rounded-[var(--radius-tech)] bg-accent px-6 py-2.5 font-display font-semibold uppercase tracking-wider text-white transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {generating ? "starting…" : "Generate ▸"}
            </motion.button>
          </div>
        </div>
      )}
    </div>
  );
}
