"use client";

import { useState } from "react";
import { motion } from "motion/react";
import { apiUrl } from "@/lib/api";
import { fmtBbox, fmtVolume } from "@/lib/score";
import type { IterationPayload } from "@/lib/types";
import { ScoreBadge } from "./ScoreBadge";
import { CritiquePanel } from "./CritiquePanel";
import { CodeBlock } from "./CodeBlock";
import { StlViewer } from "./StlViewer";

type Tab = "3d" | "renders" | "code";

export function IterationCard({
  payload,
  threshold,
  isBest,
}: {
  payload: IterationPayload;
  threshold: number;
  isBest: boolean;
}) {
  const { record, urls } = payload;
  const [tab, setTab] = useState<Tab>("3d");
  const success = record.execution?.success ?? false;
  const metrics = record.execution?.metrics ?? null;

  return (
    <motion.article
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className="panel panel-ticks"
      style={isBest ? { borderColor: "color-mix(in srgb, var(--color-good) 55%, transparent)" } : undefined}
    >
      {/* header */}
      <div className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <div className="flex items-center gap-3">
          <span className="tech-head text-lg text-ink">
            ITER<span className="text-accent">·</span>
            {String(record.index).padStart(2, "0")}
          </span>
          {success ? (
            <span className="tech-label text-good">● executed</span>
          ) : (
            <span className="tech-label text-bad">✕ build failed</span>
          )}
          {isBest && (
            <span className="rounded-[var(--radius-tech)] border border-good/40 bg-good/10 px-1.5 py-0.5 text-[0.62rem] uppercase tracking-widest text-good">
              best
            </span>
          )}
        </div>
        {record.critique && <ScoreBadge score={record.critique.score} threshold={threshold} />}
      </div>

      {/* metrics strip */}
      {metrics && (
        <div className="flex flex-wrap gap-x-5 gap-y-1 border-b border-line px-4 py-2 text-[0.72rem] text-ink-dim">
          <Metric label="bbox" value={`${fmtBbox(metrics.bbox_mm)} mm`} />
          <Metric label="vol" value={fmtVolume(metrics.volume_mm3)} />
          <Metric label="solids" value={String(metrics.n_solids)} />
          <Metric
            label="watertight"
            value={metrics.is_watertight === null ? "?" : metrics.is_watertight ? "yes" : "NO"}
            warn={metrics.is_watertight === false}
          />
          {record.execution && (
            <Metric label="t" value={`${record.execution.duration_s.toFixed(1)}s`} />
          )}
        </div>
      )}

      {/* body */}
      {success ? (
        <div className="grid gap-4 p-4 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
          <div className="flex flex-col">
            <TabBar tab={tab} setTab={setTab} hasViews={!!urls.views} />
            <div className="relative h-[340px] overflow-hidden rounded-b-[var(--radius-tech)] border border-t-0 border-line bg-[#0c1016]">
              {tab === "3d" && urls.stl && <StlViewer url={apiUrl(urls.stl)} />}
              {tab === "renders" &&
                (urls.views ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={apiUrl(urls.views)}
                    alt={`rendered views, iteration ${record.index}`}
                    className="h-full w-full object-contain"
                  />
                ) : (
                  <Empty>no render</Empty>
                ))}
              {tab === "code" && (
                <div className="h-full overflow-auto">
                  <CodeBlock code={record.execution!.code} />
                </div>
              )}
            </div>
          </div>

          <div className="min-w-0">
            {record.critique ? (
              <CritiquePanel critique={record.critique} />
            ) : (
              <p className="tech-label">awaiting critique…</p>
            )}
          </div>
        </div>
      ) : (
        <div className="space-y-2 p-4">
          <div className="tech-label text-bad">traceback</div>
          <pre className="max-h-56 overflow-auto rounded-[var(--radius-tech)] border border-line bg-[#0c1016] p-3 text-[0.72rem] leading-relaxed text-bad/90">
            {record.execution?.error ?? "no code was produced"}
          </pre>
        </div>
      )}
    </motion.article>
  );
}

function TabBar({
  tab,
  setTab,
  hasViews,
}: {
  tab: Tab;
  setTab: (t: Tab) => void;
  hasViews: boolean;
}) {
  const tabs: { id: Tab; label: string }[] = [
    { id: "3d", label: "3D" },
    ...(hasViews ? [{ id: "renders" as Tab, label: "Renders" }] : []),
    { id: "code", label: "Code" },
  ];
  return (
    <div className="flex gap-px overflow-hidden rounded-t-[var(--radius-tech)] border border-line bg-line text-[0.7rem] uppercase tracking-widest">
      {tabs.map((t) => (
        <button
          key={t.id}
          onClick={() => setTab(t.id)}
          className={`flex-1 px-3 py-1.5 transition-colors ${
            tab === t.id
              ? "bg-panel-2 text-accent"
              : "bg-panel text-ink-faint hover:text-ink-dim"
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

function Metric({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <span className="inline-flex items-baseline gap-1.5">
      <span className="tech-label">{label}</span>
      <span className={`tabular-nums ${warn ? "text-bad" : "text-ink"}`}>{value}</span>
    </span>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-full items-center justify-center tech-label">{children}</div>
  );
}
