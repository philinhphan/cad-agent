"use client";

import { apiUrl } from "@/lib/api";
import type { RunStreamState } from "@/lib/useRunStream";
import { ScoreBadge } from "./ScoreBadge";
import { RunStatusBadge } from "./RunStatusBadge";
import { IterationCard } from "./IterationCard";

export function RunView({
  state,
  threshold,
}: {
  state: RunStreamState;
  threshold: number;
}) {
  const { iterations, status, accepted, bestIndex } = state;
  const best = iterations.find((it) => it.record.index === bestIndex);
  const bestScore = best?.record.critique?.score ?? null;

  return (
    <div className="mx-auto max-w-[1100px] px-5 py-8">
      {/* header */}
      <div className="panel panel-ticks p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="tech-label mb-1.5">specification</div>
            <p className="text-[1.05rem] leading-snug text-ink">
              {state.spec ?? "—"}
            </p>
          </div>
          <RunStatusBadge status={status} accepted={accepted} />
        </div>

        {status === "done" && (
          <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-3 border-t border-line pt-4">
            <div className="flex items-center gap-2.5">
              <span className="tech-label">best</span>
              {bestScore !== null ? (
                <ScoreBadge score={bestScore} threshold={threshold} size="lg" />
              ) : (
                <span className="text-ink-dim">—</span>
              )}
              {bestIndex !== null && (
                <span className="tech-label">iter {String(bestIndex).padStart(2, "0")}</span>
              )}
            </div>
            <div className="flex flex-1 flex-wrap justify-end gap-2">
              {best?.urls.step && <Download href={apiUrl(best.urls.step)} label="STEP" />}
              {best?.urls.stl && <Download href={apiUrl(best.urls.stl)} label="STL" />}
            </div>
          </div>
        )}
      </div>

      {/* iterations */}
      <div className="mt-5 space-y-4">
        {iterations.map((payload) => (
          <IterationCard
            key={payload.record.index}
            payload={payload}
            threshold={threshold}
            isBest={status === "done" && payload.record.index === bestIndex}
          />
        ))}

        {status !== "error" && (status === "connecting" || status === "running") && (
          <Pending hasAny={iterations.length > 0} />
        )}
        {status === "error" && iterations.length === 0 && (
          <p className="panel p-5 text-bad">
            {state.error ?? "this run could not be loaded."}
          </p>
        )}
      </div>
    </div>
  );
}

function Pending({ hasAny }: { hasAny: boolean }) {
  return (
    <div className="panel flex items-center gap-3 p-5 text-ink-dim">
      <span className="live-dot h-2 w-2 rounded-full bg-accent" />
      <span className="tech-label">
        {hasAny ? "refining — next iteration in progress" : "warming up · first iteration runs in ~10–40s"}
      </span>
    </div>
  );
}

function Download({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      download
      className="rounded-[var(--radius-tech)] border border-line px-3 py-1.5 text-[0.75rem] uppercase tracking-wider text-ink-dim transition-colors hover:border-accent/50 hover:text-ink"
    >
      ↓ {label}
    </a>
  );
}
