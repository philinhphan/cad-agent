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
  const { iterations, status, accepted, bestIndex, runId, drawings, interpretation } = state;
  const best = iterations.find((it) => it.record.index === bestIndex);
  const bestScore = best?.record.critique?.score ?? null;
  const inputUrl = (name: string) => apiUrl(`/api/runs/${runId}/input/${name}`);

  return (
    <div className="w-full px-6 py-8 lg:px-10">
      {/* header */}
      <div className="panel panel-ticks p-5">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="tech-label mb-1.5">specification</div>
            <p className="text-[1.05rem] leading-snug text-ink">
              {state.spec?.trim() ? state.spec : "from drawing"}
            </p>
          </div>
          <RunStatusBadge status={status} accepted={accepted} />
        </div>

        {runId && drawings.length > 0 && (
          <div className="mt-4 border-t border-line pt-4">
            <div className="tech-label mb-2">input drawing{drawings.length > 1 ? "s" : ""}</div>
            <div className="flex flex-wrap gap-2.5">
              {drawings.map((name) => (
                <a
                  key={name}
                  href={inputUrl(name)}
                  target="_blank"
                  rel="noreferrer"
                  className="block overflow-hidden rounded-[var(--radius-tech)] border border-line transition-colors hover:border-accent/60"
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={inputUrl(name)}
                    alt={name}
                    className="h-28 w-auto bg-[#0c1016] object-contain"
                  />
                </a>
              ))}
            </div>
            {interpretation && <Interpretation text={interpretation} />}
          </div>
        )}

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
            <div className="flex flex-1 flex-wrap items-center justify-end gap-2">
              {best?.urls.step && (
                <DownloadPrimary href={apiUrl(best.urls.step)} label="Download STEP" />
              )}
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

function Interpretation({ text }: { text: string }) {
  return (
    <details className="mt-3 rounded-[var(--radius-tech)] border border-line bg-[#0c1016]">
      <summary className="cursor-pointer px-3 py-2 tech-label transition-colors hover:text-ink">
        extracted dimensions
      </summary>
      <pre className="max-h-72 overflow-auto whitespace-pre-wrap border-t border-line px-3 py-2.5 text-[0.8rem] leading-relaxed text-ink-dim">
        {text}
      </pre>
    </details>
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

function DownloadPrimary({ href, label }: { href: string; label: string }) {
  return (
    <a
      href={href}
      download
      className="inline-flex items-center gap-2 rounded-full bg-accent px-5 py-2.5 font-display text-[0.85rem] font-semibold text-white shadow-lg transition-opacity hover:opacity-90"
    >
      <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
        <path d="M12 4v12m0 0l-4-4m4 4l4-4" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M4 18v1a1 1 0 001 1h14a1 1 0 001-1v-1" strokeLinecap="round" />
      </svg>
      {label}
    </a>
  );
}
