"use client";

import { useEffect, useState } from "react";
import { apiUrl, generateRunShowcase, getRunShowcase } from "@/lib/api";
import type { ShowcaseResponse } from "@/lib/types";
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
  const [showcase, setShowcase] = useState<ShowcaseResponse | null>(null);
  const [showcaseLoading, setShowcaseLoading] = useState(false);
  const [showcaseError, setShowcaseError] = useState<string | null>(null);

  useEffect(() => {
    if (status !== "done" || !runId) return;

    let cancelled = false;
    void getRunShowcase(runId)
      .then((result) => {
        if (!cancelled) setShowcase(result);
      })
      .catch(() => {
        // Cached showcase lookup is opportunistic; generation remains manual.
      });
    return () => {
      cancelled = true;
    };
  }, [runId, status]);

  async function handleShowcase() {
    if (!runId) return;
    setShowcaseLoading(true);
    setShowcaseError(null);
    try {
      setShowcase(await generateRunShowcase(runId));
    } catch (err) {
      setShowcaseError(cleanError(err));
    } finally {
      setShowcaseLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-[1100px] px-5 py-8">
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
            <div className="flex flex-1 flex-wrap justify-end gap-2">
              {best?.urls.step && <Download href={apiUrl(best.urls.step)} label="STEP" />}
              {best?.urls.stl && <Download href={apiUrl(best.urls.stl)} label="STL" />}
            </div>
          </div>
        )}

        {status === "done" && runId && (
          <div className="mt-4 border-t border-line pt-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <div className="tech-label mb-1">fal showcase</div>
                <div className="text-[0.8rem] text-ink-dim">
                  {showcase ? "image ready" : "optional final image"}
                </div>
              </div>
              {!showcase && (
                <button
                  type="button"
                  onClick={handleShowcase}
                  disabled={showcaseLoading || !best?.urls.views}
                  className="rounded-[var(--radius-tech)] border border-accent/50 bg-accent/10 px-3 py-1.5 text-[0.75rem] uppercase tracking-wider text-accent transition-colors hover:border-accent hover:bg-accent/15 disabled:cursor-not-allowed disabled:border-line disabled:bg-transparent disabled:text-ink-faint"
                >
                  {showcaseLoading ? "generating image" : "generate showcase"}
                </button>
              )}
            </div>
            {showcaseError && (
              <p className="mt-3 rounded-[var(--radius-tech)] border border-bad/40 bg-bad/10 px-3 py-2 text-[0.78rem] text-bad">
                {showcaseError}
              </p>
            )}
            {showcase && (
              <div className="mt-3 overflow-hidden rounded-[var(--radius-tech)] border border-line bg-[#0c1016]">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={showcase.image.url}
                  alt="fal.ai showcase render of the final CAD model"
                  className="max-h-[520px] w-full object-contain"
                />
                <div className="flex flex-wrap items-center justify-between gap-2 border-t border-line px-3 py-2">
                  <span className="tech-label">{showcase.model}</span>
                  <a
                    href={showcase.image.url}
                    target="_blank"
                    rel="noreferrer"
                    className="text-[0.75rem] uppercase tracking-wider text-ink-dim transition-colors hover:text-accent"
                  >
                    open image
                  </a>
                </div>
              </div>
            )}
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

function cleanError(err: unknown): string {
  if (err instanceof Error) {
    return err.message.replace(/^\d+\s*/, "");
  }
  return "could not generate showcase image";
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
