"use client";

import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { PIPELINE_STAGES, activeStageIndex, type ArticleStatus } from "@/lib/article";

/** A live elapsed-ms clock that ticks only while a read is in flight. */
function useElapsed(active: boolean, startedAt: number): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const id = setInterval(() => setNow(Date.now()), 250);
    return () => clearInterval(id);
  }, [active, startedAt]);
  return Math.max(0, now - startedAt);
}

export function ReadingPipeline({
  status,
  startedAt,
  compact = false,
}: {
  status: ArticleStatus;
  startedAt: number;
  compact?: boolean;
}) {
  const reading = status === "reading";
  const elapsed = useElapsed(reading, startedAt);
  const active = activeStageIndex(status, elapsed);
  const failed = status === "error";

  if (compact) {
    return (
      <div className="flex items-center gap-1.5">
        {PIPELINE_STAGES.map((s, i) => {
          const state = stageState(i, active, status, failed);
          return (
            <span
              key={s.key}
              className={`h-1.5 w-1.5 rounded-full ${dotClass(state)} ${
                state === "active" ? "live-dot" : ""
              }`}
            />
          );
        })}
      </div>
    );
  }

  return (
    <div className="panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <span className="tech-label">reading pipeline</span>
        <span className="tech-label">
          {failed ? "failed" : status === "ready" ? "complete" : `${(elapsed / 1000).toFixed(1)}s`}
        </span>
      </div>
      <div className="flex items-center">
        {PIPELINE_STAGES.map((s, i) => {
          const state = stageState(i, active, status, failed);
          return (
            <div key={s.key} className="flex flex-1 items-center last:flex-none">
              <div className="flex flex-col items-center gap-1.5">
                <Node state={state} index={i} />
                <span
                  className={`whitespace-nowrap text-[0.62rem] uppercase tracking-wider ${
                    state === "pending" ? "text-ink-faint" : "text-ink-dim"
                  }`}
                >
                  {s.label}
                </span>
              </div>
              {i < PIPELINE_STAGES.length - 1 && (
                <Connector done={i < active || status === "ready"} failed={failed && i === active} />
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

type NodeState = "done" | "active" | "pending" | "failed";

function stageState(
  i: number,
  active: number,
  status: ArticleStatus,
  failed: boolean,
): NodeState {
  if (status === "ready") return "done";
  if (failed) {
    if (i < active) return "done";
    if (i === active) return "failed";
    return "pending";
  }
  if (i < active) return "done";
  if (i === active) return "active";
  return "pending";
}

function Node({ state, index }: { state: NodeState; index: number }) {
  const base =
    "relative flex h-8 w-8 items-center justify-center rounded-full border text-[0.7rem] font-display";
  if (state === "done") {
    return (
      <div className={`${base} border-accent bg-accent text-white`}>
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="3">
          <path d="M5 13l4 4L19 7" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>
    );
  }
  if (state === "failed") {
    return <div className={`${base} border-bad bg-bad/15 text-bad`}>×</div>;
  }
  if (state === "active") {
    return (
      <div className={`${base} border-accent text-accent`}>
        <motion.span
          className="absolute inset-0 rounded-full border border-accent"
          animate={{ scale: [1, 1.45], opacity: [0.7, 0] }}
          transition={{ duration: 1.1, repeat: Infinity, ease: "easeOut" }}
        />
        {index + 1}
      </div>
    );
  }
  return <div className={`${base} border-line text-ink-faint`}>{index + 1}</div>;
}

function Connector({ done, failed }: { done: boolean; failed: boolean }) {
  return (
    <div className="mx-1 h-px flex-1 overflow-hidden bg-line">
      <motion.div
        className={`h-full ${failed ? "bg-bad" : "bg-accent"}`}
        initial={false}
        animate={{ width: done ? "100%" : "0%" }}
        transition={{ duration: 0.5, ease: "easeOut" }}
      />
    </div>
  );
}

function dotClass(state: NodeState): string {
  switch (state) {
    case "done":
      return "bg-accent";
    case "active":
      return "bg-accent";
    case "failed":
      return "bg-bad";
    default:
      return "bg-line";
  }
}
