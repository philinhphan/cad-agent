"use client";

import { useEffect, useState } from "react";
import { getRun } from "@/lib/api";
import { useRunStream, staticRunState, type RunStreamState } from "@/lib/useRunStream";
import { RunView } from "@/components/RunView";

/**
 * Run detail page. Subscribes to the live SSE stream; if the run has already
 * finished (no longer in the backend's live registry, so the events endpoint
 * 404s) it falls back to the REST snapshot.
 */
export function RunPage({ runId }: { runId: string }) {
  const live = useRunStream(runId);
  const [fallback, setFallback] = useState<RunStreamState | null>(null);

  useEffect(() => {
    if (live.status === "error" && live.iterations.length === 0 && !fallback) {
      getRun(runId)
        .then((d) =>
          setFallback(
            staticRunState(
              d.iterations,
              d.accepted,
              d.best_index,
              d.spec,
              d.id,
              d.drawings ?? [],
              d.interpretation ?? null,
            ),
          ),
        )
        .catch(() => {
          /* keep the live error state */
        });
    }
  }, [live.status, live.iterations.length, fallback, runId]);

  const state = fallback ?? live;
  const threshold = state.config?.score_threshold ?? 8;

  return <RunView state={state} threshold={threshold} />;
}
