"use client";

import { use, useEffect, useState } from "react";
import { RunView } from "@/components/RunView";
import { getRun } from "@/lib/api";
import {
  staticRunState,
  useRunStream,
  type RunStreamState,
} from "@/lib/useRunStream";

export default function RunPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = use(params);
  return <RunPageContent key={id} runId={id} />;
}

function RunPageContent({ runId }: { runId: string }) {
  const live = useRunStream(runId);
  const [fallback, setFallback] = useState<RunStreamState | null>(null);

  // A finished/history run has no live SSE stream (no handle in the backend registry, e.g.
  // after a restart), so the stream errors with no iterations. Fall back to the static REST
  // detail so past runs still render instead of showing "could not be loaded".
  useEffect(() => {
    if (live.status !== "error" || live.iterations.length > 0) return;
    let cancelled = false;
    getRun(runId)
      .then((d) => {
        if (cancelled) return;
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
        );
      })
      .catch(() => {
        /* leave the live error state to surface in RunView */
      });
    return () => {
      cancelled = true;
    };
  }, [live.status, live.iterations.length, runId]);

  const state = fallback ?? live;
  const threshold = state.config?.score_threshold ?? 8;
  return <RunView state={state} threshold={threshold} />;
}
