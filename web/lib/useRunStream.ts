"use client";

import { useEffect, useState } from "react";
import { subscribeRun } from "./api";
import type { IterationPayload, RunConfig, RunEvent } from "./types";

export type RunStatus = "connecting" | "running" | "done" | "error";

export interface RunStreamState {
  status: RunStatus;
  runId: string | null;
  spec: string | null;
  config: RunConfig | null;
  drawings: string[];
  interpretation: string | null;
  iterations: IterationPayload[];
  accepted: boolean | null;
  bestIndex: number | null;
  error: string | null;
}

const INITIAL: RunStreamState = {
  status: "connecting",
  runId: null,
  spec: null,
  config: null,
  drawings: [],
  interpretation: null,
  iterations: [],
  accepted: null,
  bestIndex: null,
  error: null,
};

function reduce(state: RunStreamState, event: RunEvent): RunStreamState {
  switch (event.type) {
    case "started":
      return {
        ...state,
        status: "running",
        runId: event.run_id,
        spec: event.spec,
        config: event.config,
        drawings: event.drawings ?? [],
        interpretation: event.interpretation ?? null,
      };
    case "iteration":
      return {
        ...state,
        status: "running",
        iterations: [...state.iterations, { record: event.record, urls: event.urls }],
      };
    case "result":
      return {
        ...state,
        status: "done",
        accepted: event.result.accepted,
        bestIndex: event.result.best.index,
        spec: state.spec ?? event.result.spec,
        drawings: state.drawings.length ? state.drawings : event.result.drawings ?? [],
        interpretation: state.interpretation ?? event.result.interpretation ?? null,
      };
    case "error":
      return { ...state, status: "error", error: event.message };
    default:
      return state;
  }
}

/** Subscribe to a live run's SSE stream and accumulate its iterations. */
export function useRunStream(runId: string): RunStreamState {
  const [state, setState] = useState<RunStreamState>(INITIAL);

  useEffect(() => {
    setState(INITIAL);
    const unsubscribe = subscribeRun(
      runId,
      (event) => setState((prev) => reduce(prev, event)),
      () =>
        setState((prev) =>
          prev.status === "done"
            ? prev
            : { ...prev, status: "error", error: "connection to backend lost" },
        ),
    );
    return unsubscribe;
  }, [runId]);

  return state;
}

/** Adapt a static (already-finished) run into the same shape the live view uses. */
export function staticRunState(
  iterations: IterationPayload[],
  accepted: boolean,
  bestIndex: number,
  spec: string,
  runId: string,
  drawings: string[] = [],
  interpretation: string | null = null,
): RunStreamState {
  return {
    status: "done",
    runId,
    spec,
    config: null,
    drawings,
    interpretation,
    iterations,
    accepted,
    bestIndex,
    error: null,
  };
}
