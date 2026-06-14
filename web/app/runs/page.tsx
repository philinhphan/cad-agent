"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { listRuns } from "@/lib/api";
import { ScoreBadge } from "@/components/ScoreBadge";
import type { RunSummary } from "@/lib/types";

type LoadState =
  | { status: "loading" }
  | { status: "error"; error: string }
  | { status: "ready"; runs: RunSummary[] };

export default function RunsHistoryPage() {
  const [state, setState] = useState<LoadState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;
    listRuns()
      .then((runs) => !cancelled && setState({ status: "ready", runs }))
      .catch(
        (e) =>
          !cancelled &&
          setState({ status: "error", error: e?.message ?? "failed to load runs" }),
      );
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="mx-auto max-w-[1100px] px-5 py-8">
      <div className="mb-6 flex items-baseline justify-between">
        <h1 className="tech-head text-3xl text-ink">Run history</h1>
        <Link
          href="/"
          className="tech-label transition-colors hover:text-ink"
        >
          + new run
        </Link>
      </div>

      {state.status === "loading" && (
        <p className="panel flex items-center gap-3 p-5 text-ink-dim">
          <span className="live-dot h-2 w-2 rounded-full bg-accent" />
          <span className="tech-label">loading runs…</span>
        </p>
      )}

      {state.status === "error" && (
        <p className="panel p-5 text-bad">
          {state.error} — is the backend running on the configured API base?
        </p>
      )}

      {state.status === "ready" && state.runs.length === 0 && (
        <div className="panel p-8 text-center">
          <p className="text-ink-dim">No runs yet.</p>
          <Link
            href="/"
            className="mt-3 inline-block tech-label text-accent transition-opacity hover:opacity-80"
          >
            generate your first part →
          </Link>
        </div>
      )}

      {state.status === "ready" && state.runs.length > 0 && (
        <ul className="space-y-3">
          {state.runs.map((run) => (
            <li key={run.id}>
              <Link
                href={`/runs/${run.id}`}
                className="panel panel-ticks flex items-center gap-4 p-4 transition-colors hover:border-accent/50"
              >
                <ScoreBadge score={run.score} />
                <div className="min-w-0 flex-1">
                  <p className="truncate text-[1rem] text-ink">
                    {run.spec?.trim()
                      ? run.spec
                      : (run.drawings?.length ?? 0) > 0
                        ? "from drawing"
                        : "(no spec)"}
                  </p>
                  <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 tech-label">
                    <span>{fmtDate(run.created_at)}</span>
                    <span>·</span>
                    <span>
                      {run.n_iterations} iter
                      {run.n_iterations === 1 ? "" : "s"}
                    </span>
                    {(run.drawings?.length ?? 0) > 0 && (
                      <>
                        <span>·</span>
                        <span>
                          {run.drawings!.length} drawing
                          {run.drawings!.length === 1 ? "" : "s"}
                        </span>
                      </>
                    )}
                  </div>
                </div>
                <span
                  className={`tech-label uppercase tracking-wider ${
                    run.accepted ? "text-good" : "text-warn"
                  }`}
                >
                  {run.accepted ? "accepted" : "best effort"}
                </span>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function fmtDate(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
