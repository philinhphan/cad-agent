// Client for the cad-gen FastAPI backend.

import type {
  RunConfigInput,
  RunDetail,
  RunEvent,
  RunSummary,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

export function apiUrl(path: string): string {
  return `${API_BASE}${path}`;
}

async function jsonOrThrow<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const detail = await resp.text().catch(() => resp.statusText);
    throw new Error(`${resp.status} ${detail}`);
  }
  return resp.json() as Promise<T>;
}

export async function startRun(
  spec: string,
  config: RunConfigInput,
): Promise<{ run_id: string }> {
  const resp = await fetch(apiUrl("/api/runs"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ spec, config }),
  });
  return jsonOrThrow(resp);
}

export async function listRuns(): Promise<RunSummary[]> {
  return jsonOrThrow(await fetch(apiUrl("/api/runs"), { cache: "no-store" }));
}

export async function getRun(id: string): Promise<RunDetail> {
  return jsonOrThrow(
    await fetch(apiUrl(`/api/runs/${id}`), { cache: "no-store" }),
  );
}

/**
 * Subscribe to a run's live event stream. Returns an unsubscribe function.
 * The stream closes itself after the terminal `result`/`error` event.
 */
export function subscribeRun(
  runId: string,
  onEvent: (event: RunEvent) => void,
  onError?: (err: Event) => void,
): () => void {
  const source = new EventSource(apiUrl(`/api/runs/${runId}/events`));

  source.onmessage = (msg) => {
    let event: RunEvent;
    try {
      event = JSON.parse(msg.data) as RunEvent;
    } catch {
      return;
    }
    onEvent(event);
    if (event.type === "result" || event.type === "error") {
      source.close();
    }
  };

  source.onerror = (err) => {
    // Normal completion closes the stream via close() above and fires no error.
    // A fatal failure (e.g. 404 for a non-live run, or backend down) leaves the
    // connection CLOSED — surface that so the caller can fall back to REST.
    // Transient drops (readyState CONNECTING) auto-reconnect; ignore them.
    if (source.readyState === EventSource.CLOSED) onError?.(err);
  };

  return () => source.close();
}
