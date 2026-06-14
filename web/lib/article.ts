// Client-side model for the article workspace (browser-only state).

export type ArticleStatus = "reading" | "ready" | "error";

export interface Article {
  id: string;
  file: File;
  name: string;
  thumbUrl: string;
  status: ArticleStatus;
  startedAt: number;
  interpretation: string;
  notes: string;
  error: string | null;
}

export const PIPELINE_STAGES = [
  { key: "upload", label: "Upload" },
  { key: "analyze", label: "Analyze" },
  { key: "extract", label: "Extract dims" },
  { key: "ready", label: "Ready" },
] as const;

// While a read is in flight the backend gives us no sub-steps, so we advance the
// displayed stage on a timer to convey progress. Thresholds in milliseconds.
export const STAGE_THRESHOLDS_MS = [0, 700, 4500];

/**
 * Index of the active pipeline stage given status + elapsed time.
 * - ready  → last stage (Ready) complete
 * - error  → fails at whatever stage was active
 * - reading→ derived from elapsed; caps at "Extract dims" until the call returns
 */
export function activeStageIndex(status: ArticleStatus, elapsedMs: number): number {
  if (status === "ready") return PIPELINE_STAGES.length - 1;
  let idx = 0;
  for (let i = 0; i < STAGE_THRESHOLDS_MS.length; i++) {
    if (elapsedMs >= STAGE_THRESHOLDS_MS[i]) idx = i;
  }
  // cap at "extract" (index 2) while still reading — "ready" only on success
  return Math.min(idx, PIPELINE_STAGES.length - 2);
}
