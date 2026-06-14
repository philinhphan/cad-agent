// Shared score → color/label helpers (0-10 critic scale).

import type { CheckReport } from "./types";

export type ScoreTone = "bad" | "warn" | "good";

/** A critical deterministic check failed (advisory) — surfaced as a disagreement flag. */
export function hasCriticalFailure(report: CheckReport | null | undefined): boolean {
  return !!report?.checks.some((c) => c.critical && c.status === "fail");
}

export function scoreTone(score: number, threshold = 8): ScoreTone {
  if (score >= threshold) return "good";
  if (score >= 5) return "warn";
  return "bad";
}

export const toneColor: Record<ScoreTone, string> = {
  bad: "var(--color-bad)",
  warn: "var(--color-warn)",
  good: "var(--color-good)",
};

export const toneTextClass: Record<ScoreTone, string> = {
  bad: "text-bad",
  warn: "text-warn",
  good: "text-good",
};

export function fmtBbox(bbox: [number, number, number]): string {
  return bbox.map((v) => v.toFixed(1)).join(" × ");
}

export function fmtVolume(mm3: number): string {
  return `${Math.round(mm3).toLocaleString()} mm³`;
}
