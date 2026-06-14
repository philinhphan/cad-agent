import type { Check, CheckReport } from "@/lib/types";

const TONE: Record<string, { color: string; mark: string }> = {
  pass: { color: "var(--color-good)", mark: "✓" },
  fail: { color: "var(--color-bad)", mark: "✕" },
  skip: { color: "#6b7280", mark: "–" },
};

/** Advisory deterministic-check strip (mass / envelope / topology). These never gate
 *  acceptance — they ground the critic and flag disagreements to the user. */
export function CheckReportPanel({ report }: { report: CheckReport }) {
  if (!report.checks.length) return null;
  return (
    <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 border-b border-line px-4 py-2 text-[0.72rem]">
      <span className="tech-label">checks</span>
      {report.checks.map((c) => (
        <CheckChip key={c.name} check={c} />
      ))}
    </div>
  );
}

function CheckChip({ check }: { check: Check }) {
  const tone = TONE[check.status] ?? TONE.skip;
  const showValue = check.status !== "skip" && check.observed != null;
  return (
    <span className="inline-flex items-baseline gap-1" title={check.message}>
      <span style={{ color: tone.color }} className="select-none">
        {tone.mark}
      </span>
      <span className="text-ink-dim">{check.name}</span>
      {showValue && (
        <span className="tabular-nums text-ink">
          {String(check.observed)}
          {check.target != null ? ` / ${String(check.target)}` : ""}
        </span>
      )}
    </span>
  );
}
