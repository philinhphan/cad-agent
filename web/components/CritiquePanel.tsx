import type { ChecklistItem, Critique } from "@/lib/types";

/** Renders the vision critic's structured verdict for one iteration. */
export function CritiquePanel({ critique }: { critique: Critique }) {
  const subs = [
    { label: "dimensional", value: critique.dimensional_score },
    { label: "features", value: critique.feature_completeness_score },
    { label: "proportion", value: critique.proportion_score },
  ].filter((s) => s.value != null);

  return (
    <div className="space-y-3 text-[0.82rem] leading-relaxed">
      <p className="text-ink-dim">
        <span className="tech-label mr-2 align-middle">verdict</span>
        {critique.summary}
      </p>

      {subs.length > 0 && (
        <div className="flex flex-wrap gap-x-4 gap-y-1">
          {subs.map((s) => (
            <span key={s.label} className="inline-flex items-baseline gap-1.5">
              <span className="tech-label">{s.label}</span>
              <span className="tabular-nums text-ink">{s.value}/10</span>
            </span>
          ))}
        </div>
      )}

      {critique.checklist.length > 0 && <Checklist items={critique.checklist} />}

      {critique.issues.length > 0 && (
        <Section label="issues" tone="bad" items={critique.issues} />
      )}
      {critique.suggestions.length > 0 && (
        <Section label="suggestions" tone="viewport" items={critique.suggestions} />
      )}
    </div>
  );
}

const STATUS: Record<string, { color: string; mark: string }> = {
  pass: { color: "var(--color-good)", mark: "✓" },
  fail: { color: "var(--color-bad)", mark: "✕" },
  uncertain: { color: "var(--color-warn)", mark: "?" },
};

function Checklist({ items }: { items: ChecklistItem[] }) {
  return (
    <div>
      <div className="tech-label mb-1">requirement checklist</div>
      <ul className="space-y-1">
        {items.map((it, i) => {
          const s = STATUS[it.status] ?? STATUS.uncertain;
          return (
            <li key={i} className="flex gap-2 text-ink">
              <span style={{ color: s.color }} className="mt-px select-none">
                {s.mark}
              </span>
              <span>
                {it.requirement}
                {it.observed && (
                  <span className="text-ink-dim">
                    {" "}
                    — {it.observed}
                    {it.target ? ` (target ${it.target})` : ""}
                  </span>
                )}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Section({
  label,
  tone,
  items,
}: {
  label: string;
  tone: "bad" | "viewport";
  items: string[];
}) {
  const color = tone === "bad" ? "var(--color-bad)" : "var(--color-viewport)";
  return (
    <div>
      <div className="tech-label mb-1">{label}</div>
      <ul className="space-y-1">
        {items.map((item, i) => (
          <li key={i} className="flex gap-2 text-ink">
            <span style={{ color }} className="mt-px select-none">
              {tone === "bad" ? "✕" : "→"}
            </span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
