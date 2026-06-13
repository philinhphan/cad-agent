import type { Critique } from "@/lib/types";

/** Renders the vision critic's structured verdict for one iteration. */
export function CritiquePanel({ critique }: { critique: Critique }) {
  return (
    <div className="space-y-3 text-[0.82rem] leading-relaxed">
      <p className="text-ink-dim">
        <span className="tech-label mr-2 align-middle">verdict</span>
        {critique.summary}
      </p>

      {critique.issues.length > 0 && (
        <Section label="issues" tone="bad" items={critique.issues} />
      )}
      {critique.suggestions.length > 0 && (
        <Section label="suggestions" tone="viewport" items={critique.suggestions} />
      )}
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
