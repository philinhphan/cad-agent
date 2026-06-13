import { scoreTone, toneColor } from "@/lib/score";

/** A compact instrument-style score readout: NN/10 with a tone color + gauge. */
export function ScoreBadge({
  score,
  threshold = 8,
  size = "md",
}: {
  score: number;
  threshold?: number;
  size?: "sm" | "md" | "lg";
}) {
  const tone = scoreTone(score, threshold);
  const color = toneColor[tone];
  const dims =
    size === "lg"
      ? "text-3xl px-3 py-1.5"
      : size === "sm"
        ? "text-sm px-1.5 py-0.5"
        : "text-xl px-2 py-1";

  return (
    <span
      className={`inline-flex items-baseline gap-0.5 rounded-[var(--radius-tech)] border tabular-nums font-display font-semibold leading-none ${dims}`}
      style={{
        color,
        borderColor: `color-mix(in srgb, ${color} 45%, transparent)`,
        background: `color-mix(in srgb, ${color} 12%, transparent)`,
      }}
      title={`critic score ${score}/10 (accept ≥ ${threshold})`}
    >
      {score}
      <span className="text-ink-faint text-[0.6em] font-normal">/10</span>
    </span>
  );
}

/** Horizontal 0-10 gauge with a threshold tick. */
export function ScoreGauge({ score, threshold = 8 }: { score: number; threshold?: number }) {
  const tone = scoreTone(score, threshold);
  return (
    <div className="relative h-1.5 w-full overflow-hidden rounded-full bg-panel-2">
      <div
        className="h-full rounded-full transition-[width] duration-500"
        style={{ width: `${score * 10}%`, background: toneColor[tone] }}
      />
      <div
        className="absolute top-[-2px] h-[10px] w-px bg-ink-faint"
        style={{ left: `${threshold * 10}%` }}
        title={`accept threshold ${threshold}`}
      />
    </div>
  );
}
