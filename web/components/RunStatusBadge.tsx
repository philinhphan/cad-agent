import type { RunStatus } from "@/lib/useRunStream";

export function RunStatusBadge({
  status,
  accepted,
}: {
  status: RunStatus;
  accepted: boolean | null;
}) {
  if (status === "done") {
    const ok = accepted === true;
    return (
      <Chip
        color={ok ? "var(--color-good)" : "var(--color-warn)"}
        label={ok ? "accepted" : "best effort"}
      />
    );
  }
  if (status === "error") return <Chip color="var(--color-bad)" label="error" />;

  return (
    <Chip
      color="var(--color-accent)"
      label={status === "connecting" ? "connecting" : "generating"}
      live
    />
  );
}

function Chip({ color, label, live }: { color: string; label: string; live?: boolean }) {
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[0.7rem] uppercase tracking-[0.15em]"
      style={{
        color,
        borderColor: `color-mix(in srgb, ${color} 40%, transparent)`,
        background: `color-mix(in srgb, ${color} 10%, transparent)`,
      }}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${live ? "live-dot" : ""}`}
        style={{ background: color }}
      />
      {label}
    </span>
  );
}
