"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "motion/react";
import { startRun } from "@/lib/api";
import type { RunConfigInput } from "@/lib/types";

const EXAMPLES = [
  "a 40mm cube with a 10mm diameter centered through-hole",
  "rectangular mounting bracket 60x40x8mm with 4x M4 clearance holes (4.5mm) inset 6mm from corners, 3mm filleted vertical edges",
  "a hexagonal nut, 19mm across flats, 8mm thick, with a 10.5mm through-hole",
];

const DEFAULTS = {
  model: "openai:gpt-5.2",
  critic_model: "",
  max_iterations: 5,
  score_threshold: 8,
  exec_timeout_s: 60,
};

export function SpecForm() {
  const router = useRouter();
  const [spec, setSpec] = useState("");
  const [cfg, setCfg] = useState(DEFAULTS);
  const [advanced, setAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit() {
    if (!spec.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    const config: RunConfigInput = {
      model: cfg.model,
      critic_model: cfg.critic_model.trim() || null,
      max_iterations: cfg.max_iterations,
      score_threshold: cfg.score_threshold,
      exec_timeout_s: cfg.exec_timeout_s,
    };
    try {
      const { run_id } = await startRun(spec.trim(), config);
      router.push(`/runs/${run_id}`);
    } catch (e) {
      setError(
        `${e instanceof Error ? e.message : "request failed"} — is the backend running on ${process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"}?`,
      );
      setSubmitting(false);
    }
  }

  return (
    <div className="panel panel-ticks p-5 sm:p-6">
      <label className="tech-label">specification</label>
      <textarea
        value={spec}
        onChange={(e) => setSpec(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") submit();
        }}
        rows={4}
        placeholder="describe the part in plain language — dimensions in mm, features, holes, fillets…"
        className="mt-2 w-full resize-y rounded-[var(--radius-tech)] border border-line bg-[#0c1016] p-3.5 text-[0.95rem] leading-relaxed text-ink outline-none transition-colors placeholder:text-ink-faint focus:border-accent/60"
      />

      <div className="mt-2 flex flex-wrap gap-1.5">
        <span className="tech-label mr-1 self-center">try</span>
        {EXAMPLES.map((ex, i) => (
          <button
            key={i}
            onClick={() => setSpec(ex)}
            className="rounded-full border border-line px-2.5 py-1 text-[0.7rem] text-ink-dim transition-colors hover:border-accent/50 hover:text-ink"
          >
            {ex.split(",")[0].slice(0, 38)}…
          </button>
        ))}
      </div>

      {/* primary controls */}
      <div className="mt-5 grid grid-cols-2 gap-4 sm:grid-cols-[1fr_1fr_auto]">
        <NumberField
          label="iterations"
          value={cfg.max_iterations}
          min={1}
          max={12}
          onChange={(v) => setCfg({ ...cfg, max_iterations: v })}
        />
        <div>
          <div className="flex items-baseline justify-between">
            <label className="tech-label">accept ≥</label>
            <span className="font-display text-lg text-accent">{cfg.score_threshold}</span>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            value={cfg.score_threshold}
            onChange={(e) => setCfg({ ...cfg, score_threshold: Number(e.target.value) })}
            className="mt-2 w-full accent-[var(--color-accent)]"
          />
        </div>
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={submit}
          disabled={!spec.trim() || submitting}
          className="self-end rounded-[var(--radius-tech)] bg-accent px-6 py-2.5 font-display text-base font-semibold uppercase tracking-wider text-[#1a0e05] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting ? "starting…" : "Generate ▸"}
        </motion.button>
      </div>

      {/* advanced */}
      <button
        onClick={() => setAdvanced((a) => !a)}
        className="mt-4 tech-label transition-colors hover:text-ink-dim"
      >
        {advanced ? "▾" : "▸"} model & sandbox
      </button>
      {advanced && (
        <div className="mt-3 grid gap-4 border-t border-line pt-4 sm:grid-cols-3">
          <TextField
            label="model"
            value={cfg.model}
            onChange={(v) => setCfg({ ...cfg, model: v })}
          />
          <TextField
            label="critic model"
            placeholder="(same as model)"
            value={cfg.critic_model}
            onChange={(v) => setCfg({ ...cfg, critic_model: v })}
          />
          <NumberField
            label="exec timeout (s)"
            value={cfg.exec_timeout_s}
            min={10}
            max={600}
            onChange={(v) => setCfg({ ...cfg, exec_timeout_s: v })}
          />
        </div>
      )}

      {error && (
        <p className="mt-4 rounded-[var(--radius-tech)] border border-bad/40 bg-bad/10 px-3 py-2 text-[0.8rem] text-bad">
          {error}
        </p>
      )}
      <p className="mt-3 tech-label">⌘/ctrl + enter to run</p>
    </div>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <label className="tech-label">{label}</label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-2 w-full rounded-[var(--radius-tech)] border border-line bg-[#0c1016] px-3 py-2 text-ink outline-none focus:border-accent/60"
      />
    </div>
  );
}

function TextField({
  label,
  value,
  placeholder,
  onChange,
}: {
  label: string;
  value: string;
  placeholder?: string;
  onChange: (v: string) => void;
}) {
  return (
    <div>
      <label className="tech-label">{label}</label>
      <input
        type="text"
        value={value}
        placeholder={placeholder}
        onChange={(e) => onChange(e.target.value)}
        className="mt-2 w-full rounded-[var(--radius-tech)] border border-line bg-[#0c1016] px-3 py-2 text-[0.85rem] text-ink outline-none placeholder:text-ink-faint focus:border-accent/60"
      />
    </div>
  );
}
