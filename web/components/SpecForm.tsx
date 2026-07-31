"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "motion/react";
import { getConfigDefaults, interpretDrawing, startRun } from "@/lib/api";
import type { CadLibrary, RunConfigInput } from "@/lib/types";

const EXAMPLES = [
  "a 40mm cube with a 10mm diameter centered through-hole",
  "rectangular mounting bracket 60x40x8mm with 4x M4 clearance holes (4.5mm) inset 6mm from corners, 3mm filleted vertical edges",
  "a hexagonal nut, 19mm across flats, 8mm thick, with a 10.5mm through-hole",
];

// model is seeded from the backend's env-resolved defaults on mount (see the
// useEffect below); empty until then, and an empty model field is omitted from
// the request so the server's default applies.
const DEFAULTS = {
  model: "",
  critic_model: "",
  // Unlike the model fields this has no env-resolved backend default to wait for —
  // it mirrors DEFAULT_LIBRARY in src/cad_gen/models.py.
  library: "cadquery" as CadLibrary,
  max_iterations: 5,
  score_threshold: 8,
  exec_timeout_s: 60,
};

const LIBRARY_OPTIONS: { value: CadLibrary; label: string }[] = [
  { value: "cadquery", label: "CadQuery" },
  { value: "build123d", label: "build123d" },
];

const CRITIC_FALLBACK = "google:gemini-3.5-flash";

const MAX_FILES = 5;
const ACCEPT = ["image/png", "image/jpeg"];

export function SpecForm() {
  const router = useRouter();
  const [spec, setSpec] = useState("");
  const [cfg, setCfg] = useState(DEFAULTS);
  const [criticDefault, setCriticDefault] = useState(CRITIC_FALLBACK);
  const [advanced, setAdvanced] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const [phase, setPhase] = useState<"compose" | "review">("compose");
  const [interpretation, setInterpretation] = useState("");
  const [interpreting, setInterpreting] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Seed the model fields from the backend's env-resolved defaults
  // (CAD_GEN_MODEL / CAD_GEN_CRITIC_MODEL). Falls back to the built-ins if the
  // backend is unreachable. Won't clobber a model the user already typed.
  useEffect(() => {
    let cancelled = false;
    getConfigDefaults()
      .then((d) => {
        if (cancelled) return;
        if (d.critic_model) setCriticDefault(d.critic_model);
        setCfg((prev) => (prev.model ? prev : { ...prev, model: d.model }));
      })
      .catch(() => {
        /* backend down — keep built-in placeholder defaults */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function buildConfig(): RunConfigInput {
    return {
      // empty model -> omit so the server's env-resolved default applies
      model: cfg.model.trim() || undefined,
      critic_model: cfg.critic_model.trim() || null,
      library: cfg.library,
      max_iterations: cfg.max_iterations,
      score_threshold: cfg.score_threshold,
      exec_timeout_s: cfg.exec_timeout_s,
    };
  }

  function addFiles(incoming: FileList | File[]) {
    const accepted = Array.from(incoming).filter((f) => ACCEPT.includes(f.type));
    if (accepted.length === 0) return;
    setFiles((prev) => [...prev, ...accepted].slice(0, MAX_FILES));
  }

  function reportError(e: unknown) {
    setError(
      `${e instanceof Error ? e.message : "request failed"} — is the backend running on ${process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"}?`,
    );
  }

  async function readDrawing() {
    if (files.length === 0 || interpreting) return;
    setInterpreting(true);
    setError(null);
    try {
      const { interpretation } = await interpretDrawing(spec.trim(), buildConfig(), files);
      setInterpretation(interpretation);
      setPhase("review");
    } catch (e) {
      reportError(e);
    } finally {
      setInterpreting(false);
    }
  }

  async function generate(withInterpretation: boolean) {
    if ((!spec.trim() && files.length === 0) || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const { run_id } = await startRun(
        spec.trim(),
        buildConfig(),
        files,
        withInterpretation ? interpretation : undefined,
      );
      router.push(`/runs/${run_id}`);
    } catch (e) {
      reportError(e);
      setSubmitting(false);
    }
  }

  // Primary action: drawings go through the review gate; text-only generates directly.
  function primaryAction() {
    if (files.length > 0) return readDrawing();
    return generate(false);
  }

  const canStart = (spec.trim().length > 0 || files.length > 0) && !submitting && !interpreting;

  if (phase === "review") {
    return (
      <ReviewPanel
        files={files}
        interpretation={interpretation}
        onChange={setInterpretation}
        onBack={() => setPhase("compose")}
        onGenerate={() => generate(true)}
        submitting={submitting}
        error={error}
      />
    );
  }

  return (
    <div className="panel panel-ticks p-5 sm:p-6">
      <label className="tech-label">specification</label>
      <textarea
        value={spec}
        onChange={(e) => setSpec(e.target.value)}
        onKeyDown={(e) => {
          if ((e.metaKey || e.ctrlKey) && e.key === "Enter") primaryAction();
        }}
        rows={4}
        placeholder="describe the part in plain language — or drop an engineering drawing below"
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

      {/* drawing upload */}
      <label className="tech-label mt-5 block">technical drawing (optional)</label>
      <div
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          addFiles(e.dataTransfer.files);
        }}
        className={`mt-2 cursor-pointer rounded-[var(--radius-tech)] border border-dashed px-4 py-5 text-center text-[0.8rem] transition-colors ${
          dragging ? "border-accent/70 text-ink" : "border-line text-ink-dim hover:border-accent/40"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files) addFiles(e.target.files);
            e.target.value = "";
          }}
        />
        drop a JPEG/PNG engineering drawing here, or click to choose (up to {MAX_FILES})
      </div>

      {files.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2.5">
          {files.map((f, i) => (
            <Thumb key={`${f.name}-${i}`} file={f} onRemove={() => setFiles((p) => p.filter((_, j) => j !== i))} />
          ))}
        </div>
      )}

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
          onClick={primaryAction}
          disabled={!canStart}
          className="self-end rounded-[var(--radius-tech)] bg-accent px-6 py-2.5 font-display text-base font-semibold uppercase tracking-wider text-[#1a0e05] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {interpreting
            ? "reading…"
            : submitting
              ? "starting…"
              : files.length > 0
                ? "Read drawing ▸"
                : "Generate ▸"}
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
            placeholder={`(default: ${criticDefault})`}
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
          <SelectField
            label="cad library"
            value={cfg.library}
            options={LIBRARY_OPTIONS}
            onChange={(v) => setCfg({ ...cfg, library: v })}
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

function ReviewPanel({
  files,
  interpretation,
  onChange,
  onBack,
  onGenerate,
  submitting,
  error,
}: {
  files: File[];
  interpretation: string;
  onChange: (v: string) => void;
  onBack: () => void;
  onGenerate: () => void;
  submitting: boolean;
  error: string | null;
}) {
  return (
    <div className="panel panel-ticks p-5 sm:p-6">
      <label className="tech-label">review extracted dimensions</label>
      <p className="mt-1 text-[0.82rem] text-ink-dim">
        correct anything the drawing reader got wrong — the drawing image is still sent as the
        source of truth, so this only guides the build.
      </p>

      {files.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2.5">
          {files.map((f, i) => (
            <Thumb key={`${f.name}-${i}`} file={f} />
          ))}
        </div>
      )}

      <textarea
        value={interpretation}
        onChange={(e) => onChange(e.target.value)}
        rows={16}
        className="mt-3 w-full resize-y rounded-[var(--radius-tech)] border border-line bg-[#0c1016] p-3.5 font-mono text-[0.82rem] leading-relaxed text-ink outline-none transition-colors focus:border-accent/60"
      />

      <div className="mt-4 flex items-center justify-between gap-3">
        <button
          onClick={onBack}
          disabled={submitting}
          className="tech-label transition-colors hover:text-ink disabled:opacity-40"
        >
          ◂ edit inputs
        </button>
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={onGenerate}
          disabled={submitting}
          className="rounded-[var(--radius-tech)] bg-accent px-6 py-2.5 font-display text-base font-semibold uppercase tracking-wider text-[#1a0e05] transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {submitting ? "starting…" : "Generate ▸"}
        </motion.button>
      </div>

      {error && (
        <p className="mt-4 rounded-[var(--radius-tech)] border border-bad/40 bg-bad/10 px-3 py-2 text-[0.8rem] text-bad">
          {error}
        </p>
      )}
    </div>
  );
}

function Thumb({ file, onRemove }: { file: File; onRemove?: () => void }) {
  // Create the object URL inside the effect (not useMemo) so each StrictMode
  // double-invoke makes its own URL and revokes exactly that one — a memoized
  // URL would be revoked by the first cleanup and leave the <img> broken.
  const [url, setUrl] = useState("");
  useEffect(() => {
    const u = URL.createObjectURL(file);
    // eslint-disable-next-line react-hooks/set-state-in-effect -- object-URL pattern; sync set is intentional
    setUrl(u);
    return () => URL.revokeObjectURL(u);
  }, [file]);

  return (
    <div className="relative">
      {url && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={url}
          alt={file.name}
          className="h-24 w-auto rounded-[var(--radius-tech)] border border-line bg-[#0c1016] object-contain"
        />
      )}
      {onRemove && (
        <button
          onClick={onRemove}
          aria-label={`remove ${file.name}`}
          className="absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full border border-line bg-[#0c1016] text-[0.7rem] text-ink-dim transition-colors hover:border-bad/60 hover:text-bad"
        >
          ×
        </button>
      )}
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

function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div>
      <label className="tech-label">{label}</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="mt-2 w-full rounded-[var(--radius-tech)] border border-line bg-[#0c1016] px-3 py-2 text-[0.85rem] text-ink outline-none focus:border-accent/60"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
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
