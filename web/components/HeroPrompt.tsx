"use client";

import { useEffect, useRef, useState } from "react";
import { motion } from "motion/react";

const ACCEPT = ["image/png", "image/jpeg"];

export function HeroPrompt({
  onGenerate,
  busy,
}: {
  onGenerate: (files: File[]) => void;
  busy: boolean;
}) {
  const [files, setFiles] = useState<File[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  function addStaged(list: FileList | File[]) {
    const accepted = Array.from(list).filter((f) => ACCEPT.includes(f.type));
    if (accepted.length) setFiles((prev) => [...prev, ...accepted]);
  }

  const canGenerate = files.length > 0 && !busy;

  function handleGenerate() {
    if (!canGenerate) return;
    onGenerate(files);
    setFiles([]);
  }

  return (
    <div className="mx-auto w-full max-w-2xl">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          addStaged(e.dataTransfer.files);
        }}
        className={`relative rounded-3xl border bg-panel/80 p-2 backdrop-blur transition-colors ${
          dragging ? "border-accent" : "border-line"
        }`}
        style={{ boxShadow: "0 24px 80px -32px rgba(0,136,255,0.35)" }}
      >
        <div
          aria-hidden
          className="pointer-events-none absolute inset-0 rounded-3xl opacity-60"
          style={{
            background:
              "radial-gradient(120% 120% at 0% 0%, rgba(0,136,255,0.10), transparent 45%), radial-gradient(120% 120% at 100% 100%, rgba(110,110,255,0.10), transparent 45%)",
          }}
        />

        <div className="relative rounded-[1.25rem] bg-base/60 p-6">
          {files.length === 0 ? (
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="flex w-full flex-col items-center gap-3 rounded-2xl border border-dashed border-line px-6 py-10 text-center transition-colors hover:border-accent/60"
            >
              <span className="flex h-12 w-12 items-center justify-center rounded-full bg-accent/15 text-accent">
                <svg viewBox="0 0 24 24" className="h-6 w-6" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M12 16V4m0 0l-4 4m4-4l4 4" strokeLinecap="round" strokeLinejoin="round" />
                  <path d="M4 17v2a1 1 0 001 1h14a1 1 0 001-1v-2" strokeLinecap="round" />
                </svg>
              </span>
              <span className="font-display text-lg font-semibold text-ink">
                Upload technical drawing
              </span>
              <span className="text-[0.82rem] text-ink-faint">
                PNG · JPEG · click or drag &amp; drop
              </span>
            </button>
          ) : (
            <div className="flex flex-wrap items-center gap-3">
              {files.map((f, i) => (
                <StagedThumb
                  key={`${f.name}-${i}`}
                  file={f}
                  onRemove={() => setFiles((prev) => prev.filter((_, j) => j !== i))}
                />
              ))}
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="flex h-24 w-24 flex-col items-center justify-center gap-1 rounded-[var(--radius-tech)] border border-dashed border-line text-ink-faint transition-colors hover:border-accent/60 hover:text-ink"
              >
                <span className="text-2xl">+</span>
                <span className="text-[0.65rem] uppercase tracking-wider">add</span>
              </button>
            </div>
          )}

          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg"
            multiple
            hidden
            onChange={(e) => {
              if (e.target.files) addStaged(e.target.files);
              e.target.value = "";
            }}
          />
        </div>
      </div>

      <div className="mt-6 flex justify-center">
        <motion.button
          whileTap={{ scale: 0.97 }}
          onClick={handleGenerate}
          disabled={!canGenerate}
          className="flex items-center gap-2 rounded-full bg-accent px-8 py-3.5 font-display text-base font-semibold text-white shadow-lg transition-all hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {busy ? "Starting…" : "Generate 3D-Model"}
          <span aria-hidden>→</span>
        </motion.button>
      </div>
    </div>
  );
}

function StagedThumb({ file, onRemove }: { file: File; onRemove: () => void }) {
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
          className="h-24 w-24 rounded-[var(--radius-tech)] border border-line bg-[#0c1016] object-cover"
        />
      )}
      <button
        onClick={onRemove}
        aria-label={`remove ${file.name}`}
        className="absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full border border-line bg-base text-[0.7rem] text-ink-dim transition-colors hover:border-bad/60 hover:text-bad"
      >
        ×
      </button>
    </div>
  );
}
