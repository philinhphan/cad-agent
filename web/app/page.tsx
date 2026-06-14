import { ArticleWorkspace } from "@/components/ArticleWorkspace";

export default function HomePage() {
  return (
    <div className="w-full">
      {/* Hero */}
      <section className="px-6 pt-16 lg:px-10">
        <div className="mx-auto max-w-4xl text-center">
          <span className="inline-flex items-center gap-2 rounded-full border border-line bg-panel/50 px-3.5 py-1.5 text-[0.72rem] uppercase tracking-[0.16em] text-ink-dim">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" />
            AI-powered 3D modeling
          </span>
          <h1 className="tech-head mx-auto mt-6 max-w-4xl text-5xl text-ink sm:text-6xl lg:text-7xl">
            Automating design for the{" "}
            <span className="bg-linear-to-r from-accent to-viewport bg-clip-text text-transparent">
              physical world
            </span>
          </h1>
          <p className="mx-auto mt-5 max-w-xl text-[1rem] leading-relaxed text-ink-dim">
            Upload an engineering drawing and hit generate. We read it, extract the dimensions through
            a live pipeline, and build real, self-refining CAD geometry.
          </p>
        </div>

        <div className="mt-12">
          <ArticleWorkspace />
        </div>
      </section>

      <Features />
      <Faq />
    </div>
  );
}

function Features() {
  const items = [
    [
      "Upload a drawing",
      "Drop a technical drawing and hit Generate 3D-Model — no CAD experience required.",
    ],
    [
      "Multiple at once",
      "Upload several drawings as articles. A vision model reads each one in parallel.",
    ],
    [
      "Live pipeline",
      "Every read streams through a visual pipeline: upload → analyze → extract dimensions → ready.",
    ],
    [
      "Self-refine",
      "An agent generates, executes, and a critic scores the geometry — looping until it passes.",
    ],
  ];
  return (
    <section className="mt-28 border-t border-line px-6 py-20 lg:px-10">
      <div className="mx-auto max-w-2xl text-center">
        <h2 className="tech-head text-3xl text-ink sm:text-4xl">From idea to geometry</h2>
        <p className="mt-3 text-[0.95rem] text-ink-dim">
          Built on a self-refine loop with visual critique.
        </p>
      </div>
      <div className="mx-auto mt-12 grid max-w-6xl gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {items.map(([title, desc], i) => (
          <div
            key={title}
            className="rounded-2xl border border-line bg-panel/60 p-6 transition-colors hover:border-line-bright"
          >
            <div className="font-display text-3xl font-semibold text-accent/80">
              {String(i + 1).padStart(2, "0")}
            </div>
            <div className="tech-head mt-3 text-lg text-ink">{title}</div>
            <p className="mt-2 text-[0.85rem] leading-relaxed text-ink-dim">{desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function Faq() {
  const faqs = [
    [
      "Do I need CAD experience to use it?",
      "No. Just upload a technical drawing and hit Generate 3D-Model — the system handles the modeling.",
    ],
    [
      "What can I upload?",
      "Engineering drawings as PNG or JPEG images. Each file becomes its own article and is read in parallel.",
    ],
    [
      "Can I refine or customize the model?",
      "Yes. Review and edit the extracted dimensions before generating, and the self-refine loop iterates until the critic accepts the result.",
    ],
  ];
  return (
    <section className="border-t border-line px-6 py-20 lg:px-10">
      <div className="mx-auto max-w-3xl">
        <h2 className="tech-head text-center text-3xl text-ink sm:text-4xl">Questions</h2>
        <div className="mt-10 flex flex-col gap-3">
          {faqs.map(([q, a]) => (
            <details
              key={q}
              className="group rounded-2xl border border-line bg-panel/60 p-5 transition-colors hover:border-line-bright"
            >
              <summary className="flex cursor-pointer list-none items-center justify-between gap-4 text-[0.98rem] font-medium text-ink">
                {q}
                <span className="text-ink-faint transition-transform group-open:rotate-45">+</span>
              </summary>
              <p className="mt-3 text-[0.88rem] leading-relaxed text-ink-dim">{a}</p>
            </details>
          ))}
        </div>
      </div>
    </section>
  );
}
