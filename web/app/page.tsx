import { SpecForm } from "@/components/SpecForm";

export default function HomePage() {
  return (
    <div className="mx-auto max-w-[900px] px-5 py-12 sm:py-16">
      <div className="mb-8">
        <div className="tech-label mb-3">natural language → parametric CAD</div>
        <h1 className="tech-head text-4xl text-ink sm:text-5xl">
          Describe a part.
          <br />
          Watch it <span className="text-accent">refine itself</span>.
        </h1>
        <p className="mt-4 max-w-2xl text-[0.95rem] leading-relaxed text-ink-dim">
          An agent writes CadQuery code, executes it in a sandbox, and a vision
          model critiques the rendered geometry against your spec — looping until
          it passes or the budget runs out. Every iteration streams here live,
          with the real 3D model.
        </p>
      </div>

      <SpecForm />

      <Steps />
    </div>
  );
}

function Steps() {
  const steps = [
    ["01", "generate", "LLM writes parametric CadQuery for your spec"],
    ["02", "execute", "runs in a subprocess → STL · STEP · measured metrics"],
    ["03", "critique", "vision model scores the render 0–10 vs the spec"],
    ["04", "refine", "feedback drives the next iteration; best result wins"],
  ];
  return (
    <ol className="mt-10 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {steps.map(([n, title, desc]) => (
        <li key={n} className="panel p-4">
          <div className="font-display text-2xl text-accent/80">{n}</div>
          <div className="tech-head mt-1 text-base text-ink">{title}</div>
          <p className="mt-1.5 text-[0.78rem] leading-relaxed text-ink-dim">{desc}</p>
        </li>
      ))}
    </ol>
  );
}
