"""The self-refine loop: generate -> execute -> render -> critique -> refine."""

import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from pydantic_ai import BinaryContent
from pydantic_ai.models import Model

from cad_gen.agents.critic import build_critic_agent, run_critique
from cad_gen.agents.drawing_parser import build_drawing_parser_agent, interpret_drawing
from cad_gen.agents.generator import (
    ExecutorFn,
    IterationWorkspace,
    build_generator_agent,
)
from cad_gen.imaging import drawing_filename
from cad_gen.models import DrawingAttachment, IterationRecord, RunConfig, RunResult
from cad_gen.rendering.renderer import render_views
from cad_gen.sandbox.executor import run_cad_code

RendererFn = Callable[..., Path]
IterationCallback = Callable[[IterationRecord], None]


async def generate_cad(
    spec: str,
    config: RunConfig | None = None,
    *,
    drawings: list[DrawingAttachment] | None = None,
    interpretation: str | None = None,
    generator_model: str | Model | None = None,
    critic_model: str | Model | None = None,
    interpreter_model: str | Model | None = None,
    executor: ExecutorFn = run_cad_code,
    renderer: RendererFn = render_views,
    on_iteration: IterationCallback | None = None,
) -> RunResult:
    """Run the full self-refine loop for `spec`; artifacts land under config.out_dir.

    `drawings` are input engineering-drawing images; when present they are persisted,
    threaded to the generator (every iteration) and critic as authoritative ground truth,
    and `interpretation` (an extracted-dimensions digest) is auto-generated if not supplied
    by the caller (e.g. a human-reviewed/edited digest from the CLI or web gate).
    """
    config = config or RunConfig()
    drawings = drawings or []
    run_dir = _new_run_dir(Path(config.out_dir))
    (run_dir / "spec.txt").write_text(spec)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

    drawing_names = _persist_drawings(run_dir, drawings)
    if drawings and interpretation is None:
        interpreter = build_drawing_parser_agent(interpreter_model or config.model)
        interpretation = await interpret_drawing(interpreter, spec=spec, drawings=drawings)
    if interpretation is not None:
        (run_dir / "drawing_interpretation.md").write_text(interpretation)

    generator = build_generator_agent(generator_model or config.model)
    critic = build_critic_agent(critic_model or config.critic_model)

    iterations: list[IterationRecord] = []
    feedback: str | None = None

    for index in range(1, config.max_iterations + 1):
        iter_dir = run_dir / f"iter_{index:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        workspace = IterationWorkspace(
            iter_dir=iter_dir,
            timeout_s=config.exec_timeout_s,
            max_attempts=config.max_exec_attempts_per_iteration,
            executor=executor,
        )

        prompt = _build_prompt(spec, feedback, interpretation, drawings)
        gen_result = await generator.run(prompt, deps=workspace)

        execution = workspace.last_success or (
            workspace.attempts[-1] if workspace.attempts else None
        )
        record = IterationRecord(
            index=index, execution=execution, summary=gen_result.output
        )

        if execution is not None and execution.success:
            record.render_path = renderer(
                execution.stl_path, iter_dir / "views.png", execution.metrics
            )
            record.critique = await run_critique(
                critic,
                spec=spec,
                execution=execution,
                render_path=record.render_path,
                drawings=drawings,
            )

        (iter_dir / "iteration.json").write_text(record.model_dump_json(indent=2))
        iterations.append(record)
        if on_iteration is not None:
            on_iteration(record)

        if record.effective_score >= config.score_threshold:
            break
        champion = max(iterations, key=lambda r: (r.effective_score, r.index))
        feedback = _build_feedback(champion, latest=record)

    best = max(iterations, key=lambda r: (r.effective_score, r.index))
    result = RunResult(
        accepted=best.effective_score >= config.score_threshold,
        spec=spec,
        drawings=drawing_names,
        interpretation=interpretation,
        best=best,
        iterations=iterations,
        run_dir=run_dir,
    )
    _persist_final(result)
    _write_report(result, config)
    (run_dir / "run_result.json").write_text(result.model_dump_json(indent=2))
    return result


def _new_run_dir(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / base
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = out_dir / f"{base}_{suffix}"
    run_dir.mkdir()
    return run_dir


def _persist_drawings(run_dir: Path, drawings: list[DrawingAttachment]) -> list[str]:
    """Save input drawings under run_dir/input/ with sanitized names; return the names.

    The client-supplied filename is never used on disk — we name by index + a suffix
    derived from the media type, so the path can be served safely as an artifact.
    """
    if not drawings:
        return []
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i, d in enumerate(drawings, start=1):
        name = drawing_filename(i, d.media_type)
        (input_dir / name).write_bytes(d.data)
        names.append(name)
    return names


def _build_prompt(
    spec: str,
    feedback: str | None,
    interpretation: str | None,
    drawings: list[DrawingAttachment],
) -> str | list:
    """Generator prompt. Plain str for text-only runs (byte-identical to before);
    a [text, *images] list when drawings are present so the model re-reads the
    authoritative drawing on every iteration."""
    text = spec if feedback is None else f"{spec}\n\n{feedback}"
    if interpretation:
        text += (
            "\n\n## Extracted dimensions from the attached drawing (REFERENCE ONLY — "
            "the drawing image is authoritative; if anything here conflicts with the "
            f"image, trust the image):\n{interpretation}"
        )
    if not drawings:
        return text
    content: list = [text]
    content.extend(
        BinaryContent(data=d.data, media_type=d.media_type) for d in drawings
    )
    return content


def _build_feedback(champion: IterationRecord, latest: IterationRecord) -> str:
    """Refinement context for the next iteration, anchored on the best result so
    far so the loop cannot drift away from a good design.

    `champion` is the highest-scoring iteration to date; `latest` is the one that
    just ran (used only to warn the model when its most recent change regressed).
    """
    if champion.execution is None or not champion.execution.success:
        error = (
            f"Last error:\n{champion.execution.error}"
            if champion.execution is not None
            else "No code was produced at all."
        )
        return (
            "---\n"
            f"PREVIOUS ATTEMPT FAILED TO PRODUCE WORKING CODE. {error}\n"
            "Write a corrected — simpler and more robust — script, and validate it "
            "with execute_cad_code."
        )

    critique = champion.critique
    issues = "\n".join(f"- {i}" for i in critique.issues) or "- (none listed)"
    suggestions = "\n".join(f"- {s}" for s in critique.suggestions) or "- (none)"
    feedback = (
        "---\n"
        f"BEST VERSION SO FAR — a reviewer scored it {critique.score}/10. Start "
        "from THIS code; keep everything the reviewer found correct and change only "
        "what is needed to fix the issues below.\n"
        f"Code:\n```python\n{champion.execution.code}\n```\n"
        f"Reviewer issues:\n{issues}\n"
        f"Reviewer suggestions:\n{suggestions}\n"
        "Produce an improved version that fixes every issue, then validate it with "
        "execute_cad_code."
    )
    if latest is not champion and latest.effective_score < champion.effective_score:
        feedback += (
            f"\n\nNote: your most recent change scored only {latest.effective_score}"
            "/10 — it regressed below the best version above. Do not repeat that "
            "change; improve the best version instead."
        )
    return feedback


def _persist_final(result: RunResult) -> None:
    final_dir = result.run_dir / "final"
    final_dir.mkdir(exist_ok=True)
    execution = result.best.execution
    if execution is None:
        return
    (final_dir / "model.py").write_text(execution.code)
    for src in (execution.stl_path, execution.step_path):
        if src is not None and Path(src).exists():
            shutil.copy2(src, final_dir / Path(src).name)
    if result.best.render_path is not None and Path(result.best.render_path).exists():
        shutil.copy2(result.best.render_path, final_dir / "views.png")
    if result.best.critique is not None:
        (final_dir / "critique.json").write_text(
            result.best.critique.model_dump_json(indent=2)
        )


def _write_report(result: RunResult, config: RunConfig) -> None:
    verdict = "ACCEPTED" if result.accepted else "NOT ACCEPTED (budget exhausted)"
    lines = [
        "# cad-gen report",
        "",
        f"**Spec:** {result.spec or '(from drawing)'}",
        "",
    ]
    if result.drawings:
        lines += [
            f"**Input drawings:** {', '.join(result.drawings)}",
            "",
            *[f"![input drawing](input/{name})" for name in result.drawings],
            "",
        ]
    lines += [
        f"**Verdict:** {verdict} — best score {result.best.effective_score}/10 "
        f"(threshold {config.score_threshold}, iteration {result.best.index} of "
        f"{len(result.iterations)} run)",
        "",
        "| iteration | executed | score | matches spec | issues |",
        "|---|---|---|---|---|",
    ]
    for rec in result.iterations:
        executed = "yes" if rec.execution is not None and rec.execution.success else "FAILED"
        if rec.critique is not None:
            score = f"{rec.critique.score}/10"
            matches = "yes" if rec.critique.matches_spec else "no"
            issues = "; ".join(rec.critique.issues) or "—"
        else:
            score, matches = "0/10", "no"
            issues = (rec.execution.error or "no code").splitlines()[-1] if rec.execution else "no code"
        if len(issues) > 120:
            issues = issues[:117] + "..."
        lines.append(f"| {rec.index} | {executed} | {score} | {matches} | {issues} |")

    lines += [
        "",
        "## Final artifacts",
        "",
        "- `final/model.py` — CadQuery source",
        "- `final/model.stl`, `final/model.step` — geometry",
        "- `final/views.png` — rendered views (below)",
        "",
        "![final views](final/views.png)",
        "",
    ]
    if result.best.critique is not None:
        lines += [
            "## Final critique",
            "",
            result.best.critique.summary,
            "",
        ]
    (result.run_dir / "report.md").write_text("\n".join(lines))
