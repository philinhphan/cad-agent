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
from cad_gen.reproject import (
    ReprojectorFn,
    build_view_locator_agent,
    locate_drawing_views,
    reproject_report,
)
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
    view_locator_model: str | Model | None = None,
    executor: ExecutorFn = run_cad_code,
    renderer: RendererFn = render_views,
    reprojector: ReprojectorFn = reproject_report,
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

    # Locate the drawing's orthographic views ONCE (advisory; drives the per-iteration
    # reprojection check). A VLM proposes the boxes; the deterministic overlap judges.
    view_regions: dict[str, list[float]] | None = None
    if drawings and config.reproject:
        view_regions = await _locate_views(
            view_locator_model or config.view_model, drawings[0], run_dir
        )

    generator = build_generator_agent(
        generator_model or config.model, reasoning_effort=config.reasoning_effort
    )
    critic = build_critic_agent(critic_model or config.critic_model)

    iterations: list[IterationRecord] = []
    feedback: str | None = None
    composite_bytes: bytes | None = None  # champion's reprojection overlay for the next prompt

    for index in range(1, config.max_iterations + 1):
        iter_dir = run_dir / f"iter_{index:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        workspace = IterationWorkspace(
            iter_dir=iter_dir,
            timeout_s=config.exec_timeout_s,
            max_attempts=config.max_exec_attempts_per_iteration,
            executor=executor,
        )

        prompt = _build_prompt(spec, feedback, interpretation, drawings, composite_bytes)
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
            # Deterministic reprojection vs. the drawing (drawing mode only). Runs BEFORE
            # the critic so its digest can ground the critique; isolated in a subprocess
            # and degrades to evaluated=False, so it can never block the critic.
            if config.reproject and drawings and execution.step_path is not None:
                record.reprojection = reprojector(
                    execution.step_path,
                    run_dir / "input" / drawing_names[0],  # reproject expects one ortho sheet
                    iter_dir / "reproject",
                    config,
                    regions=view_regions,
                    timeout_s=config.reproject_timeout_s,
                )
            record.critique = await run_critique(
                critic,
                spec=spec,
                execution=execution,
                render_path=record.render_path,
                drawings=drawings,
                reprojection=record.reprojection,
            )

        (iter_dir / "iteration.json").write_text(record.model_dump_json(indent=2))
        iterations.append(record)
        if on_iteration is not None:
            on_iteration(record)

        if record.effective_score >= config.score_threshold:
            break
        champion = max(iterations, key=lambda r: (r.effective_score, r.index))
        feedback = _build_feedback(champion, latest=record)
        composite_bytes = _champion_composite_bytes(champion)

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


async def _locate_views(
    model: str | Model, drawing: DrawingAttachment, run_dir: Path
) -> dict[str, list[float]] | None:
    """VLM-locate the drawing's orthographic views; persist + return normalized regions.

    Any failure (no API key, bad output) degrades to ``None`` so the reprojection check
    simply withholds — it can never corrupt the run. The deterministic overlap score is
    what ultimately judges the boxes.
    """
    try:
        agent = build_view_locator_agent(model)
        layout = await locate_drawing_views(agent, drawing=drawing)
    except Exception:
        return None
    (run_dir / "view_layout.json").write_text(layout.model_dump_json(indent=2))
    if not layout.views:
        return None
    return {v.label: [v.x, v.y, v.w, v.h] for v in layout.views}


def _build_prompt(
    spec: str,
    feedback: str | None,
    interpretation: str | None,
    drawings: list[DrawingAttachment],
    reproject_composite: bytes | None = None,
) -> str | list:
    """Generator prompt. Plain str for text-only runs (byte-identical to before);
    a [text, *images] list when drawings are present so the model re-reads the
    authoritative drawing on every iteration. When a champion reprojection overlay is
    available it leads the image list as a diagnostic locator (only ever set in drawing
    mode, so the text-only path is untouched)."""
    text = spec if feedback is None else f"{spec}\n\n{feedback}"
    if interpretation:
        text += (
            "\n\n## Extracted dimensions from the attached drawing (REFERENCE ONLY — "
            "the drawing image is authoritative; if anything here conflicts with the "
            f"image, trust the image):\n{interpretation}"
        )
    if not drawings:
        return text
    if reproject_composite is not None:
        text += (
            "\n\n(The FIRST attached image is a reprojection overlay — a diagnostic "
            "locator of where your previous best version differs from the drawing: blue = "
            "a drawing line you did not reproduce, orange = a line you added that the "
            "drawing lacks. Use it only to find WHERE to fix; the remaining image(s) are "
            "the authoritative original drawing(s) — read all dimensions from them.)"
        )
    content: list = [text]
    if reproject_composite is not None:
        content.append(BinaryContent(data=reproject_composite, media_type="image/png"))
    content.extend(
        BinaryContent(data=d.data, media_type=d.media_type) for d in drawings
    )
    return content


def _champion_composite_bytes(champion: IterationRecord) -> bytes | None:
    """Bytes of the champion's reprojection overlay composite, if one was produced."""
    rp = champion.reprojection
    if rp is None or not rp.evaluated or rp.composite_path is None:
        return None
    path = Path(rp.composite_path)
    return path.read_bytes() if path.exists() else None


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
    rp = champion.reprojection
    if rp is not None and rp.evaluated:
        feedback += (
            "\n\n## Independent geometric reprojection of the BEST version "
            "(deterministic, advisory):\n"
            f"{rp.digest}"
        )
        if rp.composite_path is not None:
            feedback += (
                "\nAn overlay image is attached. Use it ONLY to locate where lines are "
                "missing (blue) or extra (orange); then read the correct dimension off the "
                "ORIGINAL drawing — never estimate sizes from the overlay (it is "
                "dimensionless)."
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
