"""The self-refine loop: generate -> execute -> render -> critique -> refine."""

import difflib
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
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
from cad_gen.drawing_constraints import derive_drawing_constraints, validate_drawing_constraints
from cad_gen.imaging import drawing_filename
from cad_gen.models import DrawingAttachment, DrawingConstraints, IterationRecord, RunConfig, RunResult
from cad_gen.rendering.renderer import render_views
from cad_gen.reproject import (
    ReprojectorFn,
    build_view_locator_agent,
    combine_reprojection_reports,
    locate_drawing_views,
    reproject_report,
)
from cad_gen.reproject.drawing_primitives import (
    extract_drawing_primitives,
    primitive_prompt_summary,
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
    base_step: bytes | None = None,
    base_step_name: str = "input.step",
    reference_images: list[DrawingAttachment] | None = None,
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

    Editing mode: pass `base_step` (the bytes of a base CAD model). It is seeded as
    `base_step_name` into every sandbox execution/probe dir so generated code can load it
    with ``cq.importers.importStep("input.step")``, and the generator + critic switch to
    editing instructions. `reference_images` (renders of the base model) are attached to the
    generator and critic as before-state context. Editing has no drawing, so the whole
    drawing pipeline (interpretation/constraints/reprojection) stays dormant.
    """
    config = config or RunConfig()
    drawings = drawings or []
    reference_images = reference_images or []
    is_editing = base_step is not None
    seed_files = {base_step_name: base_step} if base_step is not None else {}
    run_dir = _new_run_dir(Path(config.out_dir))
    (run_dir / "spec.txt").write_text(spec)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

    drawing_names = _persist_drawings(run_dir, drawings)
    if base_step is not None:
        input_dir = run_dir / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / base_step_name).write_bytes(base_step)
    _persist_reference_images(run_dir, reference_images)
    if drawings and interpretation is None:
        interpreter = build_drawing_parser_agent(interpreter_model or config.model)
        interpretation = await interpret_drawing(interpreter, spec=spec, drawings=drawings)
    if interpretation is not None:
        (run_dir / "drawing_interpretation.md").write_text(interpretation)
    constraints = _derive_and_persist_constraints(run_dir, interpretation)
    primitive_digests = _extract_and_persist_primitives(run_dir, drawing_names)

    # Locate each drawing's orthographic views ONCE (advisory; drives the per-iteration
    # reprojection check). A VLM proposes the boxes; the deterministic overlap judges.
    view_regions_by_drawing: list[dict[str, list[float]] | None] = []
    if drawings and config.reproject:
        for i, drawing in enumerate(drawings, start=1):
            view_regions_by_drawing.append(
                await _locate_views(
                    view_locator_model or config.view_model, drawing, run_dir, index=i
                )
            )

    generator = build_generator_agent(
        generator_model or config.model,
        reasoning_effort=config.reasoning_effort,
        editing=is_editing,
    )
    critic = build_critic_agent(
        critic_model or config.critic_model,
        reasoning_effort=config.critic_reasoning_effort,
        editing=is_editing,
    )

    iterations: list[IterationRecord] = []
    feedback: str | None = None
    composite_bytes: bytes | None = None  # champion's reprojection overlay for the next prompt
    ledger: list[TrackedIssue] = []  # issues tracked across iterations to escalate persisters

    for index in range(1, config.max_iterations + 1):
        iter_dir = run_dir / f"iter_{index:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        workspace = IterationWorkspace(
            iter_dir=iter_dir,
            timeout_s=config.exec_timeout_s,
            max_attempts=config.max_exec_attempts_per_iteration,
            executor=executor,
            seed_files=seed_files,
        )

        prompt = _build_prompt(
            spec,
            feedback,
            interpretation,
            drawings,
            composite_bytes,
            constraints=constraints,
            primitive_digests=primitive_digests,
            reference_images=reference_images,
        )
        gen_result = await generator.run(prompt, deps=workspace)

        execution = workspace.last_success or (
            workspace.attempts[-1] if workspace.attempts else None
        )
        record = IterationRecord(
            index=index, execution=execution, summary=gen_result.output
        )

        if execution is not None and execution.success:
            record.constraint_validation = validate_drawing_constraints(
                execution.metrics, constraints
            )
            record.render_path = renderer(
                execution.stl_path, iter_dir / "views.png", execution.metrics
            )
            # Deterministic reprojection vs. the drawing (drawing mode only). Runs BEFORE
            # the critic so its digest can ground the critique; isolated in a subprocess
            # and degrades to evaluated=False, so it can never block the critic.
            if config.reproject and drawings and execution.step_path is not None:
                reports = []
                for i, name in enumerate(drawing_names, start=1):
                    out_dir = (
                        iter_dir / "reproject"
                        if len(drawing_names) == 1
                        else iter_dir / "reproject" / f"drawing_{i:02d}"
                    )
                    report = reprojector(
                        execution.step_path,
                        run_dir / "input" / name,
                        out_dir,
                        config,
                        regions=(
                            view_regions_by_drawing[i - 1]
                            if i - 1 < len(view_regions_by_drawing)
                            else None
                        ),
                        timeout_s=config.reproject_timeout_s,
                    )
                    report.source_drawing = name
                    reports.append(report)
                record.reprojection = combine_reprojection_reports(
                    reports, iter_dir / "reproject"
                )
            record.critique = await run_critique(
                critic,
                spec=spec,
                execution=execution,
                render_path=record.render_path,
                drawings=drawings,
                reprojection=record.reprojection,
                constraints=constraints,
                constraint_validation=record.constraint_validation,
                reference_images=reference_images,
                editing=is_editing,
            )

        (iter_dir / "iteration.json").write_text(record.model_dump_json(indent=2))
        iterations.append(record)
        if on_iteration is not None:
            on_iteration(record)

        if record.effective_score >= config.score_threshold:
            break
        champion = max(iterations, key=lambda r: (r.effective_score, r.index))
        ledger = _update_ledger(ledger, champion)
        feedback = _build_feedback(champion, latest=record, ledger=ledger)
        composite_bytes = _champion_composite_bytes(champion)

    best = max(iterations, key=lambda r: (r.effective_score, r.index))
    result = RunResult(
        accepted=best.effective_score >= config.score_threshold,
        spec=spec,
        drawings=drawing_names,
        interpretation=interpretation,
        constraints=constraints,
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


def _persist_reference_images(run_dir: Path, images: list[DrawingAttachment]) -> list[str]:
    """Save editing base-model renders under run_dir/input/reference_NN.* for the trace."""
    if not images:
        return []
    input_dir = run_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i, img in enumerate(images, start=1):
        suffix = ".jpg" if img.media_type == "image/jpeg" else ".png"
        name = f"reference_{i:02d}{suffix}"
        (input_dir / name).write_bytes(img.data)
        names.append(name)
    return names


def _derive_and_persist_constraints(
    run_dir: Path, interpretation: str | None
) -> DrawingConstraints | None:
    if not interpretation:
        return None
    constraints = derive_drawing_constraints(interpretation)
    (run_dir / "drawing_constraints.json").write_text(
        constraints.model_dump_json(indent=2)
    )
    return constraints


def _extract_and_persist_primitives(run_dir: Path, drawing_names: list[str]) -> list[str]:
    digests: list[str] = []
    for i, name in enumerate(drawing_names, start=1):
        path = run_dir / "input" / name
        try:
            primitives = extract_drawing_primitives(path)
        except Exception as exc:
            digests.append(f"{name}: primitive extraction skipped ({type(exc).__name__}: {exc})")
            continue
        (run_dir / f"drawing_primitives_{i:02d}.json").write_text(
            primitives.model_dump_json(indent=2)
        )
        digests.append(primitive_prompt_summary(name, primitives))
    return digests


async def _locate_views(
    model: str | Model, drawing: DrawingAttachment, run_dir: Path, *, index: int = 1
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
    payload = layout.model_dump_json(indent=2)
    (run_dir / f"view_layout_{index:02d}.json").write_text(payload)
    if index == 1:
        (run_dir / "view_layout.json").write_text(payload)
    if not layout.views:
        return None
    return {v.label: [v.x, v.y, v.w, v.h] for v in layout.views}


def _build_prompt(
    spec: str,
    feedback: str | None,
    interpretation: str | None,
    drawings: list[DrawingAttachment],
    reproject_composite: bytes | None = None,
    *,
    constraints: DrawingConstraints | None = None,
    primitive_digests: list[str] | None = None,
    reference_images: list[DrawingAttachment] | None = None,
) -> str | list:
    """Generator prompt. Plain str for text-only runs (byte-identical to before);
    a [text, *images] list when drawings are present so the model re-reads the
    authoritative drawing on every iteration. When a champion reprojection overlay is
    available it leads the image list as a diagnostic locator (only ever set in drawing
    mode, so the text-only path is untouched). In editing mode `reference_images` (base-model
    renders) are attached instead — never together with drawings."""
    text = spec if feedback is None else f"{spec}\n\n{feedback}"
    if interpretation:
        text += (
            "\n\n## Extracted dimensions from the attached drawing (REFERENCE ONLY — "
            "the drawing image is authoritative; if anything here conflicts with the "
            f"image, trust the image):\n{interpretation}"
        )
    if constraints is not None:
        text += (
            "\n\n## Structured drawing constraints (machine-readable, derived from the "
            "dimension digest; use as a checklist, but resolve conflicts against the image):\n"
            f"{constraints.model_dump_json(indent=2)}"
        )
    if primitive_digests:
        text += "\n\n## Deterministic drawing primitive extraction:\n"
        text += "\n".join(f"- {d}" for d in primitive_digests)
    if reference_images:
        text += (
            "\n\n(The attached image(s) show the CURRENT state of the model you are "
            "editing — isometric and orthographic renders of the base `input.step`. Use "
            "them to locate the feature the instruction names; they are the starting "
            "point you are modifying, not a target to reproduce.)"
        )
        return [
            text,
            *(BinaryContent(data=d.data, media_type=d.media_type) for d in reference_images),
        ]
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


@dataclass
class TrackedIssue:
    """A critique issue tracked across iterations so persisters can be escalated.

    `streak` is how many consecutive champions have carried this issue; `first_seen`
    is the iteration index where it first appeared.
    """

    text: str
    first_seen: int
    streak: int


_MATCH_THRESHOLD = 0.7  # SequenceMatcher ratio above which two issues are "the same"
_DIFF_TAIL_CHARS = 1800  # cap the regression diff embedded in feedback


def _match_key(text: str) -> str:
    """Normalize an issue for cross-iteration matching: drop numbers (so a dimension
    issue still matches as its wrong value changes) and punctuation, lowercase."""
    t = re.sub(r"\d+(\.\d+)?", " ", text.lower())
    t = re.sub(r"[^a-z ]", " ", t)
    return " ".join(t.split())


def _update_ledger(
    prev: list[TrackedIssue], champion: IterationRecord
) -> list[TrackedIssue]:
    """Rebuild the issue ledger from the current champion's critique.

    Each champion issue is matched against the previous ledger by similarity; a match
    inherits `first_seen` and increments `streak` (it survived another round), a
    miss starts a fresh streak. Issues absent from the champion drop out (resolved).
    """
    if champion.critique is None:
        return []
    prev_keys = [(_match_key(p.text), p) for p in prev]
    updated: list[TrackedIssue] = []
    for issue in champion.critique.issues:
        key = _match_key(issue)
        best, best_ratio = None, 0.0
        for pkey, p in prev_keys:
            ratio = difflib.SequenceMatcher(None, key, pkey).ratio()
            if ratio > best_ratio:
                best, best_ratio = p, ratio
        if best is not None and best_ratio >= _MATCH_THRESHOLD:
            updated.append(TrackedIssue(issue, best.first_seen, best.streak + 1))
        else:
            updated.append(TrackedIssue(issue, champion.index, 1))
    return updated


def _build_feedback(
    champion: IterationRecord,
    latest: IterationRecord,
    ledger: list[TrackedIssue] | None = None,
) -> str:
    """Refinement context for the next iteration, anchored on the best result so
    far so the loop cannot drift away from a good design.

    `champion` is the highest-scoring iteration to date; `latest` is the one that just
    ran (used to show the model the diff when its most recent change regressed); `ledger`
    carries each issue's persistence so repeat offenders can be flagged as top priority.
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
    checklist = _format_checklist(ledger if ledger is not None else [], critique.issues)
    suggestions = "\n".join(f"- {s}" for s in critique.suggestions) or "- (none)"
    feedback = (
        "---\n"
        f"BEST VERSION SO FAR — a reviewer scored it {critique.score}/10. Start "
        "from THIS code; keep everything the reviewer found correct and change ONLY "
        "what the checklist requires.\n"
        f"Code:\n```python\n{champion.execution.code}\n```\n"
        f"Fix-it checklist — resolve EVERY item, then verify each before you finalize:\n"
        f"{checklist}\n"
        f"Reviewer suggestions:\n{suggestions}\n"
        "For any fillet/chamfer/shell or edge/face selection a fix needs, confirm the "
        "selector with check_selector BEFORE applying it. Then validate the full script "
        "with execute_cad_code."
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
        feedback += _format_regression(champion, latest)
    return feedback


def _format_checklist(ledger: list[TrackedIssue], issues: list[str]) -> str:
    """Numbered checklist of the champion's issues, persisters first and flagged loudly."""
    if not issues:
        return "- (none listed)"
    by_text = {t.text: t for t in ledger}
    ordered = sorted(
        issues, key=lambda i: (-(by_text[i].streak if i in by_text else 1), issues.index(i))
    )
    lines = []
    for n, issue in enumerate(ordered, start=1):
        streak = by_text[issue].streak if issue in by_text else 1
        if streak >= 2:
            lines.append(
                f"{n}. [STILL UNFIXED after {streak} attempts — TOP PRIORITY] {issue}"
            )
        else:
            lines.append(f"{n}. {issue}")
    return "\n".join(lines)


def _format_regression(champion: IterationRecord, latest: IterationRecord) -> str:
    """Show the exact best->last-change diff so the model sees what caused the regression."""
    lead = (
        f"\n\nYour most recent change scored only {latest.effective_score}/10 — WORSE "
        "than the best version above."
    )
    if (
        latest.execution is not None
        and latest.execution.success
        and latest.execution.code != champion.execution.code
    ):
        diff = "\n".join(
            difflib.unified_diff(
                champion.execution.code.splitlines(),
                latest.execution.code.splitlines(),
                fromfile="best.py",
                tofile="your_last_change.py",
                lineterm="",
            )
        )
        if len(diff) > _DIFF_TAIL_CHARS:
            diff = diff[:_DIFF_TAIL_CHARS] + "\n... (diff truncated)"
        return (
            f"{lead} Here is exactly what you changed (best -> your last change); it caused "
            f"the regression, so do NOT repeat it:\n```diff\n{diff}\n```"
        )
    return f"{lead} Do not repeat that change; improve the best version above instead."


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
