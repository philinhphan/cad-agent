"""The self-refine loop: generate -> execute -> render -> critique -> refine.

Deterministic checks (mass, envelope, topology) are ADVISORY: they never override the
critic's score for acceptance. They are fed to the critic as authoritative evidence, turned
into a numeric gradient for the generator, and surfaced to the user — the false-10/10 fix
comes from a grounded, enumerating critic (temp 0) plus an optional adversarial refuter,
not from a code-level score cap.
"""

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
from cad_gen.agents.panel import run_critic_panel
from cad_gen.agents.refuter import build_refuter_agent, run_refutation
from cad_gen.agents.target import build_target_agent, extract_target
from cad_gen.eval.checks import compute_mass_g, run_checks
from cad_gen.imaging import drawing_filename
from cad_gen.models import (
    Critique,
    DrawingAttachment,
    DrawingTarget,
    IterationRecord,
    RunConfig,
    RunResult,
)
from cad_gen.rendering.renderer import render_sections, render_views
from cad_gen.sandbox.executor import run_cad_code

RendererFn = Callable[..., Path]
SectionRendererFn = Callable[..., Path | None]
IterationCallback = Callable[[IterationRecord], None]


async def generate_cad(
    spec: str,
    config: RunConfig | None = None,
    *,
    drawings: list[DrawingAttachment] | None = None,
    interpretation: str | None = None,
    target: DrawingTarget | None = None,
    generator_model: str | Model | None = None,
    critic_model: str | Model | None = None,
    interpreter_model: str | Model | None = None,
    target_model: str | Model | None = None,
    refuter_model: str | Model | None = None,
    executor: ExecutorFn = run_cad_code,
    renderer: RendererFn = render_views,
    section_renderer: SectionRendererFn = render_sections,
    on_iteration: IterationCallback | None = None,
) -> RunResult:
    """Run the full self-refine loop for `spec`; artifacts land under config.out_dir.

    `drawings` are input engineering-drawing images; when present they are persisted,
    threaded to the generator (every iteration) and critic as authoritative ground truth,
    `interpretation` (an extracted-dimensions digest) is auto-generated if not supplied,
    and a typed `target` (envelope/mass/holes/...) is extracted for the deterministic checks.
    """
    config = config or RunConfig()
    drawings = drawings or []
    run_dir = _new_run_dir(Path(config.out_dir))
    (run_dir / "spec.txt").write_text(spec)
    (run_dir / "config.json").write_text(config.model_dump_json(indent=2))

    drawing_names = _persist_drawings(run_dir, drawings)
    auto_interpreted = bool(drawings) and interpretation is None
    if auto_interpreted:
        interpreter = build_drawing_parser_agent(interpreter_model or config.model)
        interpretation = await interpret_drawing(interpreter, spec=spec, drawings=drawings)
    if interpretation is not None:
        (run_dir / "drawing_interpretation.md").write_text(interpretation)

    target = await _resolve_target(
        target=target,
        config=config,
        drawings=drawings,
        spec=spec,
        interpretation=interpretation,
        target_model=target_model or config.model,
    )
    if target is not None:
        (run_dir / "drawing_target.json").write_text(target.model_dump_json(indent=2))

    generator = build_generator_agent(generator_model or config.model)
    use_panel = bool(config.critic_models) or config.critic_samples > 1
    critic_specs: list = list(config.critic_models) if config.critic_models else [
        critic_model or config.critic_model
    ]
    critic = None if use_panel else build_critic_agent(critic_specs[0])
    refuter = (
        build_refuter_agent(refuter_model or critic_specs[0])
        if config.enable_adversarial
        else None
    )

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
            density_kg_m3=target.density_kg_m3 if target else None,
            target_mass_g=target.target_mass_g if target else None,
            mass_tol_g=target.mass_tol_g if target else None,
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
            if (
                target is not None
                and target.density_kg_m3 is not None
                and execution.metrics is not None
                and execution.metrics.mass_g is None
            ):
                execution.metrics.mass_g = compute_mass_g(
                    execution.metrics.volume_mm3, target.density_kg_m3
                )
            record.render_path = renderer(
                execution.stl_path, iter_dir / "views.png", execution.metrics
            )
            if config.enable_sections:
                record.section_path = _safe_sections(
                    section_renderer,
                    execution.stl_path,
                    iter_dir / "sections.png",
                    target=target,
                    metrics=execution.metrics,
                )
            record.check_report = run_checks(execution.metrics, target)
            record.critique = await _run_review(
                use_panel=use_panel,
                critic=critic,
                critic_specs=critic_specs,
                config=config,
                spec=spec,
                execution=execution,
                record=record,
                drawings=drawings,
                target=target,
            )
            if refuter is not None:
                record.refutation = await run_refutation(
                    refuter,
                    spec=spec,
                    execution=execution,
                    render_path=record.render_path,
                    drawings=drawings,
                    target=target,
                    check_report=record.check_report,
                    section_path=record.section_path,
                )
                _apply_refutation(record, config)

        (iter_dir / "iteration.json").write_text(record.model_dump_json(indent=2))
        iterations.append(record)
        if on_iteration is not None:
            on_iteration(record)

        if record.effective_score >= config.score_threshold:
            break
        champion = max(iterations, key=_rank)
        feedback = _build_feedback(champion, latest=record, history=iterations)

    best = max(iterations, key=_rank)
    result = RunResult(
        accepted=best.effective_score >= config.score_threshold,
        spec=spec,
        drawings=drawing_names,
        interpretation=interpretation,
        target=target,
        best=best,
        iterations=iterations,
        run_dir=run_dir,
    )
    _persist_final(result)
    _write_report(result, config)
    (run_dir / "run_result.json").write_text(result.model_dump_json(indent=2))
    return result


def _rank(record: IterationRecord) -> tuple:
    """Best-iteration key. Score is primary (advisory gating); passing the deterministic
    checks only breaks ties between equal scores — it never overrides acceptance."""
    return (record.effective_score, record.passes_checks, record.index)


async def _run_review(
    *,
    use_panel: bool,
    critic,
    critic_specs: list,
    config: RunConfig,
    spec: str,
    execution,
    record: IterationRecord,
    drawings: list[DrawingAttachment],
    target: DrawingTarget | None,
):
    kwargs = dict(
        spec=spec,
        execution=execution,
        render_path=record.render_path,
        drawings=drawings,
        target=target,
        check_report=record.check_report,
        section_path=record.section_path,
    )
    if use_panel:
        critique, panel = await run_critic_panel(
            models=critic_specs,
            samples=config.critic_samples,
            aggregation=config.critic_aggregation,
            **kwargs,
        )
        record.panel_critiques = panel
        return critique
    return await run_critique(critic, **kwargs)


def _apply_refutation(record: IterationRecord, config: RunConfig) -> None:
    """Grade a proven discrepancy instead of binary-capping it. A critical/major flaw caps
    the score below the accept threshold (blocking acceptance, as before); a minor sub-mm
    nit costs at most one point and never blocks on its own — otherwise a perfect part can
    never be accepted because a skeptic always finds *something*. An LLM-ensemble decision,
    not a deterministic override; the discrepancy is fed forward either way."""
    ref = record.refutation
    if ref is None or not ref.found_discrepancy or record.critique is None:
        return
    record.critique.issues = [*record.critique.issues, *ref.discrepancies]
    if ref.severity in ("critical", "major"):
        cap = max(0, config.score_threshold - 1)
        record.critique.score = min(record.critique.score, cap)
    else:
        # "minor", or a found-but-ungraded discrepancy: register a one-point penalty so the
        # signal is not lost, but do not forbid acceptance on a cosmetic finding alone.
        record.critique.score = max(0, record.critique.score - 1)
    record.critique.matches_spec = record.critique.score >= config.score_threshold


def _safe_sections(
    section_renderer: SectionRendererFn,
    stl_path,
    out_png: Path,
    *,
    target: DrawingTarget | None = None,
    metrics=None,
) -> Path | None:
    try:
        return section_renderer(stl_path, out_png, target=target, metrics=metrics)
    except Exception:
        return None


async def _resolve_target(
    *,
    target: DrawingTarget | None,
    config: RunConfig,
    drawings: list[DrawingAttachment],
    spec: str,
    interpretation: str | None,
    target_model: str | Model,
) -> DrawingTarget | None:
    """Build the typed target: explicit override > LLM extraction from the drawing >
    digest-only. Config overrides always win; absent everywhere → None. Extraction failures
    (e.g. no API key in tests) degrade gracefully to a digest-only target."""
    overrides = _config_target_overrides(config)
    base = target
    if base is None and drawings:
        base = DrawingTarget(raw_digest=interpretation or "")
        try:
            agent = build_target_agent(target_model)
            base = await extract_target(
                agent, spec=spec, drawings=drawings, digest=interpretation or ""
            )
        except Exception:
            base = DrawingTarget(raw_digest=interpretation or "")
    if base is None and overrides:
        base = DrawingTarget()
    if base is None:
        return None
    if interpretation:
        base.raw_digest = interpretation
    for field, value in overrides.items():
        setattr(base, field, value)
    return base


def _config_target_overrides(config: RunConfig) -> dict:
    overrides = {}
    if config.target_mass_g is not None:
        overrides["target_mass_g"] = config.target_mass_g
    if config.mass_tol_g is not None:
        overrides["mass_tol_g"] = config.mass_tol_g
    if config.envelope_mm is not None:
        overrides["envelope_mm"] = config.envelope_mm
    if config.density_kg_m3 is not None:
        overrides["density_kg_m3"] = config.density_kg_m3
    return overrides


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


def _feedback_checks_block(record: IterationRecord) -> str:
    """Deterministic check deltas + any adversarial finding — the numeric gradient that
    moves the generator toward the target even though the score is the critic's call."""
    parts: list[str] = []
    if record.check_report is not None and record.check_report.checks:
        lines = [
            f"- {c.name}: {c.status.value.upper()} — {c.message}"
            for c in record.check_report.checks
        ]
        parts.append(
            "DETERMINISTIC CHECKS (authoritative, computed by the CAD kernel):\n"
            + "\n".join(lines)
        )
    ref = record.refutation
    if ref is not None and ref.found_discrepancy and ref.most_severe:
        parts.append(f"ADVERSARIAL REVIEW flagged: {ref.most_severe}")
    if not parts:
        return ""
    return (
        "\n".join(parts)
        + "\nFix any FAILING check first — these are measured facts, not opinions.\n\n"
    )


def _checklist_block(critique: "Critique | None") -> str:
    """The champion's per-requirement checklist as a keep/fix table. The critic already
    produces this (pass/fail per requirement); surfacing it tells the generator exactly
    what NOT to touch, so a targeted fix doesn't regress an already-correct feature."""
    if critique is None or not critique.checklist:
        return ""
    label = {"pass": "PASS", "fail": "FAIL", "uncertain": "UNCERTAIN"}
    lines = []
    for i in critique.checklist:
        detail = ""
        if i.target:
            detail = f" — target {i.target}"
            if i.observed:
                detail += f", observed {i.observed}"
        lines.append(f"- [{label.get(i.status, i.status.upper())}] {i.requirement}{detail}")
    return (
        "REQUIREMENT STATUS (keep every PASS exactly as-is; fix every FAIL/UNCERTAIN):\n"
        + "\n".join(lines)
        + "\n"
    )


def _regression_ledger(
    history: "tuple[IterationRecord, ...] | list[IterationRecord]",
    champion: IterationRecord,
) -> str:
    """Requirements that reached PASS in ANY earlier iteration but the champion no longer
    shows passing — folded across the whole run so a fix for one issue cannot silently
    undo another (the whack-a-mole the loop kept hitting). Items the champion already
    passes are covered by the checklist table, so they are omitted here to avoid bloat."""
    ever_passed: dict[str, str] = {}
    for rec in history:
        if rec.critique is None:
            continue
        for item in rec.critique.checklist:
            if item.status == "pass":
                ever_passed.setdefault(item.requirement.strip().lower(), item.requirement)
    champ_pass = {
        i.requirement.strip().lower()
        for i in (champion.critique.checklist if champion.critique else [])
        if i.status == "pass"
    }
    at_risk = [disp for key, disp in ever_passed.items() if key not in champ_pass]
    if not at_risk:
        return ""
    return (
        "ALREADY-SATISFIED EARLIER (an earlier version got these right — your change MUST "
        "NOT regress them):\n" + "\n".join(f"- {r}" for r in at_risk) + "\n"
    )


def _build_feedback(
    champion: IterationRecord,
    latest: IterationRecord,
    history: "tuple[IterationRecord, ...] | list[IterationRecord]" = (),
) -> str:
    """Refinement context for the next iteration, anchored on the best result so
    far so the loop cannot drift away from a good design.

    `champion` is the highest-scoring iteration to date; `latest` is the one that
    just ran (used only to warn the model when its most recent change regressed);
    `history` is every iteration so far (folded into a do-not-regress ledger).
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
        + _feedback_checks_block(champion)
        + _regression_ledger(history, champion)
        + f"BEST VERSION SO FAR — a reviewer scored it {critique.score}/10. Start "
        "from THIS code; keep everything the reviewer found correct and change only "
        "what is needed to fix the issues below.\n"
        + _checklist_block(critique)
        + f"Code:\n```python\n{champion.execution.code}\n```\n"
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
    if result.best.section_path is not None and Path(result.best.section_path).exists():
        shutil.copy2(result.best.section_path, final_dir / "sections.png")
    if result.best.critique is not None:
        (final_dir / "critique.json").write_text(
            result.best.critique.model_dump_json(indent=2)
        )
    if result.best.check_report is not None:
        (final_dir / "check_report.json").write_text(
            result.best.check_report.model_dump_json(indent=2)
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

    if result.best.check_report is not None and result.best.check_report.checks:
        lines += ["", "## Deterministic checks (advisory)", ""]
        for c in result.best.check_report.checks:
            lines.append(f"- **{c.name}** — {c.status.value.upper()}: {c.message}")

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
