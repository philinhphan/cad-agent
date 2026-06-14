"""Critic agent: vision-based evaluation of rendered geometry against the spec.

Grounded by the deterministic checks (the critic is told they are authoritative) and
forced to enumerate a per-requirement checklist, run at temperature 0 to cut variance.
The review-content builder is shared with the adversarial refuter.
"""

from pathlib import Path

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model

from cad_gen.agents._run import run_deterministic
from cad_gen.agents.prompts import CRITIC_INSTRUCTIONS
from cad_gen.models import (
    CheckReport,
    Critique,
    DrawingAttachment,
    DrawingTarget,
    ExecutionResult,
)


def build_critic_agent(model: str | Model) -> Agent[None, Critique]:
    return Agent(model, output_type=Critique, instructions=CRITIC_INSTRUCTIONS)


def _checks_block(check_report: CheckReport | None) -> str:
    if check_report is None or not check_report.checks:
        return ""
    lines = [
        f"- {c.name}: {c.status.value.upper()}{' [critical]' if c.critical else ''} — {c.message}"
        for c in check_report.checks
    ]
    return (
        "## Deterministic checks (AUTHORITATIVE — measured by the CAD kernel)\n"
        + "\n".join(lines)
        + "\nA FAIL here is a measured fact and forbids any score of 8 or above.\n\n"
    )


def _ungradeable_block(target: DrawingTarget | None) -> str:
    """Values the reviewer must NOT grade: a redacted mass and any item the extractor
    flagged uncertain. Without this the critic confabulates a target (e.g. a 247 g mass the
    drawing redacts) and rubber-stamps the part once geometry coincidentally matches it."""
    if target is None:
        return ""
    lines: list[str] = []
    if target.target_mass_g is None and target.density_kg_m3 is not None:
        lines.append(
            "MASS is REDACTED/UNKNOWN (the drawing asks for it). The measured mass is the "
            "ANSWER, not a target — do NOT grade observed mass against any value, and do NOT "
            "invent, infer or back-compute a target mass."
        )
    for h in target.holes:
        if h.uncertain:
            lines.append(f"HOLE {h.note or f'Ø{h.diameter_mm:g}'} is UNCERTAIN — status must be 'uncertain'.")
    for fl in target.fillets:
        if fl.uncertain:
            lines.append(f"FILLET R{fl.radius_mm:g} is UNCERTAIN — status must be 'uncertain'.")
    for a in target.angles:
        if a.uncertain:
            lines.append(f"ANGLE {a.angle_deg:g}° is UNCERTAIN — status must be 'uncertain'.")
    if any("uncertain" in n.lower() for n in target.notes):
        lines.append(
            "Some dimensions/positions are marked [UNCERTAIN] in the notes — for any feature "
            "whose position or size you cannot verify from the views, set its checklist status "
            "'uncertain' (severity 'major' for a real position you cannot pin down, so a "
            "guessed interpretation is not silently accepted as correct)."
        )
    if not lines:
        return ""
    return (
        "## DO NOT GRADE THESE (mark status 'uncertain', never pass/fail)\n- "
        + "\n- ".join(lines)
        + "\n\n"
    )


def build_review_content(
    *,
    spec: str,
    execution: ExecutionResult,
    render_path: Path,
    drawings: list[DrawingAttachment] | None = None,
    target: DrawingTarget | None = None,
    check_report: CheckReport | None = None,
    section_path: Path | None = None,
) -> list:
    """Assemble the multimodal review payload shared by the critic and the refuter:
    [text prompt, render image, (sections image), *original drawing images]."""
    metrics_json = (
        execution.metrics.model_dump_json(indent=2) if execution.metrics else "{}"
    )
    drawings = drawings or []
    spec_block = (
        f"## Specification\n{spec}\n\n"
        if spec.strip()
        else "## Specification\n(none — the part is defined by the attached drawing.)\n\n"
    )
    target_block = (
        "## Target extracted from the drawing (intended design)\n"
        f"{target.model_dump_json(indent=2, exclude={'raw_digest'})}\n\n"
        if target is not None
        else ""
    )
    checks_block = _checks_block(check_report)

    section_path = Path(section_path) if section_path is not None else None
    has_sections = section_path is not None and section_path.exists()
    sect_note = "Image 2 = mid-plane cross-sections (verify hole depth/type, wall thickness). " if has_sections else ""
    if drawings:
        image_note = (
            "Image 1 = shaded iso/front/top/right views (edge outlines + mm axes) of the "
            f"produced geometry. {sect_note}The remaining image(s) are the ORIGINAL "
            "engineering drawing(s) and are the source of truth — grade how faithfully the "
            "geometry reproduces every callout."
        )
    else:
        image_note = (
            f"Image 1 = shaded iso/front/top/right views of the geometry. {sect_note}"
            "Evaluate how well it satisfies the specification."
        )
    prompt = (
        f"{spec_block}"
        f"{target_block}"
        f"{_ungradeable_block(target)}"
        f"{checks_block}"
        f"## Measured geometry (ground truth)\n{metrics_json}\n\n"
        f"## CadQuery code that produced it\n```python\n{execution.code}\n```\n\n"
        f"{image_note}"
    )
    content: list = [
        prompt,
        BinaryContent(data=render_path.read_bytes(), media_type="image/png"),
    ]
    if has_sections:
        content.append(
            BinaryContent(data=section_path.read_bytes(), media_type="image/png")
        )
    content.extend(
        BinaryContent(data=d.data, media_type=d.media_type) for d in drawings
    )
    return content


async def run_critique(
    agent: Agent[None, Critique],
    *,
    spec: str,
    execution: ExecutionResult,
    render_path: Path,
    drawings: list[DrawingAttachment] | None = None,
    target: DrawingTarget | None = None,
    check_report: CheckReport | None = None,
    section_path: Path | None = None,
) -> Critique:
    content = build_review_content(
        spec=spec,
        execution=execution,
        render_path=render_path,
        drawings=drawings,
        target=target,
        check_report=check_report,
        section_path=section_path,
    )
    result = await run_deterministic(agent, content)
    return result.output
