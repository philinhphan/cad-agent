"""Critic agent: vision-based evaluation of rendered geometry against the spec."""

from pathlib import Path

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model

from cad_gen.agents.prompts import CRITIC_INSTRUCTIONS
from cad_gen.models import (
    Critique,
    DrawingAttachment,
    ExecutionResult,
    ReprojectionReport,
)


def build_critic_agent(model: str | Model) -> Agent[None, Critique]:
    return Agent(model, output_type=Critique, instructions=CRITIC_INSTRUCTIONS)


def _overlay_bytes(reprojection: ReprojectionReport | None) -> bytes | None:
    """Bytes of the reprojection overlay composite, if one was produced."""
    if (
        reprojection is None
        or not reprojection.evaluated
        or reprojection.composite_path is None
    ):
        return None
    path = Path(reprojection.composite_path)
    return path.read_bytes() if path.exists() else None


async def run_critique(
    agent: Agent[None, Critique],
    *,
    spec: str,
    execution: ExecutionResult,
    render_path: Path,
    drawings: list[DrawingAttachment] | None = None,
    reprojection: ReprojectionReport | None = None,
) -> Critique:
    metrics_json = (
        execution.metrics.model_dump_json(indent=2) if execution.metrics else "{}"
    )
    drawings = drawings or []
    spec_block = (
        f"## Specification\n{spec}\n\n"
        if spec.strip()
        else "## Specification\n(none — the part is defined by the attached drawing.)\n\n"
    )
    overlay_bytes = _overlay_bytes(reprojection)
    if drawings:
        image_note = (
            "The FIRST attached image shows isometric / front / top / right views of the "
            "produced geometry. The next image(s) are the ORIGINAL engineering drawing(s) "
            "and are the source of truth — evaluate how faithfully the geometry reproduces "
            "the drawing's dimensions, features, hole types, angles and symmetry."
        )
    else:
        image_note = (
            "The FIRST attached image shows isometric / front / top / right views of the "
            "geometry. Evaluate how well it satisfies the specification."
        )
    if overlay_bytes is not None:
        image_note += (
            " The LAST attached image is the deterministic reprojection overlay: blue = a "
            "drawing line the model failed to reproduce, orange = a model line absent from "
            "the drawing, red = match — use it to locate missing or misplaced geometry."
        )
    reproject_block = ""
    if reprojection is not None and reprojection.evaluated:
        verdict = "PASSED" if reprojection.passed else "DID NOT PASS"
        reproject_block = (
            "## Independent geometric reprojection check (deterministic)\n"
            f"This check {verdict}.\n"
            f"{reprojection.digest}\n\n"
        )
    prompt = (
        f"{spec_block}"
        f"## Measured geometry (ground truth)\n{metrics_json}\n\n"
        f"## CadQuery code that produced it\n```python\n{execution.code}\n```\n\n"
        f"{reproject_block}"
        f"{image_note}"
    )
    content: list = [
        prompt,
        BinaryContent(data=render_path.read_bytes(), media_type="image/png"),
    ]
    content.extend(
        BinaryContent(data=d.data, media_type=d.media_type) for d in drawings
    )
    if overlay_bytes is not None:
        content.append(BinaryContent(data=overlay_bytes, media_type="image/png"))
    result = await agent.run(content)
    return result.output
