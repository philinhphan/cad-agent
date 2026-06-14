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
    if drawings:
        image_note = (
            "The FIRST attached image shows isometric / front / top / right views of the "
            "produced geometry. The remaining image(s) are the ORIGINAL engineering "
            "drawing(s) and are the source of truth — evaluate how faithfully the geometry "
            "reproduces the drawing's dimensions, features, hole types, angles and symmetry."
        )
    else:
        image_note = (
            "The attached image shows isometric / front / top / right views of the "
            "geometry. Evaluate how well it satisfies the specification."
        )
    reproject_block = ""
    if reprojection is not None and reprojection.evaluated:
        reproject_block = (
            "## Independent geometric reprojection check (deterministic, ADVISORY)\n"
            f"{reprojection.digest}\n"
            "Treat low coverage as evidence that drawing geometry is missing or extra, but a "
            "uniformly low result across all views can be a global orientation/scale difference "
            "rather than a feature error — weigh it, do not treat it as decisive.\n\n"
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
    result = await agent.run(content)
    return result.output
