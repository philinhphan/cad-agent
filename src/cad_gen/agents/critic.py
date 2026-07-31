"""Critic agent: vision-based evaluation of rendered geometry against the spec."""

from pathlib import Path

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from cad_gen.agents.prompts import critic_instructions
from cad_gen.bmw import resolve_model
from cad_gen.models import (
    DEFAULT_LIBRARY,
    CadLibrary,
    ConstraintValidation,
    Critique,
    DrawingAttachment,
    DrawingConstraints,
    EditDelta,
    ExecutionResult,
    ReasoningEffort,
    ReprojectionReport,
    ValidityReport,
)

# How the reviewed code is labelled in the critique prompt, per CAD library.
_CODE_HEADING = {"cadquery": "CadQuery", "build123d": "build123d"}


def build_critic_agent(
    model: str | Model,
    *,
    reasoning_effort: ReasoningEffort | None = None,
    editing: bool = False,
    library: CadLibrary = DEFAULT_LIBRARY,
) -> Agent[None, Critique]:
    model = resolve_model(model)  # route `bmw:...` strings to the BMW gateway
    # `thinking` is pydantic-ai's provider-agnostic reasoning knob; for a Gemini critic it
    # enables thinking before judging. When unset we pass no model_settings so the
    # provider's own default is left untouched.
    model_settings = ModelSettings(thinking=reasoning_effort) if reasoning_effort else None
    # Editing mode swaps in a rubric that compares before/after and penalizes no-ops;
    # `library` selects the CAD language the reviewed code is written in.
    instructions = critic_instructions(library, editing=editing)
    return Agent(
        model,
        output_type=Critique,
        instructions=instructions,
        model_settings=model_settings,
    )


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
    constraints: DrawingConstraints | None = None,
    constraint_validation: ConstraintValidation | None = None,
    reference_images: list[DrawingAttachment] | None = None,
    editing: bool = False,
    library: CadLibrary = DEFAULT_LIBRARY,
    edit_delta: EditDelta | None = None,
    validity: ValidityReport | None = None,
) -> Critique:
    metrics_json = (
        execution.metrics.model_dump_json(indent=2) if execution.metrics else "{}"
    )
    drawings = drawings or []
    reference_images = reference_images or []
    spec_heading = "Edit instruction" if editing else "Specification"
    spec_block = (
        f"## {spec_heading}\n{spec}\n\n"
        if spec.strip()
        else "## Specification\n(none — the part is defined by the attached drawing.)\n\n"
    )
    overlay_bytes = _overlay_bytes(reprojection)
    if editing:
        image_note = (
            "The FIRST attached image shows isometric / front / top / right views of the "
            "EDITED (after) model. "
            + (
                "The following image(s) show the ORIGINAL (before) model. Compare them: the "
                "difference should be exactly the requested edit. Grade whether the change "
                "was applied correctly AND everything else was preserved; a model "
                "indistinguishable from the before is a no-op and scores 0-1."
                if reference_images
                else "Judge, from the code and measurements, whether the requested edit was "
                "applied while all other geometry was preserved; a no-op scores 0-1."
            )
        )
    elif drawings:
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
    # Measured before/after comparison for editing samples. Stated as ground truth (like
    # the reprojection digest) because a small or internal edit is invisible in a render —
    # eyeballing the two image sets is exactly how a no-op slips through.
    edit_block = ""
    if edit_delta is not None:
        edit_block = (
            "## Measured change against the base model (deterministic)\n"
            f"{edit_delta.digest}\n\n"
        )
    # Same reasoning: a B-rep defect is invisible in a shaded render. Only stated when the
    # gate actually reached a verdict — reporting "could not be checked" as a finding would
    # invite the critic to penalise geometry that was never examined.
    validity_block = ""
    if validity is not None and validity.evaluated:
        validity_block = (
            "## Benchmark validity gate (deterministic)\n"
            f"{validity.digest}\n\n"
        )
    constraint_block = ""
    if constraints is not None:
        constraint_block += (
            "## Structured drawing constraints\n"
            f"{constraints.model_dump_json(indent=2)}\n\n"
        )
    if constraint_validation is not None:
        constraint_block += (
            "## Deterministic constraint validation\n"
            f"{constraint_validation.digest}\n\n"
        )
    prompt = (
        f"{spec_block}"
        f"## Measured geometry (ground truth)\n{metrics_json}\n\n"
        f"## {_CODE_HEADING[library]} code that produced it\n"
        f"```python\n{execution.code}\n```\n\n"
        f"{validity_block}"
        f"{edit_block}"
        f"{constraint_block}"
        f"{reproject_block}"
        f"{image_note}"
    )
    content: list = [
        prompt,
        BinaryContent(data=render_path.read_bytes(), media_type="image/png"),
    ]
    # Editing: the base-model (before) renders follow the after-render. Generation:
    # the original drawing(s) do. They are mutually exclusive in practice.
    content.extend(
        BinaryContent(data=d.data, media_type=d.media_type) for d in reference_images
    )
    content.extend(
        BinaryContent(data=d.data, media_type=d.media_type) for d in drawings
    )
    if overlay_bytes is not None:
        content.append(BinaryContent(data=overlay_bytes, media_type="image/png"))
    result = await agent.run(content)
    return result.output
