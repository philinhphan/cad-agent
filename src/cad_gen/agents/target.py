"""Target-extractor agent: drawings → a typed, machine-checkable DrawingTarget.

Distinct from the free-form ``interpret_drawing`` digest: this produces the structured
``DrawingTarget`` (envelope, density, target mass, holes, …) the deterministic checks
compare against. Redacted values (a mass shown as 'XXX g') MUST come back null — never
invented — or a fabricated target would fail every correct part. A regex sanitizer backs
that rule up in code.
"""

import re

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model

from cad_gen.agents._run import run_deterministic
from cad_gen.agents.prompts import TARGET_EXTRACTOR_INSTRUCTIONS
from cad_gen.models import DrawingAttachment, DrawingTarget

# "XXX g", "??? g", "___ g" — a redacted mass the extractor must not turn into a number.
_REDACTED_MASS = re.compile(r"(?:x{2,}|\?{2,}|_{2,})\s*g\b", re.IGNORECASE)


def build_target_agent(model: str | Model) -> Agent[None, DrawingTarget]:
    return Agent(model, output_type=DrawingTarget, instructions=TARGET_EXTRACTOR_INSTRUCTIONS)


async def extract_target(
    agent: Agent[None, DrawingTarget],
    *,
    spec: str,
    drawings: list[DrawingAttachment],
    digest: str = "",
) -> DrawingTarget:
    """Extract a typed target from the drawing image(s), aided by the digest text."""
    text = (
        "Extract the machine-checkable target from the attached engineering "
        "drawing(s). Output ONLY values you can read; leave anything absent or "
        "redacted as null.\n"
    )
    if spec.strip():
        text += f"\nUser note (for disambiguation):\n{spec}\n"
    if digest.strip():
        text += (
            "\nA human-reviewed dimension digest of the same drawing follows; trust it "
            f"where it is more specific than the image:\n{digest}\n"
        )
    content: list = [text]
    content.extend(
        BinaryContent(data=d.data, media_type=d.media_type) for d in drawings
    )
    result = await run_deterministic(agent, content)
    return _sanitize(result.output, digest)


def _sanitize(target: DrawingTarget, digest: str) -> DrawingTarget:
    """Backstop the no-invented-mass rule: if the drawing redacts the mass, drop it."""
    if target.target_mass_g is not None and _REDACTED_MASS.search(digest or ""):
        target.target_mass_g = None
        target.notes = [*target.notes, "target mass redacted in drawing → not checked"]
    return target
