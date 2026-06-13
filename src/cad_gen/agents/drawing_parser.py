"""Drawing-interpreter agent: reads engineering drawings into a dimension digest.

Mirrors the critic's multimodal pattern: the prompt is a list of a text instruction
followed by one ``BinaryContent`` per drawing image. The output is free-form Markdown
(drawings are too heterogeneous for a rigid v1 schema); a downstream generator consumes
it as a non-authoritative hint while also seeing the original image.
"""

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model

from cad_gen.agents.prompts import DRAWING_PARSER_INSTRUCTIONS
from cad_gen.models import DrawingAttachment


def build_drawing_parser_agent(model: str | Model) -> Agent[None, str]:
    return Agent(model, output_type=str, instructions=DRAWING_PARSER_INSTRUCTIONS)


async def interpret_drawing(
    agent: Agent[None, str],
    *,
    spec: str,
    drawings: list[DrawingAttachment],
) -> str:
    """Run the parser agent over `drawings`, returning a Markdown dimension digest."""
    text = (
        "Read the attached engineering drawing(s) and produce a precise, structured "
        "text description suitable for driving parametric CadQuery code.\n"
    )
    if spec.strip():
        text += f"\nThe user also provided this note (use it to disambiguate):\n{spec}\n"
    content: list = [text]
    content.extend(
        BinaryContent(data=d.data, media_type=d.media_type) for d in drawings
    )
    result = await agent.run(content)
    return result.output
