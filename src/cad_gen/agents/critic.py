"""Critic agent: vision-based evaluation of rendered geometry against the spec."""

from pathlib import Path

from pydantic_ai import Agent, BinaryContent
from pydantic_ai.models import Model

from cad_gen.agents.prompts import CRITIC_INSTRUCTIONS
from cad_gen.models import Critique, ExecutionResult


def build_critic_agent(model: str | Model) -> Agent[None, Critique]:
    return Agent(model, output_type=Critique, instructions=CRITIC_INSTRUCTIONS)


async def run_critique(
    agent: Agent[None, Critique],
    *,
    spec: str,
    execution: ExecutionResult,
    render_path: Path,
) -> Critique:
    metrics_json = (
        execution.metrics.model_dump_json(indent=2) if execution.metrics else "{}"
    )
    prompt = (
        f"## Specification\n{spec}\n\n"
        f"## Measured geometry (ground truth)\n{metrics_json}\n\n"
        f"## CadQuery code that produced it\n```python\n{execution.code}\n```\n\n"
        "The attached image shows isometric / front / top / right views of the "
        "geometry. Evaluate how well it satisfies the specification."
    )
    result = await agent.run(
        [prompt, BinaryContent(data=render_path.read_bytes(), media_type="image/png")]
    )
    return result.output
