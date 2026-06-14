"""Adversarial refuter: a skeptic that must find a discrepancy or concede.

Runs the same multimodal review payload as the critic but with a default-skeptical
prompt. A found discrepancy lowers the iteration's score path and feeds the generator,
forcing another refinement pass instead of accepting a plausible-but-wrong part.
"""

from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.models import Model

from cad_gen.agents._run import run_deterministic
from cad_gen.agents.critic import build_review_content
from cad_gen.agents.prompts import REFUTER_INSTRUCTIONS
from cad_gen.models import (
    CheckReport,
    DrawingAttachment,
    DrawingTarget,
    ExecutionResult,
    Refutation,
)


def build_refuter_agent(model: str | Model) -> Agent[None, Refutation]:
    return Agent(model, output_type=Refutation, instructions=REFUTER_INSTRUCTIONS)


async def run_refutation(
    agent: Agent[None, Refutation],
    *,
    spec: str,
    execution: ExecutionResult,
    render_path: Path,
    drawings: list[DrawingAttachment] | None = None,
    target: DrawingTarget | None = None,
    check_report: CheckReport | None = None,
    section_path: Path | None = None,
) -> Refutation:
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
