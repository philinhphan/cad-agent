"""Generator agent: writes CadQuery code and validates it via the sandbox tool."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from cad_gen.agents.prompts import GENERATOR_INSTRUCTIONS
from cad_gen.models import ExecutionResult, ReasoningEffort
from cad_gen.sandbox.executor import run_cad_code

ExecutorFn = Callable[..., ExecutionResult]


@dataclass
class IterationWorkspace:
    """Per-iteration execution state shared between orchestrator and tool.

    The last successful execution recorded here is the code of record for the
    iteration — never the text the agent returns.
    """

    iter_dir: Path
    timeout_s: float = 60
    max_attempts: int = 4
    executor: ExecutorFn = run_cad_code
    attempts: list[ExecutionResult] = field(default_factory=list)

    @property
    def last_success(self) -> ExecutionResult | None:
        return next((r for r in reversed(self.attempts) if r.success), None)

    def execute(self, code: str) -> ExecutionResult:
        attempt_dir = self.iter_dir / f"attempt_{len(self.attempts) + 1:02d}"
        result = self.executor(code, attempt_dir, timeout_s=self.timeout_s)
        self.attempts.append(result)
        return result


def build_generator_agent(
    model: str | Model, *, reasoning_effort: ReasoningEffort | None = None
) -> Agent[IterationWorkspace, str]:
    # `thinking` is pydantic-ai's provider-agnostic reasoning-effort knob; when unset we
    # pass no model_settings so the provider's own default is left untouched.
    model_settings = ModelSettings(thinking=reasoning_effort) if reasoning_effort else None
    agent: Agent[IterationWorkspace, str] = Agent(
        model,
        deps_type=IterationWorkspace,
        output_type=str,
        instructions=GENERATOR_INSTRUCTIONS,
        model_settings=model_settings,
    )

    @agent.tool
    def execute_cad_code(ctx: RunContext[IterationWorkspace], code: str) -> str:
        """Run a complete CadQuery script in the sandbox.

        Returns measured geometry metrics on success, or the error traceback on
        failure.
        """
        ws = ctx.deps
        if len(ws.attempts) >= ws.max_attempts:
            return (
                "EXECUTION BUDGET EXHAUSTED: no more attempts are allowed in this "
                "iteration. Stop calling this tool and reply with a brief summary "
                "of what went wrong."
            )
        result = ws.execute(code)
        if result.success and result.metrics is not None:
            m = result.metrics
            return (
                "SUCCESS — the model built and exported.\n"
                f"volume: {m.volume_mm3:.1f} mm3\n"
                f"bbox: {m.bbox_mm[0]:.2f} x {m.bbox_mm[1]:.2f} x {m.bbox_mm[2]:.2f} mm\n"
                f"solids: {m.n_solids}, faces: {m.n_faces}, watertight: {m.is_watertight}\n"
                "If these measurements contradict the spec, fix the code and run it "
                "again; otherwise reply with a one-sentence summary of the part."
            )
        return (
            "FAILED — the script raised an error:\n"
            f"{result.error}\n"
            "Fix the code and call execute_cad_code again with the complete "
            "corrected script."
        )

    return agent
