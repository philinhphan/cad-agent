"""Generator agent: writes CadQuery code and validates it via the sandbox tool."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from cad_gen.agents.prompts import GENERATOR_INSTRUCTIONS
from cad_gen.models import ExecutionResult
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
    # Optional target context: lets the execute tool report measured mass vs target.
    density_kg_m3: float | None = None
    target_mass_g: float | None = None
    mass_tol_g: float | None = None
    attempts: list[ExecutionResult] = field(default_factory=list)

    @property
    def last_success(self) -> ExecutionResult | None:
        return next((r for r in reversed(self.attempts) if r.success), None)

    def execute(self, code: str) -> ExecutionResult:
        attempt_dir = self.iter_dir / f"attempt_{len(self.attempts) + 1:02d}"
        result = self.executor(code, attempt_dir, timeout_s=self.timeout_s)
        self.attempts.append(result)
        return result


def build_generator_agent(model: str | Model) -> Agent[IterationWorkspace, str]:
    agent: Agent[IterationWorkspace, str] = Agent(
        model,
        deps_type=IterationWorkspace,
        output_type=str,
        instructions=GENERATOR_INSTRUCTIONS,
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
            mass_line = ""
            if ws.density_kg_m3 is not None:
                mass = m.volume_mm3 * ws.density_kg_m3 * 1e-6
                m.mass_g = mass
                if ws.target_mass_g is not None:
                    tol = ws.mass_tol_g if ws.mass_tol_g is not None else max(0.5, 0.01 * ws.target_mass_g)
                    delta = mass - ws.target_mass_g
                    verdict = (
                        "within tolerance"
                        if abs(delta) <= tol
                        else f"{abs(delta):.2f} g too {'heavy' if delta > 0 else 'light'}"
                    )
                    mass_line = f"mass: {mass:.2f} g (target {ws.target_mass_g:.2f} ± {tol:.2f} g — {verdict})\n"
                else:
                    mass_line = f"mass: {mass:.2f} g (no target mass given)\n"
            return (
                "SUCCESS — the model built and exported.\n"
                f"volume: {m.volume_mm3:.1f} mm3\n"
                f"bbox: {m.bbox_mm[0]:.2f} x {m.bbox_mm[1]:.2f} x {m.bbox_mm[2]:.2f} mm\n"
                f"{mass_line}"
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
