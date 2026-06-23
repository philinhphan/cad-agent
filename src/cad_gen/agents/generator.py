"""Generator agent: writes CadQuery code and validates it via the sandbox tool."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from cad_gen.agents.prompts import GENERATOR_INSTRUCTIONS
from cad_gen.bmw import resolve_model
from cad_gen.models import ExecutionResult, IntrospectionResult, ReasoningEffort
from cad_gen.sandbox.executor import introspect_cad_code, run_cad_code

ExecutorFn = Callable[..., ExecutionResult]
IntrospectorFn = Callable[..., IntrospectionResult]


@dataclass
class IterationWorkspace:
    """Per-iteration execution state shared between orchestrator and tool.

    The last successful execution recorded here is the code of record for the
    iteration — never the text the agent returns. Read-only introspection probes
    are tracked separately (their own budget + subdirs) so they never affect
    `last_success` or the validation budget.
    """

    iter_dir: Path
    timeout_s: float = 60
    max_attempts: int = 4
    executor: ExecutorFn = run_cad_code
    attempts: list[ExecutionResult] = field(default_factory=list)
    introspector: IntrospectorFn = introspect_cad_code
    max_inspect: int = 8
    inspect_timeout_s: float = 30
    introspections: list[IntrospectionResult] = field(default_factory=list)

    @property
    def last_success(self) -> ExecutionResult | None:
        return next((r for r in reversed(self.attempts) if r.success), None)

    def execute(self, code: str) -> ExecutionResult:
        attempt_dir = self.iter_dir / f"attempt_{len(self.attempts) + 1:02d}"
        result = self.executor(code, attempt_dir, timeout_s=self.timeout_s)
        self.attempts.append(result)
        return result

    def introspect(self, code: str, query: dict) -> IntrospectionResult:
        probe_dir = self.iter_dir / f"inspect_{len(self.introspections) + 1:02d}"
        result = self.introspector(code, query, probe_dir, timeout_s=self.inspect_timeout_s)
        self.introspections.append(result)
        return result


_TEXT_SAMPLE = 4  # entities shown per group/selection in the LLM-facing text


def _fmt_pt(p: list) -> str:
    return "(" + ",".join(f"{v:g}" for v in p) + ")"


def _fmt_entity(e: dict, with_type: bool = True) -> str:
    parts = [e["type"]] if with_type else []
    if "center" in e:
        parts.append(f"center {_fmt_pt(e['center'])}")
    if "normal" in e:
        parts.append(f"normal {_fmt_pt(e['normal'])}")
    if "dir" in e:
        parts.append(f"dir {_fmt_pt(e['dir'])}")
    if "radius" in e:
        parts.append(f"r{e['radius']:g}")
    if "length" in e:
        parts.append(f"len {e['length']:g}")
    if "area" in e:
        parts.append(f"area {e['area']:g}")
    return " ".join(parts)


def _fmt_group(group: dict) -> str:
    sample = "; ".join(_fmt_entity(e, with_type=False) for e in group["sample"][:_TEXT_SAMPLE])
    more = ", ..." if group["count"] > _TEXT_SAMPLE else ""
    return f"  {group['type']} x{group['count']}: {sample}{more}"


def format_describe(data: dict) -> str:
    """Render a `describe` probe into a compact, LLM-readable topology summary."""
    bb = data["bbox_mm"]
    lines = [
        "GEOMETRY OF THE BUILT MODEL (read-only probe — nothing was exported):",
        f"bbox {bb[0]:g} x {bb[1]:g} x {bb[2]:g} mm | "
        f"solids {data['n_solids']}, faces {data['n_faces']}, edges {data['n_edges']}",
        "Faces:",
        *[_fmt_group(g) for g in data["faces"]],
        "Edges:",
        *[_fmt_group(g) for g in data["edges"]],
        "Use these coordinates to write a selector that matches the edges/faces you intend "
        "to modify, then confirm it with check_selector before applying fillet/chamfer/shell.",
    ]
    return "\n".join(lines)


def format_selector(data: dict) -> str:
    """Render a `selector` probe into a verdict the model can act on."""
    target, sel, count = data["target"], data["selector"], data["count"]
    head = f'.{target}("{sel}") matched {count} {target}'
    if data.get("selector_error"):
        return (
            f"{head} — the selector RAISED {data['selector_error']}\n"
            "Fix the selector (it is malformed or out of range); do not apply a fillet/"
            "chamfer/shell with it."
        )
    if count == 0:
        return (
            f"{head}. This selection is EMPTY — .fillet()/.chamfer()/.shell() on it would "
            "crash with 'requires that edges be selected'. Choose a different selector."
        )
    sample = data.get("matches", [])
    lines = [f"{head}:"]
    lines += [f"  - {_fmt_entity(e)}" for e in sample[:_TEXT_SAMPLE]]
    if count > _TEXT_SAMPLE:
        lines.append(f"  ... and {count - _TEXT_SAMPLE} more")
    lines.append(
        "If this is exactly the set you intend to modify, apply the op; otherwise refine "
        "the selector."
    )
    return "\n".join(lines)


def build_generator_agent(
    model: str | Model, *, reasoning_effort: ReasoningEffort | None = None
) -> Agent[IterationWorkspace, str]:
    model = resolve_model(model)  # route `bmw:...` strings to the BMW gateway
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

    def _probe(ctx: RunContext[IterationWorkspace], code: str, query: dict) -> str | None:
        """Run a read-only probe; return an error/budget message, or None on success."""
        ws = ctx.deps
        if len(ws.introspections) >= ws.max_inspect:
            return (
                "INSPECTION BUDGET EXHAUSTED: stop probing and commit to a script with "
                "execute_cad_code."
            )
        result = ws.introspect(code, query)
        if not result.ok or result.data is None:
            return (
                "The probe could not build your code:\n"
                f"{result.error}\n"
                "Fix the code, then probe or run it again."
            )
        return None

    @agent.tool
    def inspect_geometry(ctx: RunContext[IterationWorkspace], code: str) -> str:
        """Read-only probe: build `code` and report its solids/faces/edges with
        coordinates, WITHOUT exporting or scoring. Use it to understand the topology
        you actually built before selecting edges/faces to modify. Does not consume the
        execute_cad_code attempt budget.
        """
        err = _probe(ctx, code, {"mode": "describe"})
        return err if err is not None else format_describe(ctx.deps.introspections[-1].data)

    @agent.tool
    def check_selector(
        ctx: RunContext[IterationWorkspace], code: str, target: str, selector: str
    ) -> str:
        """Read-only probe: build `code`, then report which `target` ('edges' or 'faces')
        the CadQuery string `selector` matches, with their coordinates. Use this to VERIFY
        a selector before .fillet()/.chamfer()/.shell()/edge-cut — an empty or malformed
        selection crashes the real script. Does not consume the execute_cad_code budget.
        """
        query = {"mode": "selector", "target": target, "selector": selector}
        err = _probe(ctx, code, query)
        return err if err is not None else format_selector(ctx.deps.introspections[-1].data)

    return agent
