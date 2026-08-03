"""Generator agent: writes CAD code and validates it via the sandbox tool."""

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from cad_gen.agents.prompts import generator_instructions
from cad_gen.bmw import resolve_model
from cad_gen.models import (
    DEFAULT_LIBRARY,
    CadLibrary,
    ExecutionResult,
    IntrospectionResult,
    ReasoningEffort,
)
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
    # Files seeded into every execution/probe dir before the code runs (editing mode
    # drops the base model here, e.g. {"input.step": <bytes>}). Empty for generation.
    seed_files: dict[str, bytes] = field(default_factory=dict)
    # CAD library the generated code is written in; selects the sandbox harness.
    library: CadLibrary = DEFAULT_LIBRARY
    # Guard for slot reservation and the result lists. pydantic-ai dispatches sync tool
    # functions to a thread pool, so two tool calls in one model turn genuinely race here.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    _counters: dict[str, int] = field(default_factory=dict, repr=False, compare=False)

    @property
    def last_success(self) -> ExecutionResult | None:
        return next((r for r in reversed(self.attempts) if r.success), None)

    def _extra_kwargs(self) -> dict:
        """Optional executor kwargs, omitted when they carry no information.

        Both `seed_files` and `library` are passed ONLY when they differ from the default,
        so the plain generation call keeps its original signature and executor/introspector
        test-doubles that accept neither kwarg keep working.
        """
        extra: dict = {}
        if self.seed_files:
            extra["seed_files"] = self.seed_files
        if self.library != DEFAULT_LIBRARY:
            extra["library"] = self.library
        return extra

    def _reserve(self, counter: str) -> int:
        """Atomically claim the next 1-based working-directory slot for `counter`.

        Deliberately NOT derived from len(attempts)/len(introspections): those lists are
        appended only AFTER the blocking subprocess returns, and pydantic-ai dispatches
        sync tool functions to a thread pool. Two tool calls emitted in one model turn
        therefore computed the same index and shared one working directory — where
        `introspect_cad_code` writes `query.json`, so the second probe clobbered the
        first and a subprocess asked for "describe" answered a "query" instead. That
        surfaced as `KeyError: 'bbox_mm'` / `KeyError: 'target'` and zeroed 12 samples
        of a CADGenBench run.
        """
        with self._lock:
            value = self._counters.get(counter, 0) + 1
            self._counters[counter] = value
            return value

    def execute(self, code: str) -> ExecutionResult:
        attempt_dir = self.iter_dir / f"attempt_{self._reserve('attempt'):02d}"
        result = self.executor(
            code, attempt_dir, timeout_s=self.timeout_s, **self._extra_kwargs()
        )
        with self._lock:
            self.attempts.append(result)
        return result

    def introspect(self, code: str, query: dict) -> IntrospectionResult:
        probe_dir = self.iter_dir / f"inspect_{self._reserve('inspect'):02d}"
        result = self.introspector(
            code, query, probe_dir, timeout_s=self.inspect_timeout_s, **self._extra_kwargs()
        )
        with self._lock:
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
    if "axis" in e:
        parts.append(f"axis {_fmt_pt(e['axis'])}")
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


def format_query(data: dict) -> str:
    """Render a `query` probe: what the filter found, and how much it did not show."""
    if data.get("query_error"):
        return f"find_geometry could not run: {data['query_error']}"
    target, count, total = data["target"], data["count"], data["total"]
    lo, hi = data["model_bbox_min_mm"], data["model_bbox_max_mm"]
    head = f"{count} of {total} {target} match this filter"
    if count == 0:
        return (
            f"{head}. Nothing matched — loosen the filter (a too-tight radius or normal "
            f"tolerance is the usual cause). Model bounds: {_fmt_pt(lo)} to {_fmt_pt(hi)} mm."
        )
    shown = data.get("matches", [])
    lines = [f"{head} (largest first, showing {len(shown)}):"]
    lines += [f"  - {_fmt_entity(e)}" for e in shown]
    if data.get("truncated"):
        lines.append(f"  ... and {count - len(shown)} more — narrow the filter or raise limit")
    lines.append(f"Model bounds: {_fmt_pt(lo)} to {_fmt_pt(hi)} mm.")
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


def format_selection(data: dict) -> str:
    """Render a build123d `selection` probe into a verdict the model can act on.

    The build123d counterpart of :func:`format_selector`: there are no string selectors, so
    the probe reports on an evaluated ShapeList expression instead of a `.faces(sel)` call.
    """
    expression, count = data["expression"], data["count"]
    head = f"{expression or '(empty)'} matched {count} shape(s)"
    if data.get("selection_error"):
        # The error may be a raised exception OR a well-formed expression that produced
        # something other than shapes, so state it plainly rather than claiming a raise.
        return (
            f"{expression or '(empty)'} is not a usable selection: "
            f"{data['selection_error']}\n"
            "Fix the expression — it must evaluate to a Shape or a ShapeList, e.g. end it "
            "with .edges()/.faces() or an index. Do not apply a fillet/chamfer/offset with it."
        )
    if count == 0:
        return (
            f"{head}. This selection is EMPTY — fillet()/chamfer() on it would raise. "
            "Choose a different expression."
        )
    sample = data.get("matches", [])
    lines = [f"{head}:"]
    lines += [f"  - {_fmt_entity(e)}" for e in sample[:_TEXT_SAMPLE]]
    if count > _TEXT_SAMPLE:
        lines.append(f"  ... and {count - _TEXT_SAMPLE} more")
    lines.append(
        "If this is exactly the set you intend to modify, apply the op; otherwise refine "
        "the expression."
    )
    return "\n".join(lines)


def build_generator_agent(
    model: str | Model,
    *,
    reasoning_effort: ReasoningEffort | None = None,
    editing: bool = False,
    library: CadLibrary = DEFAULT_LIBRARY,
) -> Agent[IterationWorkspace, str]:
    model = resolve_model(model)  # route `bmw:...` strings to the BMW gateway
    # `thinking` is pydantic-ai's provider-agnostic reasoning-effort knob; when unset we
    # pass no model_settings so the provider's own default is left untouched.
    model_settings = ModelSettings(thinking=reasoning_effort) if reasoning_effort else None
    # Editing mode swaps in instructions that permit importing the seeded base model and
    # frame the task as a minimal modification; `library` selects the CAD language taught.
    instructions = generator_instructions(library, editing=editing)
    agent: Agent[IterationWorkspace, str] = Agent(
        model,
        deps_type=IterationWorkspace,
        output_type=str,
        instructions=instructions,
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

    def _probe(
        ctx: RunContext[IterationWorkspace], code: str, query: dict
    ) -> tuple[str | None, dict | None]:
        """Run a read-only probe. Returns (error_message, data) — exactly one is set.

        The data is RETURNED rather than left for the caller to read back off
        `ws.introspections[-1]`. A model may emit several tool calls in one turn and
        pydantic-ai runs them concurrently, so `[-1]` can be a sibling probe's result:
        that is how a `check_selection` result reached `format_describe` and raised
        `KeyError: 'bbox_mm'` mid-benchmark.
        """
        ws = ctx.deps
        if len(ws.introspections) >= ws.max_inspect:
            return (
                "INSPECTION BUDGET EXHAUSTED: stop probing and commit to a script with "
                "execute_cad_code.",
                None,
            )
        result = ws.introspect(code, query)
        if not result.ok or result.data is None:
            return (
                "The probe could not build your code:\n"
                f"{result.error}\n"
                "Fix the code, then probe or run it again.",
                None,
            )
        # Belt and braces: the harness echoes back the mode it actually ran. A mismatch
        # means the payload does not match the formatter about to read it, which raises an
        # opaque KeyError deep in formatting and zeroes the sample. Fail loudly instead —
        # this catches both a shared-directory race and a harness that silently falls
        # through on a mode it does not implement.
        wanted = query.get("mode", "describe")
        got = result.data.get("mode")
        if got != wanted:
            return (
                f"Probe returned a {got!r} result for a {wanted!r} request — the "
                f"{ws.library} introspection harness does not support {wanted!r}. "
                "Use a different probe tool.",
                None,
            )
        return None, result.data

    @agent.tool
    def inspect_geometry(ctx: RunContext[IterationWorkspace], code: str) -> str:
        """Read-only probe: build `code` and report its solids/faces/edges with
        coordinates, WITHOUT exporting or scoring. Use it to understand the topology
        you actually built before selecting edges/faces to modify. Does not consume the
        execute_cad_code attempt budget.
        """
        err, data = _probe(ctx, code, {"mode": "describe"})
        return err if err is not None else format_describe(data)

    # The two libraries select geometry in fundamentally different ways — CadQuery with
    # string selectors, build123d with ShapeList expressions — so each gets the probe tool
    # that matches its API rather than a lowest-common-denominator one.
    if library == "build123d":

        @agent.tool
        def check_selection(
            ctx: RunContext[IterationWorkspace], code: str, expression: str
        ) -> str:
            """Read-only probe: build `code`, then evaluate a build123d ShapeList
            `expression` against it (e.g. `result.edges().filter_by(Axis.Z)` or
            `result.faces().sort_by(Axis.Z)[-1]`) and report what it matches, with
            coordinates. Use this to VERIFY a selection before fillet()/chamfer()/offset()
            or an edge/face-based cut — fillet() and chamfer() raise on an empty ShapeList.
            Does not consume the execute_cad_code budget.
            """
            query = {"mode": "selection", "expression": expression}
            err, data = _probe(ctx, code, query)
            return err if err is not None else format_selection(data)

    else:

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
            err, data = _probe(ctx, code, query)
            return err if err is not None else format_selector(data)

    # Editing only. `inspect_geometry` samples 12 entities per geometry type in traversal
    # order, which is plenty for a part the model just built and useless on an imported base
    # carrying 334-2157 faces — there, the sample is effectively random and the feature the
    # instruction names is almost certainly not in it. Registered only in editing mode so the
    # generation agent's tool set is untouched.
    if editing:

        @agent.tool
        def find_geometry(
            ctx: RunContext[IterationWorkspace],
            code: str,
            target: str = "faces",
            geom_type: str | None = None,
            area_min: float | None = None,
            area_max: float | None = None,
            radius_min: float | None = None,
            radius_max: float | None = None,
            normal: list[float] | None = None,
            center_box: list[float] | None = None,
            limit: int = 20,
        ) -> str:
            """Read-only probe: build `code`, then FIND the faces or edges matching a filter,
            largest first, with their coordinates. This is how you locate a feature on an
            imported base model — `inspect_geometry` only samples a handful per type.

            `target`: 'faces' (default) or 'edges'.
            `geom_type`: 'PLANE', 'CYLINDER', 'CONE', 'CIRCLE', 'LINE', 'BSPLINE', ...
            `area_min`/`area_max`: mm2, faces only.
            `radius_min`/`radius_max`: mm — matches CYLINDER/CONE faces and CIRCLE edges, so
                this is how you find a bore of a known size ("the largest-diameter bore").
            `normal`: unit vector, e.g. [1,0,0] for faces looking along +X (within ~8 deg).
            `center_box`: [xmin,ymin,zmin,xmax,ymax,zmax] — restrict to a region, e.g. the
                +X half of the part.
            `limit`: how many matches to show (the full match count is always reported).

            Reports the model's absolute bounding box too, so you can position a cutting
            primitive by coordinate. Does not consume the execute_cad_code budget.
            """
            where: dict = {}
            if geom_type:
                where["type"] = geom_type
            for key, value in (
                ("area_min", area_min),
                ("area_max", area_max),
                ("radius_min", radius_min),
                ("radius_max", radius_max),
                ("normal", normal),
                ("center_box", center_box),
            ):
                if value is not None:
                    where[key] = value
            query = {"mode": "query", "target": target, "where": where, "limit": limit}
            err, data = _probe(ctx, code, query)
            return err if err is not None else format_query(data)

    return agent
