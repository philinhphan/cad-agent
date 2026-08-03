from pathlib import Path

from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from cad_gen.agents.critic import build_critic_agent, run_critique
from cad_gen.agents.drawing_parser import build_drawing_parser_agent, interpret_drawing
from cad_gen.agents.generator import (
    IterationWorkspace,
    build_generator_agent,
    format_describe,
    format_selection,
    format_selector,
)
from cad_gen.models import (
    DrawingAttachment,
    ExecutionResult,
    GeometryMetrics,
    IntrospectionResult,
)

PNG = b"\x89PNG\r\n\x1a\nfakepngdata"


def _images_in(messages: list[ModelMessage]) -> list[BinaryContent]:
    """Collect every BinaryContent the model was sent (across all message parts)."""
    found: list[BinaryContent] = []
    for m in messages:
        for part in getattr(m, "parts", []):
            content = getattr(part, "content", None)
            if isinstance(content, list):
                found.extend(c for c in content if isinstance(c, BinaryContent))
    return found

BAD_CODE = "result = cq.Workplane().box(undefined, 1, 1)"
GOOD_CODE = "import cadquery as cq\nresult = cq.Workplane().box(10, 10, 10)"


def fake_executor_factory(call_log: list):
    """Stub executor: succeeds iff the code contains 'box(10'."""

    def fake_executor(code: str, out_dir: Path, timeout_s: float = 60) -> ExecutionResult:
        call_log.append(code)
        if "box(10" in code:
            return ExecutionResult(
                success=True,
                code=code,
                metrics=GeometryMetrics(
                    volume_mm3=1000.0,
                    bbox_mm=(10.0, 10.0, 10.0),
                    center_of_mass=(0.0, 0.0, 0.0),
                    n_solids=1,
                    n_faces=6,
                    is_watertight=True,
                ),
                stl_path=out_dir / "model.stl",
                step_path=out_dir / "model.step",
                duration_s=0.01,
            )
        return ExecutionResult(
            success=False,
            code=code,
            error="Traceback (most recent call last):\nNameError: name 'undefined' is not defined",
            duration_s=0.01,
        )

    return fake_executor


def scripted_generator_model(tool_codes: list[str], final_text: str) -> FunctionModel:
    """A model that emits one execute_cad_code call per entry, then final text."""
    state = {"i": 0}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        i = state["i"]
        state["i"] += 1
        if i < len(tool_codes):
            return ModelResponse(
                parts=[ToolCallPart("execute_cad_code", {"code": tool_codes[i]})]
            )
        return ModelResponse(parts=[TextPart(final_text)])

    return FunctionModel(model_fn)


async def test_generator_retries_after_failure_then_succeeds(tmp_path):
    calls: list[str] = []
    ws = IterationWorkspace(
        iter_dir=tmp_path,
        timeout_s=60,
        max_attempts=4,
        executor=fake_executor_factory(calls),
    )
    model = scripted_generator_model([BAD_CODE, GOOD_CODE], "Built a 10mm cube.")
    agent = build_generator_agent(model)

    result = await agent.run("a 10mm cube", deps=ws)

    assert calls == [BAD_CODE, GOOD_CODE]
    assert len(ws.attempts) == 2
    assert not ws.attempts[0].success
    assert ws.last_success is not None
    assert ws.last_success.code == GOOD_CODE
    assert result.output == "Built a 10mm cube."


async def test_generator_attempt_budget_blocks_further_executions(tmp_path):
    calls: list[str] = []
    ws = IterationWorkspace(
        iter_dir=tmp_path,
        timeout_s=60,
        max_attempts=1,
        executor=fake_executor_factory(calls),
    )
    model = scripted_generator_model([BAD_CODE, BAD_CODE], "Could not build it.")
    agent = build_generator_agent(model)

    await agent.run("a 10mm cube", deps=ws)

    assert calls == [BAD_CODE], "executor must not run past the attempt budget"
    assert len(ws.attempts) == 1
    assert ws.last_success is None


def test_reasoning_effort_sets_thinking_model_setting():
    # pydantic-ai's provider-agnostic `thinking` ModelSettings field carries the effort.
    agent = build_generator_agent(scripted_generator_model([], "x"), reasoning_effort="high")
    assert agent.model_settings == {"thinking": "high"}


def test_no_reasoning_effort_leaves_model_settings_unset():
    # No model_settings injected → the provider's own default is left untouched.
    agent = build_generator_agent(scripted_generator_model([], "x"))
    assert agent.model_settings is None


def test_critic_reasoning_effort_sets_thinking_model_setting():
    # The Gemini critic gets the same provider-agnostic `thinking` knob as the generator.
    agent = build_critic_agent(scripted_generator_model([], "x"), reasoning_effort="high")
    assert agent.model_settings == {"thinking": "high"}


def test_critic_no_reasoning_effort_leaves_model_settings_unset():
    agent = build_critic_agent(scripted_generator_model([], "x"))
    assert agent.model_settings is None


async def test_attempts_run_in_separate_subdirectories(tmp_path):
    calls: list[str] = []
    seen_dirs: list[Path] = []

    base = fake_executor_factory(calls)

    def tracking_executor(code, out_dir, timeout_s=60):
        seen_dirs.append(Path(out_dir))
        return base(code, out_dir, timeout_s)

    ws = IterationWorkspace(
        iter_dir=tmp_path, timeout_s=60, max_attempts=4, executor=tracking_executor
    )
    model = scripted_generator_model([BAD_CODE, GOOD_CODE], "done")
    agent = build_generator_agent(model)

    await agent.run("a cube", deps=ws)

    assert len(seen_dirs) == 2
    assert len(set(seen_dirs)) == 2, "each attempt gets its own directory"
    assert all(d.is_relative_to(tmp_path) for d in seen_dirs)


async def test_critic_returns_structured_critique(tmp_path):
    render = tmp_path / "views.png"
    render.write_bytes(b"\x89PNG\r\n\x1a\nfakedata")
    execution = fake_executor_factory([])(GOOD_CODE, tmp_path)

    model = TestModel(
        custom_output_args={
            "matches_spec": False,
            "score": 4,
            "issues": ["hole is missing"],
            "suggestions": ["add .faces('>Z').hole(10)"],
            "summary": "Cube ok but the required hole is absent.",
        }
    )
    agent = build_critic_agent(model)

    critique = await run_critique(
        agent, spec="a 10mm cube with a hole", execution=execution, render_path=render
    )

    assert critique.score == 4
    assert critique.matches_spec is False
    assert "hole" in critique.issues[0]


async def test_drawing_parser_returns_text_and_receives_image():
    captured: list[ModelMessage] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.extend(messages)
        return ModelResponse(parts=[TextPart("85 x 135 x 25 plate, R20 corners, Ø15 THRU")])

    agent = build_drawing_parser_agent(FunctionModel(model_fn))
    drawings = [DrawingAttachment(filename="d.png", media_type="image/png", data=PNG)]

    out = await interpret_drawing(agent, spec="a bracket", drawings=drawings)

    assert "Ø15" in out
    images = _images_in(captured)
    assert len(images) == 1
    assert images[0].media_type == "image/png"


async def test_critic_receives_render_then_input_drawings(tmp_path):
    render = tmp_path / "views.png"
    render.write_bytes(b"\x89PNG\r\n\x1a\nfakerender")
    execution = fake_executor_factory([])(GOOD_CODE, tmp_path)
    captured: list[ModelMessage] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured.extend(messages)
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {
                        "matches_spec": True,
                        "score": 9,
                        "issues": [],
                        "suggestions": [],
                        "summary": "matches the drawing",
                    },
                )
            ]
        )

    agent = build_critic_agent(FunctionModel(model_fn))
    drawings = [DrawingAttachment(filename="d.jpg", media_type="image/jpeg", data=PNG)]

    critique = await run_critique(
        agent, spec="", execution=execution, render_path=render, drawings=drawings
    )

    assert critique.score == 9
    images = _images_in(captured)
    # first image is the render, then each input drawing
    assert len(images) == 2
    assert images[0].media_type == "image/png"  # the rendered views
    assert images[1].media_type == "image/jpeg"  # the input drawing


# --- Introspection tools (B) -------------------------------------------------

SELECTOR_DATA = {
    "mode": "selector", "target": "edges", "selector": "|Z", "count": 4,
    "matches": [{"type": "LINE", "center": [10.0, 0, 0], "dir": [0, 0, 1.0], "length": 10.0}],
    "truncated": False, "selector_error": None,
}
DESCRIBE_DATA = {
    "mode": "describe", "bbox_mm": [10.0, 10.0, 10.0],
    "n_solids": 1, "n_faces": 6, "n_edges": 12,
    "faces": [{"type": "PLANE", "count": 6,
               "sample": [{"type": "PLANE", "center": [0, 0, 5.0], "normal": [0, 0, 1.0],
                           "area": 100.0}], "truncated": False}],
    "edges": [{"type": "LINE", "count": 12,
               "sample": [{"type": "LINE", "center": [0, 0, 0], "dir": [0, 0, 1.0],
                           "length": 10.0}], "truncated": False}],
}


def fake_introspector_factory(call_log: list):
    """Stub introspector: records calls, returns canned data per query mode."""

    def fake(code: str, query: dict, out_dir, timeout_s: float = 30) -> IntrospectionResult:
        call_log.append((code, query))
        data = SELECTOR_DATA if query.get("mode") == "selector" else DESCRIBE_DATA
        return IntrospectionResult(ok=True, data=data)

    return fake


def scripted_multitool_model(script: list) -> FunctionModel:
    """`script` entries are (tool_name, args) or ("text", message), consumed in order."""
    state = {"i": 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        name, payload = script[state["i"]]
        state["i"] += 1
        if name == "text":
            return ModelResponse(parts=[TextPart(payload)])
        return ModelResponse(parts=[ToolCallPart(name, payload)])

    return FunctionModel(fn)


async def test_probe_tools_do_not_consume_execute_budget(tmp_path):
    exec_calls: list[str] = []
    probe_calls: list = []
    ws = IterationWorkspace(
        iter_dir=tmp_path,
        max_attempts=1,  # a single execute attempt allowed
        executor=fake_executor_factory(exec_calls),
        introspector=fake_introspector_factory(probe_calls),
    )
    model = scripted_multitool_model([
        ("check_selector", {"code": GOOD_CODE, "target": "edges", "selector": "|Z"}),
        ("inspect_geometry", {"code": GOOD_CODE}),
        ("execute_cad_code", {"code": GOOD_CODE}),
        ("text", "Built a 10mm cube with verified fillet edges."),
    ])
    agent = build_generator_agent(model)

    result = await agent.run("a 10mm cube", deps=ws)

    # two probes ran but cost nothing from the execute budget
    assert len(probe_calls) == 2
    assert len(ws.introspections) == 2
    assert exec_calls == [GOOD_CODE], "execute ran once despite max_attempts=1"
    assert len(ws.attempts) == 1
    assert ws.last_success is not None and ws.last_success.code == GOOD_CODE
    assert result.output.startswith("Built a 10mm cube")


async def test_probe_budget_blocks_further_probes(tmp_path):
    probe_calls: list = []
    ws = IterationWorkspace(
        iter_dir=tmp_path,
        max_attempts=4,
        max_inspect=1,  # only one probe allowed
        executor=fake_executor_factory([]),
        introspector=fake_introspector_factory(probe_calls),
    )
    model = scripted_multitool_model([
        ("check_selector", {"code": GOOD_CODE, "target": "edges", "selector": "|Z"}),
        ("check_selector", {"code": GOOD_CODE, "target": "edges", "selector": ">Z"}),
        ("execute_cad_code", {"code": GOOD_CODE}),
        ("text", "done"),
    ])
    agent = build_generator_agent(model)

    await agent.run("a cube", deps=ws)

    assert len(probe_calls) == 1, "the introspector must not run past the inspect budget"
    assert len(ws.introspections) == 1


def test_format_selector_flags_empty_and_error():
    ok = format_selector(SELECTOR_DATA)
    assert '.edges("|Z") matched 4 edges' in ok

    empty = format_selector({**SELECTOR_DATA, "count": 0, "matches": []})
    assert "EMPTY" in empty and "crash" in empty

    err = format_selector({**SELECTOR_DATA, "count": 0, "selector_error": "IndexError: boom"})
    assert "RAISED IndexError: boom" in err


def test_format_describe_summarizes_topology():
    out = format_describe(DESCRIBE_DATA)
    assert "solids 1, faces 6, edges 12" in out
    assert "PLANE x6" in out and "LINE x12" in out


# --- CAD library selection -----------------------------------------------------------


async def _agent_info(tmp_path, **kwargs) -> AgentInfo:
    """Run the generator once and capture what it actually exposed to the model.

    Asserting on AgentInfo rather than the Agent's private attributes keeps these tests
    pinned to the tools and instructions the model really receives.
    """
    captured: dict[str, AgentInfo] = {}

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        captured["info"] = info
        return ModelResponse(parts=[TextPart("done")])

    agent = build_generator_agent(FunctionModel(model_fn), **kwargs)
    await agent.run("a 10mm cube", deps=IterationWorkspace(iter_dir=tmp_path))
    return captured["info"]


async def test_generator_registers_the_probe_tool_matching_the_library(tmp_path):
    """The two libraries select geometry differently, so they get different probe tools.

    Offering a CadQuery string-selector tool to a build123d model (or vice versa) invites
    calls that can never succeed.
    """
    cq_tools = {t.name for t in (await _agent_info(tmp_path)).function_tools}
    b3d_tools = {
        t.name
        for t in (await _agent_info(tmp_path, library="build123d")).function_tools
    }

    assert "check_selector" in cq_tools and "check_selection" not in cq_tools
    assert "check_selection" in b3d_tools and "check_selector" not in b3d_tools
    # Everything else is shared.
    assert {"execute_cad_code", "inspect_geometry"} <= cq_tools
    assert {"execute_cad_code", "inspect_geometry"} <= b3d_tools


async def test_generator_instructions_follow_the_library(tmp_path):
    cq = (await _agent_info(tmp_path)).instructions or ""
    b3d = (await _agent_info(tmp_path, library="build123d")).instructions or ""

    assert "CadQuery" in cq
    assert "build123d" in b3d and "CadQuery" not in b3d


def test_workspace_omits_library_kwarg_at_the_default(tmp_path):
    """Executor doubles in this suite take no **kwargs, so the default call must be bare.

    `seed_files` already relies on this; `library` must not regress it.
    """
    seen: list[dict] = []

    def picky_executor(code, out_dir, timeout_s=60):
        seen.append({"timeout_s": timeout_s})
        return ExecutionResult(success=True, code=code, duration_s=0.01)

    ws = IterationWorkspace(iter_dir=tmp_path, executor=picky_executor)
    ws.execute("result = None")

    assert seen == [{"timeout_s": 60}]


def test_workspace_passes_library_when_not_default(tmp_path):
    seen: list[dict] = []

    def executor(code, out_dir, timeout_s=60, **kwargs):
        seen.append(kwargs)
        return ExecutionResult(success=True, code=code, duration_s=0.01)

    ws = IterationWorkspace(iter_dir=tmp_path, executor=executor, library="build123d")
    ws.execute("result = None")

    assert seen == [{"library": "build123d"}]


def test_format_selection_renders_matches_and_errors():
    data = {
        "expression": "result.edges().filter_by(Axis.Z)",
        "count": 4,
        "matches": [{"type": "LINE", "center": [0, 0, 0], "length": 10}],
        "selection_error": None,
    }
    ok = format_selection(data)
    assert "result.edges().filter_by(Axis.Z) matched 4 shape(s)" in ok

    empty = format_selection({**data, "count": 0, "matches": []})
    assert "EMPTY" in empty

    # A well-formed expression returning a non-Shape did not raise — say so accurately.
    bad = format_selection(
        {**data, "count": 0, "selection_error": "expression produced float, ..."}
    )
    assert "not a usable selection" in bad
    assert "RAISED" not in bad


async def test_concurrent_probes_do_not_cross_results(tmp_path):
    """Regression: two probes in ONE model turn must not read each other's data.

    pydantic-ai runs tool calls emitted in the same response concurrently. The tools used
    to read `ws.introspections[-1]` after probing, so a sibling probe finishing in between
    handed `format_describe` a selection payload — observed live as
    `KeyError: 'bbox_mm'`, which zeroed a CADGenBench sample.
    """
    describe_data = {"mode": "describe", **DESCRIBE_DATA}
    selector_data = {"mode": "selector", **SELECTOR_DATA}
    # Serve describe first, then the selector payload: whichever tool reads back the
    # shared list last would see the wrong one.
    payloads = [describe_data, selector_data]

    def introspector(code, query, out_dir, timeout_s=30, **kwargs):
        return IntrospectionResult(ok=True, data=payloads.pop(0))

    ws = IterationWorkspace(iter_dir=tmp_path, introspector=introspector)

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:  # first turn: emit BOTH probes together
            return ModelResponse(parts=[
                ToolCallPart("inspect_geometry", {"code": GOOD_CODE}),
                ToolCallPart("check_selector",
                             {"code": GOOD_CODE, "target": "edges", "selector": "|Z"}),
            ])
        return ModelResponse(parts=[TextPart("done")])

    agent = build_generator_agent(FunctionModel(model_fn))
    result = await agent.run("a 10mm cube", deps=ws)

    assert result.output == "done"
    assert len(ws.introspections) == 2


def test_concurrent_probes_get_distinct_working_directories(tmp_path):
    """Regression: two probes in one model turn must not share a working directory.

    pydantic-ai dispatches sync tool functions to a thread pool. The slot index used to
    come from len(introspections), which is only appended AFTER the blocking subprocess
    returns — so concurrent probes computed the same index and shared one directory.
    `introspect_cad_code` writes `query.json` there, so the second probe clobbered the
    first and a subprocess asked for "describe" answered a "query". Observed live as
    KeyError: 'bbox_mm' / 'target', which zeroed 12 CADGenBench samples.
    """
    import concurrent.futures as cf
    import time

    seen: list[str] = []

    def slow_introspector(code, query, out_dir, timeout_s=30, **kwargs):
        seen.append(Path(out_dir).name)
        time.sleep(0.05)  # hold the slot open, like a real subprocess
        return IntrospectionResult(ok=True, data={"mode": query["mode"]})

    ws = IterationWorkspace(iter_dir=tmp_path, introspector=slow_introspector)
    modes = ["describe", "selector", "query", "describe"]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda m: ws.introspect("code", {"mode": m}), modes))

    assert len(seen) == 4
    assert len(set(seen)) == 4, f"probe dirs collided: {seen}"
    assert len(ws.introspections) == 4


def test_concurrent_executions_get_distinct_working_directories(tmp_path):
    """Same reservation bug on the execute path: attempts must not share a directory."""
    import concurrent.futures as cf
    import time

    seen: list[str] = []

    def slow_executor(code, out_dir, timeout_s=60, **kwargs):
        seen.append(Path(out_dir).name)
        time.sleep(0.05)
        return ExecutionResult(success=True, code=code, duration_s=0.01)

    ws = IterationWorkspace(iter_dir=tmp_path, executor=slow_executor)
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda c: ws.execute(c), ["a", "b", "c"]))

    assert len(set(seen)) == 3, f"attempt dirs collided: {seen}"
    assert len(ws.attempts) == 3


async def test_probe_rejects_a_mode_the_harness_did_not_run(tmp_path):
    """A harness that falls through on an unsupported mode must not reach the formatter.

    build123d's introspection harness has no 'query' mode and silently returned a
    describe payload, which `format_query` then read for data['target'].
    """
    def wrong_mode_introspector(code, query, out_dir, timeout_s=30, **kwargs):
        # Answer every request with a describe payload, as the fall-through did.
        return IntrospectionResult(ok=True, data={"mode": "describe", **DESCRIBE_DATA})

    ws = IterationWorkspace(
        iter_dir=tmp_path, introspector=wrong_mode_introspector, library="build123d"
    )

    captured: list[str] = []

    def model_fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for m in messages:
            for part in getattr(m, "parts", []):
                if type(part).__name__ == "ToolReturnPart":
                    captured.append(str(part.content))
        if len(messages) == 1:
            return ModelResponse(parts=[
                ToolCallPart("find_geometry", {"code": GOOD_CODE, "target": "faces"}),
            ])
        return ModelResponse(parts=[TextPart("done")])

    agent = build_generator_agent(FunctionModel(model_fn), editing=True, library="build123d")
    result = await agent.run("move the +X wall", deps=ws)

    assert result.output == "done"
    # The tool reported the mismatch instead of raising KeyError deep in formatting.
    assert any("does not support" in c for c in captured), captured
