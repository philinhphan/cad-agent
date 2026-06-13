from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from cad_gen.agents.critic import build_critic_agent, run_critique
from cad_gen.agents.generator import IterationWorkspace, build_generator_agent
from cad_gen.models import ExecutionResult, GeometryMetrics

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
