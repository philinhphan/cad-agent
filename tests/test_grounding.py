"""Grounded-critic, panel, refuter, and advisory-gating behaviors (scripted models)."""

from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from cad_gen.agents.critic import build_critic_agent, run_critique
from cad_gen.agents.panel import _aggregate
from cad_gen.models import (
    Check,
    CheckReport,
    CheckStatus,
    ChecklistItem,
    Critique,
    DrawingTarget,
    ExecutionResult,
    GeometryMetrics,
    RunConfig,
)
from cad_gen.orchestrator import generate_cad

GOOD = "import cadquery as cq\nresult = cq.Workplane().box(10, 10, 10)"


# --------------------------------------------------------------------------- #
# Shared stubs
# --------------------------------------------------------------------------- #
def stub_executor(code: str, out_dir: Path, timeout_s: float = 60) -> ExecutionResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.stl").write_bytes(b"solid fake\nendsolid fake\n")
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


def stub_renderer(stl_path, out_png, metrics=None) -> Path:
    out_png = Path(out_png)
    out_png.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    return out_png


def scripted_generator(script: list, prompts: list | None = None) -> FunctionModel:
    state = {"i": 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if prompts is not None and len(messages) == 1:
            prompts.append(str(messages[0].parts[0].content))
        kind, payload = script[state["i"]]
        state["i"] += 1
        if kind == "tool":
            return ModelResponse(parts=[ToolCallPart("execute_cad_code", {"code": payload})])
        return ModelResponse(parts=[TextPart(payload)])

    return FunctionModel(fn)


def scripted_struct(args_list: list[dict], captured_text: list | None = None) -> FunctionModel:
    """Emit successive structured outputs (Critique / Refutation) via the output tool."""
    state = {"i": 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if captured_text is not None:
            content = messages[0].parts[0].content
            captured_text.append(content[0] if isinstance(content, list) else content)
        args = args_list[min(state["i"], len(args_list) - 1)]
        state["i"] += 1
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(fn)


def _critique(score, issues=None, **extra) -> dict:
    return {
        "matches_spec": score >= 8,
        "score": score,
        "issues": issues or [],
        "suggestions": [],
        "summary": f"scored {score}",
        **extra,
    }


def _execution() -> ExecutionResult:
    return ExecutionResult(
        success=True,
        code=GOOD,
        metrics=GeometryMetrics(
            volume_mm3=1000.0,
            bbox_mm=(10.0, 10.0, 10.0),
            center_of_mass=(0.0, 0.0, 0.0),
            n_solids=1,
            n_faces=6,
            is_watertight=True,
            mass_g=1.02,
        ),
        duration_s=0.1,
    )


# --------------------------------------------------------------------------- #
# Critic grounding
# --------------------------------------------------------------------------- #
async def test_critic_runs_at_temperature_zero(tmp_path):
    render = tmp_path / "v.png"
    render.write_bytes(b"\x89PNG\r\n\x1a\nx")
    captured: dict = {}

    def fn(messages, info):
        captured["settings"] = info.model_settings
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, _critique(7))])

    agent = build_critic_agent(FunctionModel(fn))
    await run_critique(agent, spec="x", execution=_execution(), render_path=render)

    assert captured["settings"] is not None
    assert captured["settings"].get("temperature") == 0.0


async def test_run_deterministic_falls_back_when_temperature_rejected():
    from pydantic_ai import Agent
    from pydantic_ai.exceptions import ModelHTTPError

    from cad_gen.agents._run import run_deterministic

    seen_settings: list = []

    def fn(messages, info):
        seen_settings.append(info.model_settings)
        if info.model_settings and info.model_settings.get("temperature") == 0.0:
            raise ModelHTTPError(
                status_code=400,
                model_name="gpt-x",
                body={"message": "temperature does not support 0.0"},
            )
        return ModelResponse(parts=[TextPart("ok")])

    agent = Agent(FunctionModel(fn), output_type=str)
    result = await run_deterministic(agent, "hi")
    # first attempt sends temp 0 (rejected), retry omits it (succeeds)
    assert seen_settings[0].get("temperature") == 0.0
    assert seen_settings[1] is None or seen_settings[1].get("temperature") is None
    assert result.output == "ok"


async def test_critic_prompt_includes_checks_and_target(tmp_path):
    render = tmp_path / "views.png"
    render.write_bytes(b"\x89PNG\r\n\x1a\nx")
    captured: list = []
    agent = build_critic_agent(scripted_struct([_critique(3, ["mass off"])], captured))

    report = CheckReport(
        checks=[
            Check(
                name="mass",
                status=CheckStatus.FAIL,
                critical=True,
                message="250 g vs 248 ± 1 g → 2.0 g too heavy",
            )
        ]
    )
    target = DrawingTarget(
        envelope_mm=(135.0, 85.0, 65.0), density_kg_m3=1020, target_mass_g=248, mass_tol_g=1
    )

    critique = await run_critique(
        agent,
        spec="",
        execution=_execution(),
        render_path=render,
        target=target,
        check_report=report,
    )

    assert critique.score == 3
    text = captured[0]
    assert "Deterministic checks" in text and "AUTHORITATIVE" in text
    assert "mass: FAIL" in text
    assert "Target extracted" in text and "135" in text


def test_critique_clamps_inconsistent_perfect_score():
    # A self-reported 10 cannot coexist with the critic's OWN failing checklist item.
    c = Critique(
        matches_spec=True,
        score=10,
        issues=[],
        suggestions=[],
        summary="x",
        checklist=[ChecklistItem(requirement="hole", status="fail")],
    )
    assert c.score == 9


# --------------------------------------------------------------------------- #
# Panel aggregation
# --------------------------------------------------------------------------- #
def test_panel_aggregate_is_conservative():
    a = Critique(
        matches_spec=True,
        score=9,
        issues=["i1"],
        suggestions=[],
        summary="A",
        checklist=[ChecklistItem(requirement="hole", status="pass")],
    )
    b = Critique(
        matches_spec=False,
        score=5,
        issues=["i2"],
        suggestions=[],
        summary="B",
        checklist=[ChecklistItem(requirement="hole", status="fail")],
    )
    agg = _aggregate([a, b], "min")
    assert agg.score == 5  # conservative min
    assert agg.matches_spec is False  # all panelists must agree
    assert set(agg.issues) == {"i1", "i2"}
    assert next(i for i in agg.checklist if i.requirement == "hole").status == "fail"
    assert agg.summary == "B"  # worst panelist headlines


def test_panel_median_aggregation():
    cs = [_make_critique(s) for s in (4, 7, 9)]
    assert _aggregate(cs, "median").score == 7


def _make_critique(score: int) -> Critique:
    return Critique(matches_spec=score >= 8, score=score, issues=[], suggestions=[], summary="x")


# --------------------------------------------------------------------------- #
# Orchestrator: advisory gating + refuter
# --------------------------------------------------------------------------- #
async def test_failed_mass_check_is_advisory_not_blocking(tmp_path):
    """A perfect critic score is accepted even when a critical mass check FAILS — the
    deterministic checks are advisory. The failure is recorded for the UI/feedback."""
    config = RunConfig(
        max_iterations=1,
        score_threshold=8,
        out_dir=tmp_path / "runs",
        density_kg_m3=1020,
        target_mass_g=248,  # stub mass is ~1.02 g → far outside ±1 g
        mass_tol_g=1,
    )

    result = await generate_cad(
        "a cube",
        config,
        generator_model=scripted_generator([("tool", GOOD), ("text", "built")]),
        critic_model=scripted_struct([_critique(10)]),
        executor=stub_executor,
        renderer=stub_renderer,
    )

    assert result.accepted is True  # advisory: checks never block acceptance
    assert result.best.passes_checks is False
    mass = next(c for c in result.best.check_report.checks if c.name == "mass")
    assert mass.status is CheckStatus.FAIL
    # the target overrides were applied even without a drawing
    assert result.target is not None and result.target.target_mass_g == 248


async def test_adversarial_refuter_caps_score_and_feeds_discrepancy(tmp_path):
    prompts: list = []
    generator = scripted_generator(
        [("tool", GOOD), ("text", "v1"), ("tool", GOOD), ("text", "v2")], prompts
    )
    critic = scripted_struct([_critique(10), _critique(10)])  # critic says perfect both times
    refuter = scripted_struct(
        [
            {
                "found_discrepancy": True,
                "discrepancies": ["counterbore on the wrong face"],
                "most_severe": "counterbore on the wrong face",
                "severity": "major",  # a wrong-face counterbore is major → still caps
            },
            {"found_discrepancy": False, "discrepancies": [], "most_severe": None},
        ]
    )
    config = RunConfig(
        max_iterations=3, score_threshold=8, out_dir=tmp_path / "runs", enable_adversarial=True
    )

    result = await generate_cad(
        "a cube",
        config,
        generator_model=generator,
        critic_model=critic,
        refuter_model=refuter,
        executor=stub_executor,
        renderer=stub_renderer,
    )

    # iter 1: critic 10 but refuter refuted → capped below threshold → not accepted yet
    assert result.iterations[0].critique.score == config.score_threshold - 1
    assert result.iterations[0].refutation.found_discrepancy is True
    # the discrepancy is carried into iteration 2's prompt as feedback
    assert "counterbore on the wrong face" in prompts[1]
    # iter 2: refuter concedes → score stands → accepted
    assert result.iterations[1].critique.score == 10
    assert result.accepted is True
    assert result.best.index == 2


async def test_minor_refutation_does_not_block_acceptance(tmp_path):
    """A graded MINOR discrepancy (a sub-mm nit) costs at most one point and must NOT
    forbid acceptance — otherwise a correct part can never cross threshold because a
    skeptic always finds something. This is the fix for the stuck-at-7 Tier-4 bracket."""
    generator = scripted_generator([("tool", GOOD), ("text", "v1")])
    critic = scripted_struct([_critique(9)])
    refuter = scripted_struct(
        [
            {
                "found_discrepancy": True,
                "discrepancies": ["slot cutter overshoots the pad by ~1 mm"],
                "most_severe": "slot cutter overshoots the pad by ~1 mm",
                "severity": "minor",
            }
        ]
    )
    config = RunConfig(
        max_iterations=1, score_threshold=8, out_dir=tmp_path / "runs", enable_adversarial=True
    )

    result = await generate_cad(
        "a cube",
        config,
        generator_model=generator,
        critic_model=critic,
        refuter_model=refuter,
        executor=stub_executor,
        renderer=stub_renderer,
    )

    # 9 − 1 (minor) = 8 ≥ threshold → accepted, and the nit is still recorded.
    assert result.iterations[0].critique.score == 8
    assert result.accepted is True
    assert any("overshoot" in i for i in result.iterations[0].critique.issues)
