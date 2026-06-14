"""Self-refine loop tests with fully scripted models — no LLM, no CadQuery."""

import json
from pathlib import Path

from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from cad_gen.models import (
    DrawingAttachment,
    ExecutionResult,
    GeometryMetrics,
    RunConfig,
)
from cad_gen.orchestrator import generate_cad

PNG = b"\x89PNG\r\n\x1a\nfakepng"

GOOD_V1 = "import cadquery as cq\nresult = cq.Workplane().box(10, 10, 10)  # v1"
GOOD_V2 = "import cadquery as cq\nresult = cq.Workplane().box(10, 10, 10).faces('>Z').hole(4)  # v2"
BAD = "result = cq.Workplane().box(undefined, 1, 1)"


def stub_executor(code: str, out_dir: Path, timeout_s: float = 60) -> ExecutionResult:
    """Succeeds unless the code mentions `undefined`; writes real dummy artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model.py").write_text(code)
    if "undefined" in code:
        return ExecutionResult(
            success=False, code=code, error="NameError: name 'undefined' is not defined",
            duration_s=0.01,
        )
    (out_dir / "model.stl").write_bytes(b"solid fake\nendsolid fake\n")
    (out_dir / "model.step").write_bytes(b"ISO-10303-21;")
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


def scripted_generator(script: list, prompts_seen: list[str]) -> FunctionModel:
    """`script` entries: ("tool", code) or ("text", message), consumed across runs."""
    state = {"i": 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        first = messages[0].parts[0]
        if len(messages) == 1:  # first model call of an agent.run
            prompts_seen.append(str(first.content))
        kind, payload = script[state["i"]]
        state["i"] += 1
        if kind == "tool":
            return ModelResponse(parts=[ToolCallPart("execute_cad_code", {"code": payload})])
        return ModelResponse(parts=[TextPart(payload)])

    return FunctionModel(fn)


def capturing_generator(
    script: list, first_contents: list, images: list
) -> FunctionModel:
    """Like scripted_generator but records the first user-prompt content object and any
    images the generator was sent, so tests can assert how the prompt was built."""
    state = {"i": 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if len(messages) == 1:
            content = messages[0].parts[0].content
            first_contents.append(content)
            if isinstance(content, list):
                images.extend(c for c in content if isinstance(c, BinaryContent))
        kind, payload = script[state["i"]]
        state["i"] += 1
        if kind == "tool":
            return ModelResponse(parts=[ToolCallPart("execute_cad_code", {"code": payload})])
        return ModelResponse(parts=[TextPart(payload)])

    return FunctionModel(fn)


def scripted_text_model(text: str) -> FunctionModel:
    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(text)])

    return FunctionModel(fn)


def scripted_critic(critiques: list[dict], calls: list[int]) -> FunctionModel:
    state = {"i": 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        args = critiques[state["i"]]
        state["i"] += 1
        calls.append(state["i"])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(fn)


def critique_args(score: int, issues: list[str]) -> dict:
    return {
        "matches_spec": score >= 8,
        "score": score,
        "issues": issues,
        "suggestions": [f"fix: {i}" for i in issues],
        "summary": f"scored {score}",
    }


async def test_refine_loop_improves_then_accepts(tmp_path):
    prompts: list[str] = []
    critic_calls: list[int] = []
    generator = scripted_generator(
        [("tool", GOOD_V1), ("text", "v1 built"), ("tool", GOOD_V2), ("text", "v2 built")],
        prompts,
    )
    critic = scripted_critic(
        [critique_args(5, ["hole is too small"]), critique_args(9, [])], critic_calls
    )
    config = RunConfig(max_iterations=5, score_threshold=8, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "a cube with a hole",
        config,
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
    )

    assert result.accepted is True
    assert len(result.iterations) == 2
    assert result.best.index == 2
    assert result.best.critique.score == 9
    assert result.best.execution.code == GOOD_V2

    # feedback wiring: iteration 2's prompt must carry iteration 1's critique
    assert len(prompts) == 2
    assert "hole is too small" in prompts[1]
    assert GOOD_V1 in prompts[1]

    # persistence
    run_dir = result.run_dir
    assert (run_dir / "spec.txt").read_text() == "a cube with a hole"
    assert (run_dir / "iter_01" / "iteration.json").exists()
    assert (run_dir / "iter_02" / "views.png").exists()
    assert (run_dir / "final" / "model.stl").exists()
    assert (run_dir / "final" / "model.py").read_text() == GOOD_V2
    report = (run_dir / "report.md").read_text()
    assert "9" in report and "a cube with a hole" in report


async def test_refine_anchors_on_best_not_previous_regression(tmp_path):
    """When an iteration regresses below the best so far, the next iteration must
    refine the champion — not the regression — so the loop cannot diverge away
    from a good result (the failure mode the mug E2E exposed)."""
    code_c = "import cadquery as cq\nresult = cq.Workplane().box(10, 10, 10).faces('>Z').hole(4).edges('|Z').fillet(0.5)  # v3"
    prompts: list[str] = []
    critic_calls: list[int] = []
    generator = scripted_generator(
        [
            ("tool", GOOD_V1), ("text", "champion body"),
            ("tool", GOOD_V2), ("text", "regressed attempt"),
            ("tool", code_c), ("text", "final"),
        ],
        prompts,
    )
    # iter1=6 (champion), iter2=3 (regression below champion), iter3=9 (accept)
    critic = scripted_critic(
        [critique_args(6, ["needs a hole"]), critique_args(3, ["broke the body"]), critique_args(9, [])],
        critic_calls,
    )
    config = RunConfig(max_iterations=5, score_threshold=8, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "spec",
        config,
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
    )

    assert result.accepted is True
    assert result.best.index == 3
    # iteration 3 must be anchored on the champion (iter1, GOOD_V1), not the
    # regressed iteration 2 (GOOD_V2).
    assert GOOD_V1 in prompts[2]
    assert GOOD_V2 not in prompts[2]
    assert "needs a hole" in prompts[2], "champion's critique must carry forward"
    assert "regress" in prompts[2].lower(), "model should be told its last change regressed"


async def test_feedback_carries_checklist_and_regression_ledger(tmp_path):
    """The champion's per-requirement checklist is forwarded, and any requirement an
    EARLIER iteration satisfied but the champion no longer passes is surfaced as a
    do-not-regress ledger — the anti whack-a-mole guard."""
    code_c = "import cadquery as cq\nresult = cq.Workplane().box(10, 10, 10)  # v3"
    prompts: list[str] = []
    generator = scripted_generator(
        [
            ("tool", GOOD_V1), ("text", "v1"),
            ("tool", GOOD_V2), ("text", "v2"),
            ("tool", code_c), ("text", "v3"),
        ],
        prompts,
    )

    def cl(env_status, hole_status):
        return [
            {"requirement": "Overall envelope", "status": env_status, "severity": "critical"},
            {"requirement": "Central hole", "status": hole_status, "severity": "critical"},
        ]

    # iter1 (score 6): envelope PASS, hole FAIL. iter2 (score 7, new champion):
    # envelope FAIL, hole PASS — so envelope was satisfied earlier but the champion lost it.
    iter1 = {**critique_args(6, ["hole missing"]), "checklist": cl("pass", "fail")}
    iter2 = {**critique_args(7, ["envelope off"]), "checklist": cl("fail", "pass")}
    critic = scripted_critic([iter1, iter2, critique_args(9, [])], [])
    config = RunConfig(max_iterations=3, score_threshold=8, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "spec",
        config,
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
    )

    fb = prompts[2]  # feedback for iter3, built from champion iter2 + full history
    assert "REQUIREMENT STATUS" in fb
    assert "[FAIL] Overall envelope" in fb  # champion's current checklist
    assert "[PASS] Central hole" in fb
    # envelope passed in iter1 but champion (iter2) no longer passes it → ledger flags it
    assert "ALREADY-SATISFIED EARLIER" in fb
    assert result.accepted is True


async def test_budget_exhaustion_returns_best_iteration(tmp_path):
    prompts: list[str] = []
    critic_calls: list[int] = []
    generator = scripted_generator(
        [
            ("tool", GOOD_V1), ("text", "t1"),
            ("tool", GOOD_V2), ("text", "t2"),
            ("tool", GOOD_V1), ("text", "t3"),
        ],
        prompts,
    )
    critic = scripted_critic(
        [critique_args(5, ["a"]), critique_args(6, ["b"]), critique_args(4, ["c"])],
        critic_calls,
    )
    config = RunConfig(max_iterations=3, score_threshold=8, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "spec",
        config,
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
    )

    assert result.accepted is False
    assert len(result.iterations) == 3
    assert result.best.index == 2, "highest score wins even if not the last iteration"
    assert result.best.critique.score == 6
    assert (result.run_dir / "final" / "model.py").exists()


async def test_failed_iteration_scores_zero_and_loop_recovers(tmp_path):
    prompts: list[str] = []
    critic_calls: list[int] = []
    generator = scripted_generator(
        [
            ("tool", BAD), ("text", "could not build"),
            ("tool", GOOD_V2), ("text", "fixed"),
        ],
        prompts,
    )
    critic = scripted_critic([critique_args(9, [])], critic_calls)
    config = RunConfig(
        max_iterations=3,
        score_threshold=8,
        max_exec_attempts_per_iteration=1,
        out_dir=tmp_path / "runs",
    )

    result = await generate_cad(
        "spec",
        config,
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
    )

    assert result.accepted is True
    assert len(result.iterations) == 2
    first, second = result.iterations
    assert first.critique is None
    assert first.effective_score == 0
    assert first.execution is not None and first.execution.success is False
    assert second.critique.score == 9
    assert len(critic_calls) == 1, "critic must not run for iterations without geometry"
    # error feedback reaches the next generator prompt
    assert "NameError" in prompts[1]


async def test_iteration_callback_fires(tmp_path):
    seen: list[int] = []
    generator = scripted_generator([("tool", GOOD_V1), ("text", "t")], [])
    critic = scripted_critic([critique_args(10, [])], [])
    config = RunConfig(max_iterations=2, out_dir=tmp_path / "runs")

    await generate_cad(
        "spec",
        config,
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
        on_iteration=lambda rec: seen.append(rec.index),
    )

    assert seen == [1]


async def test_run_result_json_persisted(tmp_path):
    generator = scripted_generator([("tool", GOOD_V1), ("text", "t")], [])
    critic = scripted_critic([critique_args(10, [])], [])
    config = RunConfig(max_iterations=1, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "spec",
        config,
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
    )

    payload = json.loads((result.run_dir / "run_result.json").read_text())
    assert payload["accepted"] is True
    assert payload["best"]["critique"]["score"] == 10


async def test_drawings_persisted_and_threaded_with_auto_interpretation(tmp_path):
    first_contents: list = []
    images: list = []
    generator = capturing_generator([("tool", GOOD_V1), ("text", "built")], first_contents, images)
    critic = scripted_critic([critique_args(9, [])], [])
    drawings = [DrawingAttachment(filename="orig.png", media_type="image/png", data=PNG)]
    config = RunConfig(max_iterations=2, score_threshold=8, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "",
        config,
        drawings=drawings,
        interpreter_model=scripted_text_model("ENVELOPE 85x135x25, Ø15 THRU"),
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
    )

    # input drawing persisted under run_dir/input/ with a sanitized name
    assert (result.run_dir / "input" / "drawing_01.png").read_bytes() == PNG
    assert result.drawings == ["drawing_01.png"]
    # auto interpretation produced, persisted, and recorded
    assert (result.run_dir / "drawing_interpretation.md").read_text() == "ENVELOPE 85x135x25, Ø15 THRU"
    assert result.interpretation == "ENVELOPE 85x135x25, Ø15 THRU"
    # generator iter-1 prompt carried both the image and the interpretation text
    assert len(images) == 1 and images[0].media_type == "image/png"
    text_part = next(c for c in first_contents[0] if isinstance(c, str))
    assert "Ø15 THRU" in text_part


async def test_provided_interpretation_skips_interpreter(tmp_path):
    def boom(messages, info):  # interpreter must NOT be invoked
        raise AssertionError("interpreter should not run when interpretation is provided")

    generator = scripted_generator([("tool", GOOD_V1), ("text", "built")], [])
    critic = scripted_critic([critique_args(9, [])], [])
    drawings = [DrawingAttachment(filename="x.jpg", media_type="image/jpeg", data=PNG)]
    config = RunConfig(max_iterations=1, score_threshold=8, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "",
        config,
        drawings=drawings,
        interpretation="USER-EDITED DIMS",
        interpreter_model=FunctionModel(boom),
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
    )

    assert result.interpretation == "USER-EDITED DIMS"
    assert (result.run_dir / "drawing_interpretation.md").read_text() == "USER-EDITED DIMS"
    assert (result.run_dir / "input" / "drawing_01.jpg").read_bytes() == PNG


# --------------------------------------------------------------------------- #
# Corroborated acceptance gate + open-findings ledger
# --------------------------------------------------------------------------- #
def _crit(score, checklist=None):
    from cad_gen.models import Critique

    return Critique(
        matches_spec=score >= 8, score=score, issues=[], suggestions=[],
        summary="x", checklist=checklist or [],
    )


def _gate_record(score, checklist=None, refutation=None, check_report=None):
    from cad_gen.models import IterationRecord

    return IterationRecord(
        index=1, critique=_crit(score, checklist),
        refutation=refutation, check_report=check_report,
    )


def test_acceptance_gate_accepts_clean_part():
    from cad_gen.orchestrator import _acceptance_ok

    assert _acceptance_ok(_gate_record(9), RunConfig(score_threshold=8)) is True


def test_acceptance_gate_blocks_below_threshold():
    from cad_gen.orchestrator import _acceptance_ok

    assert _acceptance_ok(_gate_record(7), RunConfig(score_threshold=8)) is False


def test_acceptance_gate_blocks_major_uncertain_item():
    from cad_gen.models import ChecklistItem
    from cad_gen.orchestrator import _acceptance_ok

    rec = _gate_record(10, checklist=[ChecklistItem(requirement="lug Z", status="uncertain", severity="major")])
    assert _acceptance_ok(rec, RunConfig(score_threshold=8)) is False


def test_acceptance_gate_allows_minor_uncertain_redacted_mass():
    from cad_gen.models import ChecklistItem
    from cad_gen.orchestrator import _acceptance_ok

    rec = _gate_record(9, checklist=[ChecklistItem(requirement="mass", status="uncertain", severity="minor")])
    assert _acceptance_ok(rec, RunConfig(score_threshold=8)) is True


def test_acceptance_gate_blocks_major_refutation_but_not_minor():
    from cad_gen.models import Refutation
    from cad_gen.orchestrator import _acceptance_ok

    cfg = RunConfig(score_threshold=8)
    major = Refutation(found_discrepancy=True, discrepancies=["wrong slope"], most_severe="wrong slope", severity="major")
    minor = Refutation(found_discrepancy=True, discrepancies=["1mm nit"], most_severe="1mm nit", severity="minor")
    assert _acceptance_ok(_gate_record(10, refutation=major), cfg) is False
    assert _acceptance_ok(_gate_record(9, refutation=minor), cfg) is True


def test_acceptance_gate_blocks_failing_critical_check():
    from cad_gen.models import Check, CheckReport, CheckStatus
    from cad_gen.orchestrator import _acceptance_ok

    rep = CheckReport(checks=[Check(name="watertight", status=CheckStatus.FAIL, critical=True)])
    assert _acceptance_ok(_gate_record(10, check_report=rep), RunConfig(score_threshold=8)) is False


def test_open_findings_carries_unresolved_refuter_finding():
    from cad_gen.models import IterationRecord, Refutation
    from cad_gen.orchestrator import _open_findings

    iter1 = IterationRecord(
        index=1, critique=_crit(6),
        refutation=Refutation(found_discrepancy=True, discrepancies=["lug holes 5 mm too low"],
                              most_severe="lug holes 5 mm too low", severity="major"),
    )
    iter2 = IterationRecord(index=2, critique=_crit(6),
                            refutation=Refutation(found_discrepancy=False, severity="none"))
    out = _open_findings([iter1, iter2], champion=iter2)
    assert "UNRESOLVED FINDINGS" in out
    assert "lug holes 5 mm too low" in out


def test_open_findings_drops_requirement_the_champion_passes():
    from cad_gen.models import ChecklistItem, IterationRecord
    from cad_gen.orchestrator import _open_findings

    iter1 = IterationRecord(index=1, critique=_crit(6, [ChecklistItem(requirement="Central hole", status="fail", severity="major")]))
    champ = IterationRecord(index=2, critique=_crit(9, [ChecklistItem(requirement="Central hole", status="pass", severity="major")]))
    assert "Central hole" not in _open_findings([iter1, champ], champion=champ)


async def test_text_only_prompt_stays_plain_string(tmp_path):
    first_contents: list = []
    images: list = []
    generator = capturing_generator([("tool", GOOD_V1), ("text", "built")], first_contents, images)
    critic = scripted_critic([critique_args(9, [])], [])
    config = RunConfig(max_iterations=1, score_threshold=8, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "a plain cube",
        config,
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
    )

    # text-only path is byte-identical to before: plain str, no images, no input dir
    assert isinstance(first_contents[0], str)
    assert first_contents[0] == "a plain cube"
    assert images == []
    assert result.drawings == []
    assert result.interpretation is None
    assert not (result.run_dir / "input").exists()
    assert not (result.run_dir / "drawing_interpretation.md").exists()
