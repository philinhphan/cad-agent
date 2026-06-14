"""Self-refine loop tests with fully scripted models — no LLM, no CadQuery."""

import json
from pathlib import Path

from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from cad_gen.models import (
    Critique,
    DrawingAttachment,
    ExecutionResult,
    GeometryMetrics,
    IterationRecord,
    ReprojectionReport,
    RunConfig,
)
from cad_gen.orchestrator import (
    _build_feedback,
    _match_key,
    _update_ledger,
    generate_cad,
)

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


COMPOSITE_PNG = b"\x89PNG\r\n\x1a\ncomposite"
VIEW_BOXES = {"front": [0.1, 0.6, 0.3, 0.3], "top": [0.1, 0.1, 0.3, 0.3]}


def inert_reprojector(
    step_path, drawing_path, out_dir, config, regions=None, timeout_s=120
) -> ReprojectionReport:
    """Default test stub: never spawns the real subprocess; withholds the signal."""
    return ReprojectionReport(evaluated=False, skipped_reason="stub")


def stub_reprojector(*, digest="REPROJECT DIGEST", composite=COMPOSITE_PNG, seen_regions=None):
    """Evaluated stub: writes a fake composite into out_dir and returns `digest`.

    Records each call's `regions` into `seen_regions` when provided."""

    def _run(
        step_path, drawing_path, out_dir, config, regions=None, timeout_s=120
    ) -> ReprojectionReport:
        if seen_regions is not None:
            seen_regions.append(regions)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        composite_path = None
        if composite is not None:
            composite_path = out_dir / "overlay_composite.png"
            composite_path.write_bytes(composite)
        return ReprojectionReport(evaluated=True, digest=digest, composite_path=composite_path)

    return _run


def scripted_view_locator(boxes: dict, calls: list | None = None) -> FunctionModel:
    """A FunctionModel that returns `boxes` as a ViewLayout structured-output tool call."""

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if calls is not None:
            calls.append(1)
        views = [
            {"label": k, "x": v[0], "y": v[1], "w": v[2], "h": v[3]} for k, v in boxes.items()
        ]
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {"views": views})])

    return FunctionModel(fn)


def _prompt_text(content) -> str:
    """Text part of a prompt: the str itself, or the first str of a [text, *images] list."""
    if isinstance(content, str):
        return content
    return next(c for c in content if isinstance(c, str))


def _images_of(content) -> list:
    """The BinaryContent images in a prompt (empty for a plain-string prompt)."""
    if not isinstance(content, list):
        return []
    return [c for c in content if isinstance(c, BinaryContent)]


def capturing_critic(
    critiques: list[dict], prompts_seen: list[str], images: list | None = None
) -> FunctionModel:
    """Like scripted_critic but records the critic's user-prompt text (and images) per call."""
    state = {"i": 0}

    def fn(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        content = messages[0].parts[0].content
        prompts_seen.append(_prompt_text(content))
        if images is not None and isinstance(content, list):
            images.extend(c for c in content if isinstance(c, BinaryContent))
        args = critiques[state["i"]]
        state["i"] += 1
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return FunctionModel(fn)


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
        view_locator_model=scripted_view_locator(VIEW_BOXES),
        reprojector=inert_reprojector,
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
        view_locator_model=scripted_view_locator(VIEW_BOXES),
        reprojector=inert_reprojector,
    )

    assert result.interpretation == "USER-EDITED DIMS"
    assert (result.run_dir / "drawing_interpretation.md").read_text() == "USER-EDITED DIMS"
    assert (result.run_dir / "input" / "drawing_01.jpg").read_bytes() == PNG


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


async def test_reproject_skipped_without_drawings(tmp_path):
    """Text-only runs must never invoke the reprojector (it needs a drawing)."""
    def boom_reprojector(*args, **kwargs):
        raise AssertionError("reprojector must not run for a text-only spec")

    generator = scripted_generator(
        [("tool", GOOD_V1), ("text", "v1"), ("tool", GOOD_V2), ("text", "v2")], []
    )
    critic = scripted_critic([critique_args(5, ["x"]), critique_args(9, [])], [])
    config = RunConfig(max_iterations=2, score_threshold=8, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "a plain cube",
        config,
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
        reprojector=boom_reprojector,
    )

    assert all(rec.reprojection is None for rec in result.iterations)


async def test_reproject_digest_reaches_critic_and_generator(tmp_path):
    """Evaluated reprojection: digest grounds the critic, and the BEST version's overlay
    composite + digest flow into the next generator iteration."""
    first_contents: list = []
    images: list = []
    critic_prompts: list[str] = []
    generator = capturing_generator(
        [("tool", GOOD_V1), ("text", "v1"), ("tool", GOOD_V2), ("text", "v2")],
        first_contents,
        images,
    )
    critic = capturing_critic(
        [critique_args(5, ["hole too small"]), critique_args(9, [])], critic_prompts
    )
    drawings = [DrawingAttachment(filename="d.png", media_type="image/png", data=PNG)]
    config = RunConfig(max_iterations=2, score_threshold=8, out_dir=tmp_path / "runs")

    await generate_cad(
        "spec",
        config,
        drawings=drawings,
        interpretation="DIMS",  # skip the interpreter
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
        view_locator_model=scripted_view_locator(VIEW_BOXES),
        reprojector=stub_reprojector(),
    )

    # the digest grounds the critic (it ran before the critique)
    assert any("REPROJECT DIGEST" in p for p in critic_prompts)
    # iteration 1's prompt had only the drawing (no champion overlay yet)
    assert len(_images_of(first_contents[0])) == 1
    # iteration 2's prompt leads with the overlay composite, then the original drawing,
    # and carries the digest text in its feedback
    imgs2 = _images_of(first_contents[1])
    assert len(imgs2) == 2 and imgs2[0].data == COMPOSITE_PNG
    assert "REPROJECT DIGEST" in _prompt_text(first_contents[1])


async def test_reproject_not_evaluated_is_withheld(tmp_path):
    """When the reprojection is not evaluated, nothing leaks to critic or generator and
    the score path is unchanged (advisory)."""
    first_contents: list = []
    images: list = []
    critic_prompts: list[str] = []
    generator = capturing_generator(
        [("tool", GOOD_V1), ("text", "v1"), ("tool", GOOD_V2), ("text", "v2")],
        first_contents,
        images,
    )
    critic = capturing_critic(
        [critique_args(5, ["x"]), critique_args(9, [])], critic_prompts
    )
    drawings = [DrawingAttachment(filename="d.png", media_type="image/png", data=PNG)]
    config = RunConfig(max_iterations=2, score_threshold=8, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "spec",
        config,
        drawings=drawings,
        interpretation="DIMS",
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
        view_locator_model=scripted_view_locator(VIEW_BOXES),
        reprojector=inert_reprojector,  # evaluated=False
    )

    assert not any("REPROJECT" in p for p in critic_prompts)
    assert len(_images_of(first_contents[1])) == 1  # only the drawing, no composite
    assert result.best.critique.score == 9  # champion still chosen by critique score


async def test_reprojection_persisted_in_iteration_json(tmp_path):
    generator = scripted_generator([("tool", GOOD_V1), ("text", "v1")], [])
    critic = scripted_critic([critique_args(9, [])], [])
    drawings = [DrawingAttachment(filename="d.png", media_type="image/png", data=PNG)]
    config = RunConfig(max_iterations=1, score_threshold=8, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "spec",
        config,
        drawings=drawings,
        interpretation="DIMS",
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
        view_locator_model=scripted_view_locator(VIEW_BOXES),
        reprojector=stub_reprojector(),
    )

    raw = (result.run_dir / "iter_01" / "iteration.json").read_text()
    record = IterationRecord.model_validate_json(raw)
    assert record.reprojection is not None
    assert record.reprojection.evaluated is True
    assert "REPROJECT DIGEST" in record.reprojection.digest


async def test_view_locator_runs_once_and_threads_regions(tmp_path):
    """The VLM view-locator runs ONCE per run (the layout is drawing-level) and its boxes
    reach every iteration's reprojection call."""
    seen_regions: list = []
    locator_calls: list = []
    generator = scripted_generator(
        [("tool", GOOD_V1), ("text", "v1"), ("tool", GOOD_V2), ("text", "v2")], []
    )
    critic = scripted_critic([critique_args(5, ["x"]), critique_args(9, [])], [])
    drawings = [DrawingAttachment(filename="d.png", media_type="image/png", data=PNG)]
    config = RunConfig(max_iterations=2, score_threshold=8, out_dir=tmp_path / "runs")

    result = await generate_cad(
        "spec",
        config,
        drawings=drawings,
        interpretation="DIMS",
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
        view_locator_model=scripted_view_locator(VIEW_BOXES, locator_calls),
        reprojector=stub_reprojector(seen_regions=seen_regions),
    )

    assert len(locator_calls) == 1  # computed once, not per iteration
    assert len(seen_regions) == 2  # but threaded into every iteration
    assert seen_regions[0] == {"front": [0.1, 0.6, 0.3, 0.3], "top": [0.1, 0.1, 0.3, 0.3]}
    assert (result.run_dir / "view_layout.json").exists()


async def test_reproject_overlay_and_verdict_reach_critic(tmp_path):
    """The reprojection verdict (text) and overlay composite (image) reach the critic so it
    can act on a failed geometric check (the runs/sucess3 false-accept fix)."""
    critic_prompts: list[str] = []
    critic_images: list = []
    generator = scripted_generator([("tool", GOOD_V1), ("text", "v1")], [])
    critic = capturing_critic([critique_args(9, [])], critic_prompts, critic_images)
    drawings = [DrawingAttachment(filename="d.png", media_type="image/png", data=PNG)]
    config = RunConfig(max_iterations=1, score_threshold=8, out_dir=tmp_path / "runs")

    await generate_cad(
        "spec",
        config,
        drawings=drawings,
        interpretation="DIMS",
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
        view_locator_model=scripted_view_locator(VIEW_BOXES),
        reprojector=stub_reprojector(),  # evaluated=True, passed=False (default)
    )

    # the failed-check verdict reaches the critic as text, and the overlay as an image
    assert any("DID NOT PASS" in p for p in critic_prompts)
    assert any(img.data == COMPOSITE_PNG for img in critic_images)


async def test_view_locator_not_called_without_drawings(tmp_path):
    def boom(messages, info):
        raise AssertionError("view locator must not run for a text-only spec")

    generator = scripted_generator([("tool", GOOD_V1), ("text", "v1")], [])
    critic = scripted_critic([critique_args(9, [])], [])
    config = RunConfig(max_iterations=1, out_dir=tmp_path / "runs")

    await generate_cad(
        "a plain cube",
        config,
        generator_model=generator,
        critic_model=critic,
        executor=stub_executor,
        renderer=stub_renderer,
        view_locator_model=FunctionModel(boom),
    )  # reaching here without boom firing = pass


# --- Issue ledger + feedback (A) ---------------------------------------------

def _champ(index: int, code: str, score: int, issues: list[str]) -> IterationRecord:
    """A successful champion record carrying a critique with `issues`."""
    rec = IterationRecord(
        index=index,
        execution=ExecutionResult(success=True, code=code, duration_s=0.1),
    )
    rec.critique = Critique(
        matches_spec=False, score=score, issues=issues,
        suggestions=[f"fix: {i}" for i in issues], summary="x",
    )
    return rec


def test_match_key_ignores_numbers():
    # the same defect with a different wrong value must still match
    a = _match_key("height is 12mm but the spec says 8mm")
    b = _match_key("height is 10mm but the spec says 8mm")
    assert a == b


def test_update_ledger_tracks_persistence_and_resolution():
    flange = "The vertical upright flange is completely missing."
    c1 = _champ(1, "v1", 4, [flange, "height is 12mm but spec says 8mm"])
    ledger = _update_ledger([], c1)
    assert {t.streak for t in ledger} == {1}

    # same champion next round: both issues persist -> streak grows, value change still matches
    c2 = _champ(2, "v1", 4, [flange, "height is 10mm but spec says 8mm"])
    ledger = _update_ledger(ledger, c2)
    assert all(t.streak == 2 for t in ledger)
    assert all(t.first_seen == 1 for t in ledger)

    # flange resolved (disappears), a brand-new issue appears -> fresh streak
    c3 = _champ(3, "v3", 6, ["the corner fillet R20 is missing"])
    ledger = _update_ledger(ledger, c3)
    assert len(ledger) == 1
    assert ledger[0].streak == 1 and ledger[0].first_seen == 3


def test_build_feedback_escalates_persistent_issues():
    flange = "vertical upright flange is missing"
    c1 = _champ(1, "v1", 4, [flange])
    ledger = _update_ledger([], c1)
    ledger = _update_ledger(ledger, _champ(2, "v1", 4, [flange]))  # survived twice

    fb = _build_feedback(c1, latest=c1, ledger=ledger)
    assert "STILL UNFIXED after 2 attempts" in fb
    assert "TOP PRIORITY" in fb
    assert "v1" in fb  # champion code carried for incremental editing


def test_build_feedback_embeds_regression_diff():
    champ = _champ(1, "import cadquery as cq\nresult = box(10)  # good", 6, ["needs hole"])
    regressed = _champ(2, "import cadquery as cq\nresult = box(9)  # broke it", 2, ["worse"])
    ledger = _update_ledger([], champ)

    fb = _build_feedback(champ, latest=regressed, ledger=ledger)
    assert "WORSE than the best" in fb
    assert "```diff" in fb
    assert "# broke it" in fb  # the regressing line is shown to the model
    assert "regression" in fb.lower()
