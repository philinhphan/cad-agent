from pathlib import Path

from typer.testing import CliRunner

from cad_gen.cli import _load_drawing, app
from cad_gen.models import (
    Critique,
    ExecutionResult,
    GeometryMetrics,
    IterationRecord,
    RunResult,
)

runner = CliRunner()
FIXTURE_PNG = Path(__file__).parent / "fixtures" / "drawing.png"


def _fake_run_result(accepted: bool, run_dir: Path) -> RunResult:
    record = IterationRecord(
        index=1,
        execution=ExecutionResult(
            success=True,
            code="result = cq.Workplane().box(1, 1, 1)",
            metrics=GeometryMetrics(
                volume_mm3=1.0,
                bbox_mm=(1.0, 1.0, 1.0),
                center_of_mass=(0.0, 0.0, 0.0),
                n_solids=1,
                n_faces=6,
            ),
            duration_s=0.1,
        ),
        critique=Critique(
            matches_spec=accepted,
            score=9 if accepted else 5,
            issues=[] if accepted else ["wrong size"],
            suggestions=[],
            summary="verdict",
        ),
    )
    return RunResult(
        accepted=accepted, spec="a cube", best=record, iterations=[record], run_dir=run_dir
    )


def test_help_lists_all_options():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    for option in (
        "--max-iterations",
        "--threshold",
        "--model",
        "--critic-model",
        "--timeout",
        "--out",
        "--drawing",
        "--no-review",
        "--library",
    ):
        assert option in result.output


def test_missing_api_key_exits_with_code_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # default provider is openai

    result = runner.invoke(app, ["a 10mm cube"])

    assert result.exit_code == 2
    assert "OPENAI_API_KEY" in result.output


def test_accepted_run_exits_0_and_reports_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    captured_kwargs = {}

    async def fake_generate_cad(spec, config=None, **kwargs):
        captured_kwargs.update(kwargs)
        result = _fake_run_result(True, tmp_path / "run")
        kwargs["on_iteration"](result.iterations[0])
        return result

    monkeypatch.setattr("cad_gen.cli.generate_cad", fake_generate_cad)

    result = runner.invoke(app, ["a cube", "--out", str(tmp_path)])

    assert result.exit_code == 0
    assert "9/10" in result.output
    assert "run" in result.output  # points the user at the artifacts


def test_rejected_run_exits_1(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    async def fake_generate_cad(spec, config=None, **kwargs):
        return _fake_run_result(False, tmp_path / "run")

    monkeypatch.setattr("cad_gen.cli.generate_cad", fake_generate_cad)

    result = runner.invoke(app, ["a cube"])

    assert result.exit_code == 1


def test_config_flags_reach_run_config(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    seen = {}

    async def fake_generate_cad(spec, config=None, **kwargs):
        seen["spec"] = spec
        seen["config"] = config
        return _fake_run_result(True, tmp_path / "run")

    monkeypatch.setattr("cad_gen.cli.generate_cad", fake_generate_cad)

    result = runner.invoke(
        app,
        [
            "a bracket",
            "--max-iterations", "7",
            "--threshold", "9",
            "--model", "google:gemini-3.5-flash",
            "--critic-model", "google:gemini-3.5-pro",
            "--timeout", "30",
            "--out", str(tmp_path / "elsewhere"),
        ],
    )

    assert result.exit_code == 0
    cfg = seen["config"]
    assert seen["spec"] == "a bracket"
    assert cfg.max_iterations == 7
    assert cfg.score_threshold == 9
    assert cfg.model == "google:gemini-3.5-flash"
    assert cfg.critic_model == "google:gemini-3.5-pro"
    assert cfg.exec_timeout_s == 30
    assert cfg.out_dir == tmp_path / "elsewhere"
    assert cfg.library == "cadquery"  # untouched by the other flags


def test_library_flag_reaches_run_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    seen = {}

    async def fake_generate_cad(spec, config=None, **kwargs):
        seen["config"] = config
        return _fake_run_result(True, tmp_path / "run")

    monkeypatch.setattr("cad_gen.cli.generate_cad", fake_generate_cad)

    result = runner.invoke(app, ["a cube", "--library", "build123d"])

    assert result.exit_code == 0
    # The enum member must reach RunConfig as the plain string its Literal expects.
    assert seen["config"].library == "build123d"
    assert isinstance(seen["config"].library, str)


def test_unknown_library_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    result = runner.invoke(app, ["a cube", "--library", "openscad"])

    assert result.exit_code != 0
    assert "openscad" in result.output


def test_missing_cad_library_fails_fast(tmp_path, monkeypatch):
    """Without the preflight this surfaces as an ImportError in every sandbox attempt."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("cad_gen.cli.importlib.util.find_spec", lambda name: None)

    called = False

    async def fake_generate_cad(spec, config=None, **kwargs):
        nonlocal called
        called = True
        return _fake_run_result(True, tmp_path / "run")

    monkeypatch.setattr("cad_gen.cli.generate_cad", fake_generate_cad)

    result = runner.invoke(app, ["a cube", "--library", "build123d"])

    assert result.exit_code == 2
    assert "not installed" in result.output
    assert "uv sync --extra build123d" in result.output
    assert not called  # bailed before spending a single model call


def test_default_library_needs_no_extra(tmp_path, monkeypatch):
    """cadquery is a core dependency, so the preflight must never gate the default path."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("cad_gen.cli.importlib.util.find_spec", lambda name: None)

    async def fake_generate_cad(spec, config=None, **kwargs):
        return _fake_run_result(True, tmp_path / "run")

    monkeypatch.setattr("cad_gen.cli.generate_cad", fake_generate_cad)

    result = runner.invoke(app, ["a cube"])

    assert result.exit_code == 0


def test_no_spec_and_no_drawing_exits_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = runner.invoke(app, [])

    assert result.exit_code == 2
    assert "drawing" in result.output.lower()


def test_load_drawing_infers_jpeg_from_suffix(tmp_path):
    path = tmp_path / "part.jpg"
    path.write_bytes(b"\xff\xd8\xff\xe0 not really a jpeg but suffix wins")

    att = _load_drawing(path)

    assert att.media_type == "image/jpeg"
    assert att.filename == "part.jpg"


def test_drawing_run_threads_drawing_and_interpretation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    captured: dict = {}

    async def fake_interpret_drawing(agent, *, spec, drawings):
        return "EXTRACTED: 85 x 135 x 25, Ø15 THRU"

    async def fake_generate_cad(spec, config=None, **kwargs):
        captured.update(kwargs)
        captured["spec"] = spec
        result = _fake_run_result(True, tmp_path / "run")
        kwargs["on_iteration"](result.iterations[0])
        return result

    monkeypatch.setattr("cad_gen.cli.interpret_drawing", fake_interpret_drawing)
    monkeypatch.setattr("cad_gen.cli.generate_cad", fake_generate_cad)

    result = runner.invoke(
        app, ["--drawing", str(FIXTURE_PNG), "--no-review", "--out", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert captured["spec"] == ""  # drawing-only run
    drawings = captured["drawings"]
    assert len(drawings) == 1
    assert drawings[0].media_type == "image/png"
    assert drawings[0].filename == "drawing.png"
    assert captured["interpretation"] == "EXTRACTED: 85 x 135 x 25, Ø15 THRU"
