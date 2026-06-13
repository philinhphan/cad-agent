from pathlib import Path

from typer.testing import CliRunner

from cad_gen.cli import app
from cad_gen.models import (
    Critique,
    ExecutionResult,
    GeometryMetrics,
    IterationRecord,
    RunResult,
)

runner = CliRunner()


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
    ):
        assert option in result.output


def test_missing_api_key_exits_with_code_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

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
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

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
            "--model", "openai:gpt-5-mini",
            "--critic-model", "openai:gpt-5.2",
            "--timeout", "30",
            "--out", str(tmp_path / "elsewhere"),
        ],
    )

    assert result.exit_code == 0
    cfg = seen["config"]
    assert seen["spec"] == "a bracket"
    assert cfg.max_iterations == 7
    assert cfg.score_threshold == 9
    assert cfg.model == "openai:gpt-5-mini"
    assert cfg.critic_model == "openai:gpt-5.2"
    assert cfg.exec_timeout_s == 30
    assert cfg.out_dir == tmp_path / "elsewhere"
