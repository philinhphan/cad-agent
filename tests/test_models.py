from pathlib import Path

import pytest
from pydantic import ValidationError

from cad_gen.models import (
    Critique,
    ExecutionResult,
    GeometryMetrics,
    IterationRecord,
    RunConfig,
    RunResult,
)


def make_metrics(**overrides) -> GeometryMetrics:
    defaults = dict(
        volume_mm3=1000.0,
        bbox_mm=(10.0, 10.0, 10.0),
        center_of_mass=(0.0, 0.0, 0.0),
        n_solids=1,
        n_faces=6,
    )
    defaults.update(overrides)
    return GeometryMetrics(**defaults)


def make_execution(**overrides) -> ExecutionResult:
    defaults = dict(
        success=True,
        code="result = cq.Workplane().box(10, 10, 10)",
        metrics=make_metrics(),
        stl_path=Path("/tmp/model.stl"),
        step_path=Path("/tmp/model.step"),
        duration_s=2.5,
    )
    defaults.update(overrides)
    return ExecutionResult(**defaults)


class TestCritique:
    def test_score_bounds_accepted(self):
        for score in (0, 10):
            c = Critique(
                matches_spec=True, score=score, issues=[], suggestions=[], summary="ok"
            )
            assert c.score == score

    @pytest.mark.parametrize("score", [-1, 11])
    def test_score_out_of_bounds_rejected(self, score):
        with pytest.raises(ValidationError):
            Critique(
                matches_spec=True, score=score, issues=[], suggestions=[], summary="bad"
            )


class TestGeometryMetrics:
    def test_watertight_unknown_by_default(self):
        assert make_metrics().is_watertight is None

    def test_round_trips_through_json(self):
        m = make_metrics(is_watertight=True)
        assert GeometryMetrics.model_validate_json(m.model_dump_json()) == m


class TestExecutionResult:
    def test_failure_carries_traceback_and_no_artifacts(self):
        r = ExecutionResult(
            success=False,
            code="result = undefined_name",
            error="NameError: name 'undefined_name' is not defined",
            duration_s=1.0,
        )
        assert r.metrics is None
        assert r.stl_path is None
        assert "NameError" in r.error


class TestIterationRecord:
    def test_effective_score_is_critique_score(self):
        rec = IterationRecord(
            index=1,
            execution=make_execution(),
            critique=Critique(
                matches_spec=False, score=6, issues=["x"], suggestions=["y"], summary="s"
            ),
        )
        assert rec.effective_score == 6

    def test_effective_score_zero_without_critique(self):
        rec = IterationRecord(index=1, execution=make_execution(success=False, error="boom"))
        assert rec.effective_score == 0

    def test_round_trips_through_json(self):
        rec = IterationRecord(index=2, execution=make_execution(), render_path=Path("/tmp/v.png"))
        assert IterationRecord.model_validate_json(rec.model_dump_json()) == rec


class TestRunConfig:
    def test_defaults(self):
        cfg = RunConfig()
        assert cfg.model == "openai:gpt-5.2"
        assert cfg.critic_model == cfg.model
        assert cfg.max_iterations == 5
        assert cfg.score_threshold == 8
        assert cfg.exec_timeout_s == 60
        assert cfg.max_exec_attempts_per_iteration == 4
        assert cfg.out_dir == Path("runs")

    def test_critic_model_follows_explicit_model(self):
        cfg = RunConfig(model="anthropic:claude-x")
        assert cfg.critic_model == "anthropic:claude-x"

    def test_critic_model_override_wins(self):
        cfg = RunConfig(model="openai:gpt-5.2", critic_model="openai:gpt-5-mini")
        assert cfg.critic_model == "openai:gpt-5-mini"


class TestRunResult:
    def test_holds_best_and_history(self):
        it1 = IterationRecord(index=1, execution=make_execution())
        result = RunResult(
            accepted=True,
            spec="a 10mm cube",
            best=it1,
            iterations=[it1],
            run_dir=Path("/tmp/run"),
        )
        assert result.best.index == 1
        assert len(result.iterations) == 1
