from pathlib import Path

import pytest
from pydantic import ValidationError

from cad_gen.models import (
    Critique,
    ExecutionResult,
    GeometryMetrics,
    IterationRecord,
    ReprojectionReport,
    ReprojectionView,
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

    def test_reprojection_never_changes_effective_score(self):
        # The reprojection signal is advisory: it must not influence the score.
        rec = IterationRecord(
            index=1,
            execution=make_execution(),
            critique=Critique(
                matches_spec=False, score=6, issues=[], suggestions=[], summary="s"
            ),
            reprojection=ReprojectionReport(evaluated=True, passed=False, digest="d"),
        )
        assert rec.effective_score == 6

    def test_round_trips_with_reprojection(self):
        rec = IterationRecord(
            index=2,
            execution=make_execution(),
            reprojection=ReprojectionReport(
                evaluated=True,
                views_found=2,
                digest="front view: 90% reproduced",
                views={
                    "front": ReprojectionView(
                        coverage=0.9, chamfer_pct=1.0, aspect_ok=True,
                        aspect_rel_err=0.0, aspect_signed=0.0,
                        overlay_path=Path("/tmp/overlay_front.png"),
                    )
                },
                composite_path=Path("/tmp/overlay_composite.png"),
            ),
        )
        assert IterationRecord.model_validate_json(rec.model_dump_json()) == rec


class TestRunConfig:
    @pytest.fixture(autouse=True)
    def _clear_model_env(self, monkeypatch):
        # Defaults resolve from the environment, so isolate these tests from any
        # CAD_GEN_MODEL / CAD_GEN_CRITIC_MODEL the developer happens to have set.
        monkeypatch.delenv("CAD_GEN_MODEL", raising=False)
        monkeypatch.delenv("CAD_GEN_CRITIC_MODEL", raising=False)
        monkeypatch.delenv("CAD_GEN_VIEW_MODEL", raising=False)
        monkeypatch.delenv("CAD_GEN_REASONING_EFFORT", raising=False)
        monkeypatch.delenv("CAD_GEN_CRITIC_REASONING_EFFORT", raising=False)

    def test_defaults(self):
        cfg = RunConfig()
        assert cfg.model == "google:gemini-3.5-flash"
        assert cfg.critic_model == "google:gemini-3.5-flash"
        assert cfg.view_model == "google:gemini-3.5-flash"
        assert cfg.max_iterations == 5
        assert cfg.score_threshold == 8
        assert cfg.exec_timeout_s == 60
        assert cfg.max_exec_attempts_per_iteration == 4
        assert cfg.out_dir == Path("runs")
        assert cfg.reproject is True
        assert cfg.reproject_timeout_s == 120
        assert cfg.reproject_low_coverage == 0.80
        assert cfg.reproject_orientation_coverage == 0.55
        assert cfg.reasoning_effort is None
        assert cfg.critic_reasoning_effort is None

    def test_reasoning_effort_resolves_from_env(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_REASONING_EFFORT", "high")
        assert RunConfig().reasoning_effort == "high"

    def test_critic_reasoning_effort_resolves_from_env(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_CRITIC_REASONING_EFFORT", "  Medium ")
        cfg = RunConfig()
        assert cfg.critic_reasoning_effort == "medium"  # normalized; independent of generator
        assert cfg.reasoning_effort is None

    def test_invalid_critic_reasoning_effort_rejected(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_CRITIC_REASONING_EFFORT", "turbo")
        with pytest.raises(ValidationError):
            RunConfig()

    def test_reasoning_effort_normalizes_case_and_whitespace(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_REASONING_EFFORT", "  HIGH ")
        assert RunConfig().reasoning_effort == "high"

    def test_blank_reasoning_effort_is_unset(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_REASONING_EFFORT", "   ")
        assert RunConfig().reasoning_effort is None

    def test_invalid_reasoning_effort_rejected(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_REASONING_EFFORT", "turbo")
        with pytest.raises(ValidationError):
            RunConfig()

    def test_critic_model_defaults_independently_of_generator(self):
        # The critic has its own default (vision model) — it does NOT inherit --model.
        cfg = RunConfig(model="anthropic:claude-x")
        assert cfg.critic_model == "google:gemini-3.5-flash"

    def test_critic_model_override_wins(self):
        cfg = RunConfig(model="anthropic:claude-x", critic_model="google:gemini-3.5-pro")
        assert cfg.critic_model == "google:gemini-3.5-pro"

    def test_models_resolve_from_env(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_MODEL", "anthropic:claude-opus-4-8")
        monkeypatch.setenv("CAD_GEN_CRITIC_MODEL", "anthropic:claude-opus-4-8")
        cfg = RunConfig()
        assert cfg.model == "anthropic:claude-opus-4-8"
        assert cfg.critic_model == "anthropic:claude-opus-4-8"

    def test_legacy_gpt_models_are_forced_to_gemini(self, monkeypatch):
        provider = "op" + "enai"
        model_name = "g" + "pt-5.5"
        monkeypatch.setenv("CAD_GEN_MODEL", f"{provider}-responses:{model_name}")
        monkeypatch.setenv("CAD_GEN_CRITIC_MODEL", f"{provider}:{model_name}")
        monkeypatch.setenv("CAD_GEN_VIEW_MODEL", model_name)

        cfg = RunConfig()

        assert cfg.model == "google:gemini-3.5-flash"
        assert cfg.critic_model == "google:gemini-3.5-flash"
        assert cfg.view_model == "google:gemini-3.5-flash"

    def test_view_model_resolves_from_env(self, monkeypatch):
        monkeypatch.setenv("CAD_GEN_VIEW_MODEL", "anthropic:claude-opus-4-8")
        assert RunConfig().view_model == "anthropic:claude-opus-4-8"

    def test_explicit_values_override_env(self, monkeypatch):
        # A flag / request-body value beats the env var (precedence: explicit > env > default).
        monkeypatch.setenv("CAD_GEN_MODEL", "anthropic:claude-opus-4-8")
        monkeypatch.setenv("CAD_GEN_CRITIC_MODEL", "anthropic:claude-opus-4-8")
        cfg = RunConfig(model="google:gemini-3.5-flash", critic_model="google:gemini-3.5-pro")
        assert cfg.model == "google:gemini-3.5-flash"
        assert cfg.critic_model == "google:gemini-3.5-pro"

    def test_explicit_legacy_gpt_values_are_forced_to_gemini(self):
        provider = "op" + "enai"
        model_name = "g" + "pt-5.5"
        cfg = RunConfig(
            model=f"{provider}:{model_name}",
            critic_model=f"{provider}-chat:{model_name}",
        )
        assert cfg.model == "google:gemini-3.5-flash"
        assert cfg.critic_model == "google:gemini-3.5-flash"


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
