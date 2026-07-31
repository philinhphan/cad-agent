"""Tests for the editing no-op guard.

Motivation: a CADGenBench audit of submission v3 found 10 of 32 editing samples were
byte-equivalent to their input. The benchmark renormalizes an editing sample's shape score
against that input, so a no-op scores 0 on shape and caps at 0.4 — the vision critic was
not catching them from renders, because a small or internal edit looks identical.
"""

import pytest

from cad_gen.orchestrator import _force_noop_critique
from cad_gen.models import Critique
from cad_gen.step_metrics import StepMeasurement, describe_edit_delta, is_noop_edit


def m(volume: float, bbox=(10.0, 20.0, 30.0)) -> StepMeasurement:
    return StepMeasurement(volume_mm3=volume, bbox_mm=bbox)


class TestIsNoopEdit:
    def test_identical_is_noop(self):
        assert is_noop_edit(m(1000.0), m(1000.0)) is True

    def test_export_roundtrip_jitter_is_still_a_noop(self):
        """Re-exporting the same solid perturbs volume by ~1e-6 relative; that is a no-op."""
        assert is_noop_edit(m(2544478.2), m(2544478.15)) is True

    def test_small_real_edit_is_not_a_noop(self):
        """A genuine edit moves volume far more than the tolerance.

        Guards the tolerance from being loosened into swallowing real edits: this is a
        0.14% change, three orders of magnitude above NOOP_VOLUME_REL_TOL.
        """
        assert is_noop_edit(m(438355.7), m(438968.5)) is False

    def test_bbox_change_alone_is_not_a_noop(self):
        """Volume can be preserved while geometry moves — e.g. a feature shifted."""
        base = m(1000.0, (10.0, 20.0, 30.0))
        moved = m(1000.0, (10.0, 20.0, 46.6))
        assert is_noop_edit(base, moved) is False

    @pytest.mark.parametrize(
        ("base_v", "cand_v"),
        [(2544478.2, 2544538.0), (848188.8, 848204.4), (439512.0, 439512.0)],
    )
    def test_real_v3_noops_are_detected(self, base_v, cand_v):
        """Volumes taken from actual v3 samples 201, 206 and 225, which shipped as no-ops."""
        assert is_noop_edit(m(base_v), m(cand_v)) is True

    @pytest.mark.parametrize(
        ("base_v", "cand_v"),
        [(4878.1, 2426.0), (77895.8, 113675.6), (75590.0, 42532.9)],
    )
    def test_real_v3_edits_are_not_flagged(self, base_v, cand_v):
        """Volumes from v3 samples 204, 209 and 224, which did change. No false positives."""
        assert is_noop_edit(m(base_v), m(cand_v)) is False

    def test_zero_volume_base_does_not_divide_by_zero(self):
        assert is_noop_edit(m(0.0), m(0.0)) is True

    def test_phantom_uniform_bbox_delta_does_not_mask_a_noop(self):
        """Regression: the tolerance-gap bug that made the guard miss 6 of 10 real no-ops.

        Bnd_Box inflated by each shape's stored OCCT tolerance produces an IDENTICAL
        phantom delta on all three axes (0.046 mm was measured between an authored
        input.step and its re-export). measure_step now uses AddOptimal_s so no such
        delta reaches this function — but if a future change reintroduces one, a real
        no-op must not be reclassified as an edit on the strength of it.
        """
        base = m(2544478.153, (434.841959, 274.087640, 91.716342))
        # Same solid, volume essentially unchanged, boxes differing only by the phantom.
        reexport = m(2544538.019, (434.841959, 274.087640, 91.716342))
        assert is_noop_edit(base, reexport) is True


class TestDescribeEditDelta:
    def test_noop_verdict_is_explicit(self):
        text = describe_edit_delta(m(1000.0), m(1000.0))
        assert "NO-OP" in text
        assert "scores 0" in text

    def test_changed_verdict_asks_for_judgement(self):
        text = describe_edit_delta(m(1000.0), m(1500.0))
        assert "NO-OP" not in text
        assert "did change" in text
        assert "+50.000%" in text


class TestForceNoopCritique:
    def test_overrides_a_passing_critique(self):
        """The measured signal must beat the critic — this is the v3 failure mode."""
        passing = Critique(
            matches_spec=True, score=9, issues=[], suggestions=[], summary="Looks right."
        )
        forced = _force_noop_critique(passing)

        assert forced.score == 0
        assert forced.matches_spec is False
        assert "NO-OP" in forced.issues[0]
        assert "Looks right." in forced.summary  # original verdict preserved for context

    def test_keeps_existing_issues(self):
        c = Critique(
            matches_spec=False, score=4, issues=["hole too small"], suggestions=["x"],
            summary="s",
        )
        forced = _force_noop_critique(c)

        assert forced.issues[1] == "hole too small"
        assert forced.suggestions == ["x"]

    def test_handles_a_missing_critique(self):
        forced = _force_noop_critique(None)

        assert forced.score == 0
        assert forced.matches_spec is False
