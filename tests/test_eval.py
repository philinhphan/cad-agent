"""Deterministic eval-engine tests: mass + checks. No LLM, no CadQuery."""

import math

from cad_gen.eval.checks import compute_mass_g, run_checks
from cad_gen.models import (
    CheckStatus,
    CylinderFace,
    DrawingTarget,
    GeometryMetrics,
    HoleTarget,
)


def _metrics(**overrides) -> GeometryMetrics:
    defaults = dict(
        volume_mm3=1000.0,
        bbox_mm=(10.0, 10.0, 10.0),
        center_of_mass=(0.0, 0.0, 0.0),
        n_solids=1,
        n_faces=6,
        is_watertight=True,
    )
    defaults.update(overrides)
    return GeometryMetrics(**defaults)


def test_compute_mass_g_matches_density_formula():
    # 1,000,000 mm³ = 1 L; ABS at 1020 kg/m³ → 1.02 kg = 1020 g
    assert compute_mass_g(1_000_000, 1020) == 1020.0
    assert math.isclose(compute_mass_g(194_746, 1020), 198.64092, rel_tol=1e-6)


def test_checks_skip_without_target():
    report = run_checks(_metrics(), None)
    by_name = {c.name: c for c in report.checks}
    assert by_name["single_solid"].status is CheckStatus.PASS
    assert by_name["watertight"].status is CheckStatus.PASS
    assert "mass" not in by_name and "envelope" not in by_name
    assert report.all_critical_pass


def test_single_solid_failure_is_critical():
    report = run_checks(_metrics(n_solids=2), None)
    solid = next(c for c in report.checks if c.name == "single_solid")
    assert solid.status is CheckStatus.FAIL and solid.critical
    assert not report.all_critical_pass


def test_mass_check_within_and_outside_tolerance():
    target = DrawingTarget(density_kg_m3=1020, target_mass_g=1020.0, mass_tol_g=1.0)
    ok = run_checks(_metrics(volume_mm3=1_000_000), target)
    mass = next(c for c in ok.checks if c.name == "mass")
    assert mass.status is CheckStatus.PASS and abs(mass.delta) < 1e-6

    heavy = run_checks(_metrics(volume_mm3=1_010_000), target)  # 1030.2 g
    mass2 = next(c for c in heavy.checks if c.name == "mass")
    assert mass2.status is CheckStatus.FAIL and mass2.delta > 0
    assert not heavy.all_critical_pass


def test_mass_uses_default_tolerance_when_unspecified():
    target = DrawingTarget(density_kg_m3=1000, target_mass_g=100.0)  # tol = max(0.5, 1) = 1
    report = run_checks(_metrics(volume_mm3=100_500), target)  # 100.5 g, within 1 g
    mass = next(c for c in report.checks if c.name == "mass")
    assert mass.status is CheckStatus.PASS
    assert mass.tolerance == 1.0


def test_envelope_check_works_for_general_drawing_without_mass():
    target = DrawingTarget(envelope_mm=(135.0, 85.0, 65.0))  # no density/mass
    ok = run_checks(_metrics(bbox_mm=(135.0, 85.0, 65.0)), target)
    env = next(c for c in ok.checks if c.name == "envelope")
    assert env.status is CheckStatus.PASS
    assert not any(c.name == "mass" for c in ok.checks)  # mass skipped (no density/target)

    off = run_checks(_metrics(bbox_mm=(135.0, 154.0, 65.0)), target)
    assert next(c for c in off.checks if c.name == "envelope").status is CheckStatus.FAIL


def test_envelope_is_order_independent():
    target = DrawingTarget(envelope_mm=(135.0, 85.0, 65.0))
    report = run_checks(_metrics(bbox_mm=(65.0, 135.0, 85.0)), target)  # permuted axes
    assert next(c for c in report.checks if c.name == "envelope").status is CheckStatus.PASS


def test_watertight_none_is_non_critical_skip():
    report = run_checks(_metrics(is_watertight=None), None)
    wt = next(c for c in report.checks if c.name == "watertight")
    assert wt.status is CheckStatus.SKIP and not wt.critical
    assert report.all_critical_pass


def test_mass_uses_precomputed_metric_when_present():
    target = DrawingTarget(density_kg_m3=1020, target_mass_g=200.0, mass_tol_g=1.0)
    report = run_checks(_metrics(volume_mm3=1.0, mass_g=200.4), target)  # trust mass_g, not volume
    mass = next(c for c in report.checks if c.name == "mass")
    assert mass.status is CheckStatus.PASS and mass.observed == 200.4


# --------------------------------------------------------------------------- #
# Deterministic hole check (cylindrical-face presence)
# --------------------------------------------------------------------------- #
def test_hole_check_passes_when_every_diameter_present():
    target = DrawingTarget(
        holes=[HoleTarget(diameter_mm=15.0), HoleTarget(diameter_mm=5.0, count=2)]
    )
    metrics = _metrics(
        cylinders=[
            CylinderFace(radius_mm=7.5),   # Ø15 through
            CylinderFace(radius_mm=15.0),  # Ø30 counterbore (ignored — primary Ø only)
            CylinderFace(radius_mm=2.5),   # Ø5 ×2
            CylinderFace(radius_mm=2.5),
            CylinderFace(radius_mm=20.0),  # an R20 fillet — must not matter
        ]
    )
    holes = next(c for c in run_checks(metrics, target).checks if c.name == "holes")
    assert holes.status is CheckStatus.PASS


def test_hole_check_fails_when_a_primary_diameter_is_absent():
    target = DrawingTarget(holes=[HoleTarget(diameter_mm=15.0)])
    metrics = _metrics(cylinders=[CylinderFace(radius_mm=2.5)])  # only a Ø5, no Ø15
    report = run_checks(metrics, target)
    holes = next(c for c in report.checks if c.name == "holes")
    assert holes.status is CheckStatus.FAIL and holes.critical
    assert not report.all_critical_pass


def test_hole_check_skips_without_cylinder_data():
    target = DrawingTarget(holes=[HoleTarget(diameter_mm=15.0)])
    holes = next(c for c in run_checks(_metrics(), target).checks if c.name == "holes")
    assert holes.status is CheckStatus.SKIP and not holes.critical


def test_hole_check_absent_without_holes_target():
    report = run_checks(_metrics(cylinders=[CylinderFace(radius_mm=7.5)]), DrawingTarget())
    assert not any(c.name == "holes" for c in report.checks)
