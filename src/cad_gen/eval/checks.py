"""Deterministic checks of executed geometry against the drawing target.

These are computed by the CAD kernel (volume, bbox, solid count, watertightness)
and compared to the typed ``DrawingTarget``. They are ADVISORY — they never override
the critic's score for acceptance. Their job is to (a) feed the critic authoritative
numbers, (b) drive the generator's numeric gradient, and (c) surface a "checks
disagree" warning to the user.

Every target-dependent check SKIPs when its data is absent, so the engine works for
general drawings (and plain text specs) where no mass/density/envelope is known. The
two topology checks (single solid, watertight) always run — they need no target.
"""

from cad_gen.models import (
    Check,
    CheckReport,
    CheckStatus,
    DrawingTarget,
    GeometryMetrics,
)


def compute_mass_g(volume_mm3: float, density_kg_m3: float) -> float:
    """Mass in grams from volume (mm³) and density (kg/m³).

    1 mm³ = 1e-9 m³ → ×density gives kg → ×1000 gives g, i.e. ×1e-6 overall.
    """
    return volume_mm3 * density_kg_m3 * 1e-6


def run_checks(
    metrics: GeometryMetrics | None, target: DrawingTarget | None
) -> CheckReport:
    """Compute every applicable deterministic check; absent targets → SKIP."""
    if metrics is None:
        return CheckReport(checks=[])
    checks = [_check_single_solid(metrics), _check_watertight(metrics)]
    for builder in (_check_mass, _check_envelope, _check_holes):
        check = builder(metrics, target)
        if check is not None:
            checks.append(check)
    return CheckReport(checks=checks)


def _check_single_solid(metrics: GeometryMetrics) -> Check:
    ok = metrics.n_solids == 1
    return Check(
        name="single_solid",
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        critical=True,
        target="1",
        observed=str(metrics.n_solids),
        message=(
            "exactly one fused solid"
            if ok
            else f"{metrics.n_solids} disjoint solids — features did not fuse"
        ),
    )


def _check_watertight(metrics: GeometryMetrics) -> Check:
    wt = metrics.is_watertight
    if wt is None:
        return Check(
            name="watertight",
            status=CheckStatus.SKIP,
            critical=False,
            message="watertightness unknown (mesh check unavailable)",
        )
    return Check(
        name="watertight",
        status=CheckStatus.PASS if wt else CheckStatus.FAIL,
        critical=True,
        observed="yes" if wt else "no",
        message="closed solid" if wt else "NOT watertight — non-manifold/leaky geometry",
    )


def _check_mass(metrics: GeometryMetrics, target: DrawingTarget | None) -> Check | None:
    if target is None or target.density_kg_m3 is None or target.target_mass_g is None:
        return None
    observed = (
        metrics.mass_g
        if metrics.mass_g is not None
        else compute_mass_g(metrics.volume_mm3, target.density_kg_m3)
    )
    tol = (
        target.mass_tol_g
        if target.mass_tol_g is not None
        else max(0.5, 0.01 * target.target_mass_g)
    )
    delta = observed - target.target_mass_g
    ok = abs(delta) <= tol
    unreliable = metrics.is_watertight is False
    detail = (
        ""
        if ok
        else f" → {abs(delta):.2f} g too {'heavy' if delta > 0 else 'light'}"
    )
    suffix = " (UNRELIABLE: geometry is not watertight)" if unreliable else ""
    return Check(
        name="mass",
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        critical=True,
        target=round(target.target_mass_g, 3),
        observed=round(observed, 3),
        delta=round(delta, 3),
        tolerance=round(tol, 3),
        message=f"{observed:.2f} g vs {target.target_mass_g:.2f} ± {tol:.2f} g{detail}{suffix}",
    )


def _check_envelope(
    metrics: GeometryMetrics, target: DrawingTarget | None
) -> Check | None:
    if target is None or target.envelope_mm is None:
        return None
    obs_sorted = sorted(metrics.bbox_mm)
    tgt_sorted = sorted(target.envelope_mm)
    worst = max(abs(o - t) for o, t in zip(obs_sorted, tgt_sorted))
    tol = max(0.5, 0.005 * max(tgt_sorted))
    ok = worst <= tol

    def fmt(values) -> str:
        return " × ".join(f"{v:.1f}" for v in sorted(values, reverse=True))

    detail = "" if ok else f" → off by {worst:.2f} mm"
    return Check(
        name="envelope",
        status=CheckStatus.PASS if ok else CheckStatus.FAIL,
        critical=True,
        target=fmt(target.envelope_mm),
        observed=fmt(metrics.bbox_mm),
        delta=round(worst, 3),
        tolerance=round(tol, 3),
        message=f"bbox {fmt(metrics.bbox_mm)} vs envelope {fmt(target.envelope_mm)} mm{detail}",
    )


def _check_holes(metrics: GeometryMetrics, target: DrawingTarget | None) -> Check | None:
    """Confirm every required hole *diameter* exists as a cylindrical face on the solid.

    Conservative by construction: a real through/blind hole of diameter D always produces a
    cylindrical B-rep face of radius D/2, so a correct part can never FAIL this. It FAILs only
    when a required primary diameter is entirely ABSENT (a missing hole, or a radius-vs-
    diameter mix-up). We check the primary diameter only — counterbore/round-slot radii can
    coincide with fillet radii, which would risk a false PASS, not a false FAIL. SKIPs when no
    holes are targeted or the kernel produced no cylinder data.
    """
    if target is None or not target.holes:
        return None
    radii = [c.radius_mm for c in (metrics.cylinders or [])]
    if not radii:
        return Check(
            name="holes",
            status=CheckStatus.SKIP,
            critical=False,
            message="no cylindrical-face data (hole geometry not measured)",
        )

    def matches(diameter_mm: float) -> int:
        r = diameter_mm / 2.0
        tol = max(0.25, 0.01 * r)
        return sum(1 for rr in radii if abs(rr - r) <= tol)

    def label(h) -> str:
        return h.note or f"{h.count}× Ø{h.diameter_mm:g}"

    missing = [label(h) for h in target.holes if matches(h.diameter_mm) == 0]
    tgt = "; ".join(label(h) for h in target.holes)
    if missing:
        return Check(
            name="holes",
            status=CheckStatus.FAIL,
            critical=True,
            target=tgt,
            observed=f"cylinder radii {sorted({round(r, 2) for r in radii})}",
            message="missing required hole diameter(s): " + "; ".join(missing),
        )
    return Check(
        name="holes",
        status=CheckStatus.PASS,
        critical=True,
        target=tgt,
        observed=f"{len(radii)} cylindrical faces",
        message="every required hole diameter is present as a cylindrical face",
    )
