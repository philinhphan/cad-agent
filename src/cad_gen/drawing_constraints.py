"""Structured drawing constraints derived from the human/VLM interpretation text.

This is deliberately conservative: it extracts only unambiguous numeric callouts and
marks anything we cannot prove from kernel metrics as ``unverified`` rather than pretending
to validate it. The resulting JSON gives the critic and generator a stable checklist.
"""

from __future__ import annotations

import re

from cad_gen.models import (
    ConstraintCheck,
    ConstraintValidation,
    DrawingAngleConstraint,
    DrawingConstraints,
    DrawingCounterboreConstraint,
    DrawingHoleConstraint,
    DrawingScalarConstraint,
    GeometryMetrics,
)

_NUM = r"\d+(?:\.\d+)?"
_ENVELOPE_RE = re.compile(
    rf"(?P<a>{_NUM})\s*(?:x|X|×)\s*(?P<b>{_NUM})\s*(?:x|X|×)\s*(?P<c>{_NUM})"
)
_HOLE_RE = re.compile(
    rf"(?:(?P<count>\d+)\s*(?:x|X|×)\s*)?Ø\s*(?P<diam>{_NUM})(?P<trail>[^.\n]*)",
    re.IGNORECASE,
)
_COUNTERBORE_RE = re.compile(
    rf"(?:⌴|counterbore|c['’]?bore)\s*Ø?\s*(?P<diam>{_NUM})(?P<trail>[^.\n]*)",
    re.IGNORECASE,
)
_DEPTH_RE = re.compile(rf"(?:depth|↧|deep)\s*(?P<depth>{_NUM})", re.IGNORECASE)
_RADIUS_RE = re.compile(rf"\bR\s*(?P<radius>{_NUM})(?:\s*,?\s*(?P<count>\d+)\s*places)?", re.IGNORECASE)
_ANGLE_RE = re.compile(rf"(?P<angle>{_NUM})\s*°")


def derive_drawing_constraints(text: str) -> DrawingConstraints:
    """Extract a stable constraint JSON from the drawing interpretation markdown."""
    constraints = DrawingConstraints(source_text=text)
    envelope = _ENVELOPE_RE.search(text)
    if envelope:
        constraints.envelope_mm = tuple(float(envelope.group(k)) for k in ("a", "b", "c"))

    constraints.holes = [
        DrawingHoleConstraint(
            diameter_mm=float(m.group("diam")),
            count=int(m.group("count") or 1),
            through=("thru" in m.group("trail").lower() or "through" in m.group("trail").lower()),
            source=_line_for(text, m.start()),
        )
        for m in _HOLE_RE.finditer(text)
    ]
    constraints.counterbores = [
        DrawingCounterboreConstraint(
            diameter_mm=float(m.group("diam")),
            depth_mm=_depth_from(m.group("trail")),
            source=_line_for(text, m.start()),
        )
        for m in _COUNTERBORE_RE.finditer(text)
    ]
    constraints.radii = [
        DrawingScalarConstraint(
            value_mm=float(m.group("radius")),
            count=int(m.group("count") or 1),
            source=_line_for(text, m.start()),
        )
        for m in _RADIUS_RE.finditer(text)
    ]
    constraints.angles = [
        DrawingAngleConstraint(degrees=float(m.group("angle")), source=_line_for(text, m.start()))
        for m in _ANGLE_RE.finditer(text)
    ]
    return constraints


def validate_drawing_constraints(
    metrics: GeometryMetrics | None, constraints: DrawingConstraints | None
) -> ConstraintValidation | None:
    """Validate constraints that can be checked deterministically from current metrics."""
    if metrics is None or constraints is None:
        return None

    checks: list[ConstraintCheck] = []
    if constraints.envelope_mm is not None:
        expected = constraints.envelope_mm
        actual = metrics.bbox_mm
        diffs = [abs(a - e) for a, e in zip(actual, expected)]
        tolerances = [max(0.2, e * 0.01) for e in expected]
        ok = all(d <= t for d, t in zip(diffs, tolerances))
        checks.append(
            ConstraintCheck(
                name="overall envelope",
                status="pass" if ok else "fail",
                expected=f"{_fmt_dims(expected)} mm",
                actual=f"{_fmt_dims(actual)} mm",
                message=(
                    f"overall envelope {'matches' if ok else 'does not match'}: "
                    f"expected {_fmt_dims(expected)} mm, actual {_fmt_dims(actual)} mm"
                ),
            )
        )

    for hole in constraints.holes:
        checks.append(
            ConstraintCheck(
                name=f"hole Ø{hole.diameter_mm:g}",
                status="unverified",
                expected=f"{hole.count}x Ø{hole.diameter_mm:g}"
                + (" THRU" if hole.through else ""),
                message=(
                    "hole diameter/count extracted from drawing; kernel metric validation "
                    "needs topology-aware feature checks"
                ),
            )
        )
    for radius in constraints.radii:
        checks.append(
            ConstraintCheck(
                name=f"radius R{radius.value_mm:g}",
                status="unverified",
                expected=f"R{radius.value_mm:g}",
                message="radius extracted from drawing; exact edge-radius validation is advisory",
            )
        )
    for angle in constraints.angles:
        checks.append(
            ConstraintCheck(
                name=f"angle {angle.degrees:g}deg",
                status="unverified",
                expected=f"{angle.degrees:g} degrees",
                message="angle extracted from drawing; exact face-angle validation is advisory",
            )
        )

    failed = [c for c in checks if c.status == "fail"]
    digest_lines = ["Constraint validation:"]
    digest_lines.extend(f"- {c.status.upper()}: {c.message}" for c in checks)
    return ConstraintValidation(
        passed=not failed,
        checks=checks,
        digest="\n".join(digest_lines) if checks else "Constraint validation: no constraints extracted.",
    )


def _depth_from(text: str) -> float | None:
    match = _DEPTH_RE.search(text)
    return float(match.group("depth")) if match else None


def _line_for(text: str, index: int) -> str:
    start = text.rfind("\n", 0, index) + 1
    end = text.find("\n", index)
    if end == -1:
        end = len(text)
    return text[start:end].strip()


def _fmt_dims(values: tuple[float, float, float]) -> str:
    return " x ".join(f"{v:g}" for v in values)
