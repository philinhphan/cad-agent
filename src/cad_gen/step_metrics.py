"""Library-agnostic measurement of a STEP file, via raw OpenCASCADE.

Used by the editing loop to compare a generated model against the base model it was
supposed to modify. It reads STEP through OCP directly — not through CadQuery or
build123d — so the same numbers come back whichever library produced the candidate,
and the comparison never depends on the optional build123d extra being installed.

(``reproject/check.py`` reads STEP the same way for the same reason.)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StepMeasurement:
    """Bulk properties of a STEP solid."""

    volume_mm3: float
    bbox_mm: tuple[float, float, float]


def measure_step(path: str | Path) -> StepMeasurement:
    """Measure `path`'s volume and axis-aligned bounding box.

    OCP is imported lazily: it is a heavyweight import and only the editing path needs it.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    # Checking ReadStatus is load-bearing: TransferRoots() after a failed parse SEGFAULTS,
    # and this runs in-process in the orchestrator, so no `except` could contain it.
    if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError(f"{path} could not be parsed as STEP")
    reader.TransferRoots()
    shape = reader.OneShape()

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)

    box = Bnd_Box()
    # AddOptimal_s, NOT Add_s. The cheap Add_s inflates the box by each shape's stored
    # OCCT tolerance, and two STEP files of the SAME solid written by different kernels
    # carry different tolerances — which shows up as an identical phantom delta on all
    # three axes (measured: 0.046 mm between an authored input.step and its CadQuery
    # re-export). That phantom is larger than a real edit's bbox change, so using Add_s
    # here silently misses genuine no-ops.
    BRepBndLib.AddOptimal_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()

    return StepMeasurement(
        volume_mm3=float(props.Mass()),
        bbox_mm=(float(xmax - xmin), float(ymax - ymin), float(zmax - zmin)),
    )


# An edit is treated as a no-op when BOTH the volume and every bounding-box extent are
# within these tolerances of the base model. They are deliberately tight: the point is to
# catch "the model returned input.step untouched", not to judge whether an edit was big
# enough. A genuine but small edit (a fillet, a shifted wall) moves volume by far more than
# 0.01%, while re-exporting the same solid through a different kernel perturbs it by ~1e-6.
NOOP_VOLUME_REL_TOL = 1e-4
NOOP_BBOX_ABS_TOL_MM = 1e-3


def is_noop_edit(base: StepMeasurement, candidate: StepMeasurement) -> bool:
    """True when `candidate` is indistinguishable from the base model it should have edited.

    CADGenBench renormalizes the shape axis of an editing sample against the unmodified
    input, so a no-op scores 0 on shape and caps at 0.4 overall — it is never worth
    submitting, no matter how valid the geometry is.
    """
    denom = max(abs(base.volume_mm3), 1e-9)
    if abs(candidate.volume_mm3 - base.volume_mm3) / denom > NOOP_VOLUME_REL_TOL:
        return False
    return all(
        abs(c - b) <= NOOP_BBOX_ABS_TOL_MM
        for b, c in zip(base.bbox_mm, candidate.bbox_mm, strict=True)
    )


def describe_edit_delta(base: StepMeasurement, candidate: StepMeasurement) -> str:
    """One-block factual summary of how far the candidate moved from the base model.

    Fed to the editing critic as measured ground truth, in the same spirit as the
    reprojection digest: the critic cannot reliably see a small or internal edit in a
    shaded render, so it is told the numbers instead of being asked to eyeball them.
    """
    dv = candidate.volume_mm3 - base.volume_mm3
    denom = max(abs(base.volume_mm3), 1e-9)
    pct = dv / denom * 100.0
    dbox = [c - b for b, c in zip(base.bbox_mm, candidate.bbox_mm, strict=True)]
    lines = [
        f"base volume:      {base.volume_mm3:.1f} mm3",
        f"candidate volume: {candidate.volume_mm3:.1f} mm3  "
        f"({dv:+.1f} mm3, {pct:+.3f}%)",
        f"base bbox:        {base.bbox_mm[0]:.3f} x {base.bbox_mm[1]:.3f} x {base.bbox_mm[2]:.3f} mm",
        f"candidate bbox:   {candidate.bbox_mm[0]:.3f} x {candidate.bbox_mm[1]:.3f} x "
        f"{candidate.bbox_mm[2]:.3f} mm  "
        f"(delta {dbox[0]:+.3f}, {dbox[1]:+.3f}, {dbox[2]:+.3f})",
    ]
    if is_noop_edit(base, candidate):
        lines.append(
            "VERDICT: NO-OP — the candidate is geometrically indistinguishable from the "
            "base model. The requested edit was NOT applied. This scores 0."
        )
    else:
        lines.append(
            "VERDICT: the geometry did change. Judge whether the change is the one the "
            "instruction asked for, in the right place and of the right magnitude."
        )
    return "\n".join(lines)
