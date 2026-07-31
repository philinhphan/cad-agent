"""What an editing candidate actually changed, measured by cutting it against its base.

`step_metrics.EditDelta` answers "did anything change at all?" from bulk volume and bounding
box. That is enough to catch a no-op and nothing else. The v3 audit shows why it is not
enough:

- Sample 224 ("remove the groove") shipped an edit that removed **33 058 mm3** in a region
  spanning the part's entire 80 mm length. Bulk volume said, correctly, "the geometry did
  change".
- Sample 205 ("remove these two holes") shipped a candidate whose bounding box runs to
  x = 200.7 mm where the base ends at x = 84.8 mm — a 116 mm spike of material outside the
  part — while adding only 611 mm3, far too little for the bulk check to flag.

Both are shape-axis zeros. CADGenBench renormalizes an editing sample's shape score against
the unmodified input, so the scoring headroom is whatever a *correct* edit is worth, and any
collateral change spends it. Catching "changed, but not the way it was asked to" needs the
actual difference set, which means booleans:

    removed = Cut(base, candidate)     material the edit took away
    added   = Cut(candidate, base)     material the edit put there

exploded into connected lumps, each with its own volume, bounding box and centroid.

Cost: 4-18 s per direction on real benchmark parts, so this runs ONCE on the champion after
the loop, not per iteration. Subprocess-isolated with a timeout, like `reproject/adapter.py`
and `step_validity.py`, because OCCT booleans are a native call that can hang.

Self-contained below the adapter section — stdlib + OCP only, no `cad_gen` imports — so this
file doubles as the subprocess script.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

_STDERR_TAIL = 2000
_MAX_LUMPS = 24  # per direction; a fragmented diff is already a verdict, the tail adds nothing

# A boolean between two solids that share coincident faces leaves razor-thin artifacts along
# those faces — geometry that exists in the difference set but not in the part. A lump is
# discarded only when it is BOTH negligible in volume AND degenerate in shape, so a small but
# real feature (a chamfer, a shallow pocket) is never mistaken for an artifact.
_SLIVER_VOLUME_MM3 = 1e-3
_SLIVER_VOLUME_FRACTION = 1e-7
_SLIVER_MIN_EXTENT_MM = 1e-3

# Slack when asking "is this added material outside the base's envelope?", so bounding-box
# round-off never reads as a protrusion.
_OUTSIDE_BBOX_TOL_MM = 1e-3


def edit_diff_report(
    base_step: Path,
    candidate_step: Path,
    *,
    timeout_s: float = 300.0,
    lumps_stl: Path | None = None,
):
    """Measure what `candidate_step` changed relative to `base_step`.

    Returns a `cad_gen.models.EditDiff`. Never raises: a timeout or an OCCT failure comes
    back with ``evaluated=False`` and `skipped_reason` set, and callers treat that as "no
    verdict" rather than as a rejection — the same withholding contract the reprojection
    check uses.

    `lumps_stl`, when given, receives a mesh of just the changed material — every lump the
    edit added or removed, and nothing else. Rendered on its own it answers "what did this
    edit actually touch?" at a glance, which is the question a table of volumes answers only
    slowly. Best-effort: a failure to write it never affects the measurement.
    """
    from cad_gen.models import EditDiff, EditRegion

    base_step, candidate_step = Path(base_step), Path(candidate_step)
    for path in (base_step, candidate_step):
        if not path.exists():
            return EditDiff(evaluated=False, skipped_reason=f"{path} does not exist")

    # Report via a FILE, not stdout: OCCT writes multi-line banners to stdout from inside
    # native calls, and anything sharing that channel gets interleaved with them.
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "diff.json"
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--base",
            str(base_step.resolve()),
            "--candidate",
            str(candidate_step.resolve()),
            "--report",
            str(report_path),
        ]
        if lumps_stl is not None:
            argv += ["--lumps-stl", str(Path(lumps_stl).resolve())]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return EditDiff(
                evaluated=False, skipped_reason=f"edit diff timed out after {timeout_s:g}s"
            )
        except OSError as exc:
            return EditDiff(evaluated=False, skipped_reason=f"could not run the edit diff: {exc}")
        if proc.returncode != 0:
            reason = proc.stderr[-_STDERR_TAIL:].strip() or f"exit code {proc.returncode}"
            return EditDiff(evaluated=False, skipped_reason=reason)
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return EditDiff(evaluated=False, skipped_reason=f"unreadable diff report: {exc}")
    if payload.get("error"):
        return EditDiff(evaluated=False, skipped_reason=payload["error"])

    diff = EditDiff(
        evaluated=True,
        removed=[EditRegion(**region) for region in payload["removed"]],
        added=[EditRegion(**region) for region in payload["added"]],
        removed_volume_mm3=payload["removed_volume_mm3"],
        added_volume_mm3=payload["added_volume_mm3"],
        base_volume_mm3=payload["base_volume_mm3"],
        changed_fraction=payload["changed_fraction"],
        locality=payload["locality"],
        outside_base_bbox_mm3=payload["outside_base_bbox_mm3"],
        overshoot_mm=payload["overshoot_mm"],
        overshoot_fraction=payload["overshoot_fraction"],
    )
    diff.plausible, diff.verdict_reason = is_plausible_local_edit(diff)
    diff.digest = describe_edit_diff(diff)
    return diff


def is_plausible_local_edit(diff) -> tuple[bool, str]:
    """Does this measured difference look like the local edit that was asked for?

    Returns ``(verdict, reason)``; the reason is shown to a human in the run summary, so it
    should say what the numbers were, not just yes or no.

    TODO(you): implement the judgement. Everything above has already measured the geometry;
    this is the policy that turns those numbers into "ship it" or "fall back". Signals
    available on `diff`:

      diff.removed_volume_mm3 / diff.added_volume_mm3  material taken away / put there
      diff.base_volume_mm3                             the part being edited
      diff.changed_fraction    (removed + added) / base volume
      diff.locality            diagonal(changed bbox) / diagonal(base bbox), 0..~1
      len(diff.removed) / len(diff.added)              connected lumps per direction
      diff.outside_base_bbox_mm3   added material lying outside the base's bounding box
      diff.overshoot_mm / diff.overshoot_fraction      how far past the envelope it reaches

    Measured on the v3 submission — these are real numbers from this pipeline, not estimates:

      sample  changed_fraction  locality  outside_mm3  overshoot        asked for   verdict
      201          200.314%       1.000          0.0   —                4 pockets    REJECT
      209           45.933%       1.000      35779.7   (whole part)     one feature  REJECT
      224           43.733%       0.863          0.0   —                one groove   REJECT
      205            0.140%       0.860        611.5   116 mm / 66.6%   two holes    REJECT
      245            2.907%       0.539       3520.0     5 mm / 13.4%   +5 mm wall   ACCEPT

    Sample 201 is the clearest signature in the set: it removed 2 548 445 mm3 and added
    2 548 505 mm3 — almost the entire part, in both directions at once. That is what a
    from-scratch rebuild looks like from here, and it scores 0 on the renormalized shape
    axis no matter how good the rebuilt part is. Note that 201 and 209 both pass the existing
    volume/bbox no-op guard (`tests/test_step_metrics.py` lists them as known real edits) —
    they changed plenty, just not in the way they were asked to.

    Two traps are visible in that table:

    - `locality` alone cannot work. 224 (a catastrophic over-cut) and 205 (a spike) sit at
      0.86 while legitimate edits spread across a part just as far — sample 201 moves four
      symmetric pocket walls and is supposed to span the whole part.
    - "grew the envelope" alone cannot work either. Both 205 and 245 have material outside
      the base bbox; 245 is correct, because raising a wall by 5 mm is exactly what it was
      asked to do. Only the magnitude separates them, which is what `overshoot_fraction`
      is for.

    `changed_fraction` is the one that isolates 224: removing 43.7% of a part's volume is not
    a groove, whatever the span.

    Until this is implemented it accepts everything, so the gate is inert rather than wrong:
    an unimplemented policy must not start rejecting candidates on its own.
    """
    return True, "not yet judged (is_plausible_local_edit is unimplemented)"


def describe_edit_diff(diff) -> str:
    """Factual summary of the measured difference, for the run report and for a human.

    Deliberately verdict-last: the numbers come first so a reader can disagree with the
    policy without having to re-run the booleans.
    """
    lines = [
        f"base volume:     {diff.base_volume_mm3:.1f} mm3",
        f"material removed: {diff.removed_volume_mm3:.1f} mm3 in {len(diff.removed)} region(s)",
        f"material added:   {diff.added_volume_mm3:.1f} mm3 in {len(diff.added)} region(s)",
        f"changed fraction: {diff.changed_fraction * 100:.3f}% of the base volume",
        f"locality:         {diff.locality:.3f} (changed bbox diagonal / part bbox diagonal)",
    ]
    if diff.outside_base_bbox_mm3 > 0:
        lines.append(
            f"outside the base envelope: {diff.outside_base_bbox_mm3:.1f} mm3 of added "
            f"material, reaching {diff.overshoot_mm:.2f} mm "
            f"({diff.overshoot_fraction * 100:.1f}% of that axis) past the base bounding box"
        )
    for label, regions in (("removed", diff.removed), ("added", diff.added)):
        for region in regions[:4]:
            box = " x ".join(f"{v:.1f}" for v in region.bbox_mm)
            center = ", ".join(f"{v:.1f}" for v in region.center_mm)
            lines.append(f"  {label}: {region.volume_mm3:.1f} mm3  bbox {box} mm  at ({center})")
    verdict = "plausible local edit" if diff.plausible else "REJECTED"
    lines.append(f"VERDICT: {verdict} — {diff.verdict_reason}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------
# Subprocess side — stdlib + OCP only below this line.
# ---------------------------------------------------------------------------------------


def _read_shape(path):
    from OCP.IFSelect import IFSelect_ReturnStatus
    from OCP.STEPControl import STEPControl_Reader

    reader = STEPControl_Reader()
    # TransferRoots() after a failed parse segfaults. Here that only kills the subprocess
    # (the caller degrades to "no verdict"), but a clean error beats an opaque exit code.
    if reader.ReadFile(str(path)) != IFSelect_ReturnStatus.IFSelect_RetDone:
        raise ValueError(f"{path} could not be parsed as STEP")
    reader.TransferRoots()
    return reader.OneShape()


def _volume(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())


def _bounds(shape):
    """Axis-aligned bounds as ``(xmin, ymin, zmin, xmax, ymax, zmax)``.

    ``AddOptimal_s``, not ``Add_s``: the cheap variant inflates the box by each shape's
    stored OCCT tolerance, which differs between two files of the same solid written by
    different kernels. `step_metrics.measure_step` documents the same trap at length.
    """
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    box = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, box)
    return box.Get()


def _lumps(shape, base_volume: float) -> list[dict]:
    """Explode a boolean result into connected solids, dropping coincident-face slivers."""
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    sliver_volume = max(_SLIVER_VOLUME_MM3, _SLIVER_VOLUME_FRACTION * abs(base_volume))
    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    out: list[dict] = []
    while explorer.More():
        solid = TopoDS.Solid_s(explorer.Current())
        explorer.Next()
        volume = _volume(solid)
        xmin, ymin, zmin, xmax, ymax, zmax = _bounds(solid)
        extents = (xmax - xmin, ymax - ymin, zmax - zmin)
        if volume < sliver_volume and min(extents) < _SLIVER_MIN_EXTENT_MM:
            continue
        out.append(
            {
                "volume_mm3": volume,
                "bbox_mm": extents,
                "center_mm": ((xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2),
                "_bounds": (xmin, ymin, zmin, xmax, ymax, zmax),
            }
        )
    out.sort(key=lambda lump: lump["volume_mm3"], reverse=True)
    return out[:_MAX_LUMPS]


def _write_lumps_stl(shapes, path) -> None:
    """Mesh the changed lumps into one STL. Best-effort; a failure is not a measurement error."""
    from OCP.BRep import BRep_Builder
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    from OCP.TopoDS import TopoDS_Compound

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for shape in shapes:
        builder.Add(compound, shape)
    BRepMesh_IncrementalMesh(compound, 0.1, False, 0.5, True)
    StlAPI_Writer().Write(compound, str(path))


def _cut(a, b):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut

    op = BRepAlgoAPI_Cut(a, b)
    op.Build()
    return op.Shape()


def _overshoot(lump_bounds, base_bounds) -> tuple[float, float]:
    """How far a lump reaches beyond the base envelope: ``(mm, fraction of that extent)``.

    Both numbers are needed, because "the candidate grew the part" is not by itself a fault:
    v3 sample 245 was asked to raise a wall by 5 mm and correctly grew Z by 5 mm on a 37 mm
    extent, while sample 205 was asked to remove two holes and grew X by 116 mm on a 174 mm
    extent. Only the magnitude separates them.
    """
    lo, hi = lump_bounds[:3], lump_bounds[3:]
    base_lo, base_hi = base_bounds[:3], base_bounds[3:]
    worst_mm = 0.0
    worst_fraction = 0.0
    for axis in range(3):
        over = max(
            base_lo[axis] - lo[axis] - _OUTSIDE_BBOX_TOL_MM,
            hi[axis] - base_hi[axis] - _OUTSIDE_BBOX_TOL_MM,
        )
        if over <= 0:
            continue
        extent = max(base_hi[axis] - base_lo[axis], 1e-9)
        worst_mm = max(worst_mm, over)
        worst_fraction = max(worst_fraction, over / extent)
    return worst_mm, worst_fraction


def _diagonal(bounds) -> float:
    return sum((bounds[i + 3] - bounds[i]) ** 2 for i in range(3)) ** 0.5


def _changed_bounds(lumps):
    """Union of every changed lump's bounds; None when nothing changed."""
    if not lumps:
        return None
    lo = [min(lump["_bounds"][i] for lump in lumps) for i in range(3)]
    hi = [max(lump["_bounds"][i + 3] for lump in lumps) for i in range(3)]
    return (*lo, *hi)


def _compute(base_path, candidate_path, lumps_stl=None) -> dict:
    base = _read_shape(base_path)
    candidate = _read_shape(candidate_path)
    if base.IsNull() or candidate.IsNull():
        return {"error": "base or candidate STEP loaded as a null shape"}

    base_volume = _volume(base)
    base_bounds = _bounds(base)

    removed_shape, added_shape = _cut(base, candidate), _cut(candidate, base)
    removed = _lumps(removed_shape, base_volume)
    added = _lumps(added_shape, base_volume)
    if lumps_stl is not None:
        try:
            _write_lumps_stl([removed_shape, added_shape], lumps_stl)
        except Exception:  # noqa: BLE001 — a debug artifact must never fail the measurement
            pass

    removed_volume = sum(lump["volume_mm3"] for lump in removed)
    added_volume = sum(lump["volume_mm3"] for lump in added)
    outside_volume = 0.0
    overshoot_mm = 0.0
    overshoot_fraction = 0.0
    for lump in added:
        lump_mm, lump_fraction = _overshoot(lump["_bounds"], base_bounds)
        if lump_mm > 0:
            outside_volume += lump["volume_mm3"]
            overshoot_mm = max(overshoot_mm, lump_mm)
            overshoot_fraction = max(overshoot_fraction, lump_fraction)

    changed_bounds = _changed_bounds(removed + added)
    base_diagonal = _diagonal(base_bounds) or 1.0
    locality = _diagonal(changed_bounds) / base_diagonal if changed_bounds else 0.0

    for lump in removed + added:
        del lump["_bounds"]
    return {
        "removed": removed,
        "added": added,
        "removed_volume_mm3": removed_volume,
        "added_volume_mm3": added_volume,
        "base_volume_mm3": base_volume,
        "changed_fraction": (removed_volume + added_volume) / max(abs(base_volume), 1e-9),
        "locality": locality,
        "outside_base_bbox_mm3": outside_volume,
        "overshoot_mm": overshoot_mm,
        "overshoot_fraction": overshoot_fraction,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Boolean before/after diff of two STEP solids")
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", help="where to write the JSON result (default: stdout)")
    parser.add_argument("--lumps-stl", help="where to mesh just the changed material")
    args = parser.parse_args()
    payload = _compute(args.base, args.candidate, args.lumps_stl)
    if args.report:
        Path(args.report).write_text(json.dumps(payload))
    else:
        json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
