"""The benchmark's validity gate, run locally against our own candidates.

CADGenBench zeroes `cad_score` outright for any candidate that is not a valid solid, so a
candidate that fails this gate is worth exactly as much as no candidate at all. An audit of
the v3 submission found 7 of 81 candidates failing it — 5 of them editing samples — while the
harness's own signal (``SampleOutcome.valid_signal``: trimesh watertight + single solid) said
they were fine. Two of those five pass trimesh and still fail ``BRepCheck_Analyzer``.

So this module runs the grader's own checks, in the grader's own order, against the STEP file
as it will actually be submitted:

1. ``BRepCheck_Analyzer.IsValid()`` over the whole shape.
2. Every shell closed (``BRepCheck_Shell.Closed``), which handles periodic seams correctly.
3. The boundary tessellation is a closed orientable manifold.

Steps 1-2 are mirrored from ``cadgenbench.common.validity`` here, verbatim in semantics, so
the verdict matches without needing the benchmark installed. Step 3 is NOT reimplemented:
CADGenBench's mesh gate is ~750 lines of seam handling, vertex welding and deflection
laddering whose entire purpose is to avoid *false* non-manifold verdicts, and a from-scratch
approximation would reject good candidates — the one failure mode worse than shipping a bad
one. It is delegated to that package when importable and reported as unchecked when not
(see :func:`_mesh_errors`). Steps 1-2 alone catch 6 of the 7 known v3 failures.

Read through raw OCP, like `step_metrics.py` and `reproject/check.py`, so the verdict is the
same whichever CAD library produced the candidate and the optional build123d extra is not
required.

This file is deliberately self-contained — stdlib + OCP only, no `cad_gen` imports — so the
same file can be executed as a subprocess script by path (``python step_validity.py check
--step model.step``). OCCT is a native call that will not honour a Python signal mid-flight,
so in-process is not a safe place to run it; :func:`check_step_validity` spawns it and bounds
it with a timeout, mirroring ``reproject/adapter.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

_STDERR_TAIL = 2000
_MAX_ERRORS = 12  # a defective shape can report thousands; the first few identify the defect

# Every rung of the repair ladder must preserve the geometry it is repairing. A rung that
# changes volume is not repairing the candidate, it is replacing it — measured:
# ShapeUpgrade_UnifySameDomain moves sample 240 by ~4000 mm3.
_REPAIR_VOLUME_REL_TOL = 1e-6


@dataclass(frozen=True)
class StepValidity:
    """Verdict of the benchmark validity gate on one STEP file.

    Fields:
        is_valid: True iff every check that RAN passed. Note `mesh_checked`: when the mesh
            gate could not run, this is "nothing disqualifying found", not "the grader will
            accept it".
        is_watertight: every shell closed AND no BRepCheck errors.
        mesh_checked: whether the tessellation gate ran. False means CADGenBench was not
            importable, so a mesh-only defect (e.g. v3 sample 115) would go unnoticed here.
        errors: verbatim OCCT reasons, e.g. ``"Face: BRepCheck_UnorientableShape"``. These
            are fed back to the generator as-is — the status name is the diagnostic.
        unknown_reason: set when the check could not be performed at all (timeout, crash,
            unreadable STEP). `is_valid` is False but nothing was proven; callers must not
            treat this as a verdict.
    """

    is_valid: bool
    is_watertight: bool
    mesh_checked: bool
    errors: tuple[str, ...] = ()
    unknown_reason: str | None = None

    @property
    def evaluated(self) -> bool:
        """False when the gate could not run — no verdict was reached either way."""
        return self.unknown_reason is None


def check_step_validity(step_path: str | Path, *, timeout_s: float = 180.0) -> StepValidity:
    """Run the validity gate on `step_path` in a subprocess.

    Never raises: a timeout, crash or unreadable file comes back as ``evaluated=False`` with
    `unknown_reason` set, so a flaky OCCT call degrades the signal instead of the run.
    """
    return _run_subcommand("check", step_path, None, timeout_s=timeout_s)


def repair_step(
    step_path: str | Path, out_path: str | Path, *, timeout_s: float = 300.0
) -> StepValidity:
    """Try to make `step_path` pass the gate, writing the result to `out_path`.

    Returns the verdict on the repaired file; `out_path` is only written when some rung
    produced a shape that passes. On the v3 evidence this fixes 0 of 7 — see the module
    docstring of the ladder in :func:`_repair_shape` for why it is kept anyway.
    """
    return _run_subcommand("repair", step_path, out_path, timeout_s=timeout_s)


def _run_subcommand(
    command: str,
    step_path: str | Path,
    out_path: str | Path | None,
    *,
    timeout_s: float,
) -> StepValidity:
    step_path = Path(step_path)
    if not step_path.exists():
        return _unknown(f"{step_path} does not exist")
    # The report goes to a FILE, not stdout: OCCT's STEP writer prints a multi-line banner
    # on stdout during Write(), which lands in the middle of anything else written there.
    with tempfile.TemporaryDirectory() as tmp:
        report_path = Path(tmp) / "report.json"
        argv = [
            sys.executable,
            str(Path(__file__).resolve()),
            command,
            "--step",
            str(step_path.resolve()),
            "--report",
            str(report_path),
        ]
        if out_path is not None:
            argv += ["--out", str(Path(out_path).resolve())]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return _unknown(f"validity check timed out after {timeout_s:g}s")
        except OSError as exc:
            return _unknown(f"could not run the validity check: {exc}")
        if proc.returncode != 0:
            return _unknown(proc.stderr[-_STDERR_TAIL:].strip() or f"exit code {proc.returncode}")
        try:
            payload = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return _unknown(f"unreadable validity report: {exc}")
    return StepValidity(
        is_valid=bool(payload.get("is_valid", False)),
        is_watertight=bool(payload.get("is_watertight", False)),
        mesh_checked=bool(payload.get("mesh_checked", False)),
        errors=tuple(payload.get("errors", ())),
        unknown_reason=payload.get("unknown_reason"),
    )


def _unknown(reason: str) -> StepValidity:
    return StepValidity(
        is_valid=False, is_watertight=False, mesh_checked=False, unknown_reason=reason
    )


def describe_validity(validity: StepValidity) -> str:
    """One-block factual summary, written to be pasted straight into a critique issue.

    The OCCT status names are kept verbatim: ``BRepCheck_UnorientableShape`` is a searchable
    term that tells a model what kind of defect it built, where a paraphrase would not.
    """
    if not validity.evaluated:
        return f"VALIDITY: could not be checked ({validity.unknown_reason})."
    if validity.is_valid:
        suffix = "" if validity.mesh_checked else " (tessellation gate not run)"
        return f"VALIDITY: the exported solid passes the benchmark validity gate{suffix}."
    lines = [
        "VALIDITY: the exported solid FAILS the benchmark validity gate. CADGenBench scores "
        "an invalid solid 0 regardless of how correct the geometry looks. OCCT reports:"
    ]
    lines += [f"  - {err}" for err in validity.errors]
    if not validity.errors:
        lines.append("  - (no per-sub-shape detail available)")
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


def _write_shape(shape, path) -> None:
    from OCP.STEPControl import STEPControl_StepModelType, STEPControl_Writer

    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_StepModelType.STEPControl_AsIs)
    writer.Write(str(path))


def _volume(shape) -> float:
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props)
    return float(props.Mass())


def _is_watertight(shape) -> bool:
    """True iff the shape has at least one shell and every shell is closed.

    ``BRepCheck_Shell.Closed`` is the predicate OCC uses internally to validate solids, and
    unlike a naive edge-incidence count it handles periodic seams (a sphere's single-face
    shell) correctly. Mirrors ``cadgenbench.common.validity._is_watertight``.
    """
    from OCP.BRepCheck import BRepCheck_NoError, BRepCheck_Shell
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    explorer = TopExp_Explorer(shape, TopAbs_SHELL)
    any_shell = False
    while explorer.More():
        shell = TopoDS.Shell_s(explorer.Value())
        if BRepCheck_Shell(shell).Closed() != BRepCheck_NoError:
            return False
        any_shell = True
        explorer.Next()
    return any_shell


def _collect_errors(analyzer, shape) -> list[str]:
    """Human-readable BRepCheck errors, de-duplicated, most specific first.

    Visits each sub-shape exactly once via ``TopTools_IndexedMapOfShape``; a plain
    ``TopExp_Explorer`` yields shared edges and vertices once per parent context and would
    report the same defect several times.
    """
    from OCP.BRepCheck import BRepCheck_NoError, BRepCheck_Status
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape

    type_names = {TopAbs_FACE: "Face", TopAbs_EDGE: "Edge", TopAbs_VERTEX: "Vertex"}
    seen: dict[str, None] = {}
    for sub_type, type_name in type_names.items():
        sub_map = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, sub_type, sub_map)
        for i in range(1, sub_map.Size() + 1):
            result = analyzer.Result(sub_map.FindKey(i))
            if result is None:
                continue
            for status in result.Status():
                if status != BRepCheck_NoError:
                    seen.setdefault(f"{type_name}: {BRepCheck_Status(status).name}", None)
            if len(seen) >= _MAX_ERRORS:
                return list(seen)
    return list(seen)


def _mesh_errors(step_path) -> tuple[list[str], bool]:
    """Delegate the tessellation gate to CADGenBench; ``(errors, ran)``.

    Deliberately not reimplemented — see the module docstring. Tries a plain import first,
    then ``$CAD_GEN_CADGENBENCH_SRC``, then a ``cadgenbench/src`` clone beside the CWD, which
    is where this repo keeps its (gitignored) copy.
    """
    candidates = [os.environ.get("CAD_GEN_CADGENBENCH_SRC"), str(Path.cwd() / "cadgenbench" / "src")]
    for candidate in candidates:
        if candidate and Path(candidate).is_dir() and candidate not in sys.path:
            sys.path.append(candidate)
    try:
        from cadgenbench.common.validity import validate_step
    except Exception:  # noqa: BLE001 — an absent/broken benchmark clone is not our failure
        return [], False
    try:
        result = validate_step(step_path)
    except Exception as exc:  # noqa: BLE001 — never let the delegate's failure fail the gate
        return [f"mesh gate could not run: {type(exc).__name__}: {exc}"[:200]], False
    # Only the mesh half is wanted here; the BREP half already ran above and would duplicate.
    return [e for e in result.topology_errors if e.startswith("mesh")], True


def _validate(step_path) -> dict:
    from OCP.BRepCheck import BRepCheck_Analyzer

    shape = _read_shape(step_path)
    if shape.IsNull():
        return {
            "is_valid": False,
            "is_watertight": False,
            "mesh_checked": False,
            "errors": [],
            "unknown_reason": "STEP file loaded as a null shape",
        }

    analyzer = BRepCheck_Analyzer(shape)
    brep_ok = bool(analyzer.IsValid())
    errors = _collect_errors(analyzer, shape) if not brep_ok else []

    # Belt and braces, as the grader does it: a closed shell whose curves are invalid is not
    # usefully watertight, so both conditions must hold.
    is_watertight = _is_watertight(shape) and not errors
    if brep_ok and not is_watertight:
        errors.append(
            "BREP not watertight: at least one shell has open / naked edges "
            "(failed _is_watertight)"
        )

    # Mesh only an otherwise-valid shape: tessellating a broken BREP burns time and reports
    # derived errors that mislead more than the root cause already recorded.
    mesh_checked = False
    if brep_ok and is_watertight:
        mesh_errors, mesh_checked = _mesh_errors(step_path)
        errors.extend(mesh_errors)

    return {
        "is_valid": brep_ok and is_watertight and not errors,
        "is_watertight": is_watertight,
        "mesh_checked": mesh_checked,
        "errors": errors[:_MAX_ERRORS],
        "unknown_reason": None,
    }


def _solids(shape) -> list:
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS

    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    out = []
    while explorer.More():
        out.append(TopoDS.Solid_s(explorer.Current()))
        explorer.Next()
    return out


def _rung_boolean_regeneration(shape):
    """Rebuild the topology by intersecting the shape with itself.

    Forces OCCT's boolean engine to re-derive every face and wire from scratch, which repairs
    some orientation defects that ShapeFix cannot touch. The result is a COMPOUND; only a
    single-solid result is usable, since the submission contract wants one solid and a
    regeneration that fragments the part into several is a different shape, not a repair.
    """
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common

    op = BRepAlgoAPI_Common(shape, shape)
    op.Build()
    solids = _solids(op.Shape())
    return solids[0] if len(solids) == 1 else None


def _rung_shape_fix(shape):
    from OCP.ShapeFix import ShapeFix_Shape

    fixer = ShapeFix_Shape(shape)
    fixer.SetPrecision(1e-4)
    fixer.SetMaxTolerance(1.0)
    fixer.Perform()
    return fixer.Shape()


def _rung_unify_same_domain(shape):
    from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain

    upgrader = ShapeUpgrade_UnifySameDomain(shape, True, True, True)
    upgrader.Build()
    return upgrader.Shape()


_REPAIR_RUNGS = (
    ("boolean-regeneration", _rung_boolean_regeneration),
    ("shape-fix", _rung_shape_fix),
    ("unify-same-domain", _rung_unify_same_domain),
)


def _repair_shape(step_path, out_path) -> dict:
    """Walk the repair ladder, keeping the first rung that produces a valid solid.

    Every rung is verified by writing it to STEP, reading it back and re-running the full
    gate — never by asking ``BRepCheck_Analyzer`` about the in-memory shape. That distinction
    is the whole point: on v3 sample 202 the boolean rung makes the in-memory analyzer return
    True at 0.000000% volume change, and the defect comes straight back on the round-trip.
    STEP is a lossy serialisation of a B-rep, and the round-tripped file is the only artifact
    the grader ever sees, so it is the only one worth asking.
    """
    original = _read_shape(step_path)
    base_volume = _volume(original)
    attempted: list[str] = []

    for name, rung in _REPAIR_RUNGS:
        try:
            repaired = rung(original)
        except Exception as exc:  # noqa: BLE001 — a rung that throws is just a rung that failed
            attempted.append(f"{name}: raised {type(exc).__name__}")
            continue
        if repaired is None or repaired.IsNull():
            attempted.append(f"{name}: produced nothing usable")
            continue
        drift = abs(_volume(repaired) - base_volume) / max(abs(base_volume), 1e-9)
        if drift > _REPAIR_VOLUME_REL_TOL:
            attempted.append(f"{name}: rejected, changed volume by {drift * 100:.4f}%")
            continue
        _write_shape(repaired, out_path)
        verdict = _validate(out_path)
        if verdict["is_valid"]:
            verdict["repair_rung"] = name
            verdict["repair_attempted"] = attempted
            return verdict
        attempted.append(f"{name}: still invalid after the STEP round-trip")

    Path(out_path).unlink(missing_ok=True)
    return {
        "is_valid": False,
        "is_watertight": False,
        "mesh_checked": False,
        "errors": [f"repair failed — {'; '.join(attempted)}"] if attempted else ["repair failed"],
        "unknown_reason": None,
        "repair_rung": None,
        "repair_attempted": attempted,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "repair"))
    parser.add_argument("--step", required=True)
    parser.add_argument("--out")
    parser.add_argument("--report", help="where to write the JSON verdict (default: stdout)")
    args = parser.parse_args()

    if args.command == "repair":
        if not args.out:
            parser.error("--out is required for repair")
        payload = _repair_shape(args.step, args.out)
    else:
        payload = _validate(args.step)
    if args.report:
        Path(args.report).write_text(json.dumps(payload))
    else:
        json.dump(payload, sys.stdout)


if __name__ == "__main__":
    main()
