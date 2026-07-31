"""Tests for the local CADGenBench validity gate.

Motivation: an audit of submission v3 found 7 of 81 candidates failing the benchmark's
validity gate — 5 of them editing samples — while the harness's own signal said they were
fine. Two of those (238, 240) are single watertight solids by trimesh and still fail
``BRepCheck_Analyzer``, so they shipped as guaranteed zeros.

These spawn real subprocesses running real OCCT, like test_executor.py and test_introspect.py.
"""

import json
import subprocess
import sys
from pathlib import Path

import cadquery as cq

from cad_gen.step_validity import (
    StepValidity,
    check_step_validity,
    describe_validity,
    repair_step,
)


def _box_step(tmp_path: Path, name: str = "box.step") -> Path:
    path = tmp_path / name
    cq.exporters.export(cq.Workplane("XY").box(20, 30, 10), str(path))
    return path


def _open_shell_step(tmp_path: Path) -> Path:
    """A single face exported as STEP: a surface, not a closed solid.

    The cheapest way to produce something the watertight half of the gate must reject
    without depending on a fixture file.
    """
    path = tmp_path / "open.step"
    face = cq.Workplane("XY").box(20, 30, 10).faces(">Z").val()
    cq.exporters.export(cq.Workplane("XY").add(face), str(path))
    return path


class TestCheckStepValidity:
    def test_clean_solid_passes(self, tmp_path):
        result = check_step_validity(_box_step(tmp_path))

        assert result.evaluated
        assert result.is_valid
        assert result.is_watertight
        assert result.errors == ()

    def test_open_shell_is_not_watertight(self, tmp_path):
        result = check_step_validity(_open_shell_step(tmp_path))

        assert result.evaluated
        assert not result.is_valid
        assert not result.is_watertight
        assert result.errors, "a rejection must say why — the reason is fed to the generator"

    def test_missing_file_is_unknown_not_invalid(self, tmp_path):
        """No verdict was reached, so nothing may be concluded from `is_valid`."""
        result = check_step_validity(tmp_path / "nope.step")

        assert not result.evaluated
        assert result.unknown_reason is not None

    def test_timeout_degrades_to_unknown(self, tmp_path):
        result = check_step_validity(_box_step(tmp_path), timeout_s=0.001)

        assert not result.evaluated
        assert "timed out" in (result.unknown_reason or "")


class TestDescribeValidity:
    def test_failure_keeps_occt_status_names_verbatim(self):
        """The status name is the diagnostic; paraphrasing it would lose the whole signal."""
        text = describe_validity(
            StepValidity(
                is_valid=False,
                is_watertight=False,
                mesh_checked=False,
                errors=("Face: BRepCheck_UnorientableShape",),
            )
        )

        assert "FAILS" in text
        assert "Face: BRepCheck_UnorientableShape" in text

    def test_unevaluated_gate_does_not_read_as_a_failure(self):
        text = describe_validity(
            StepValidity(
                is_valid=False,
                is_watertight=False,
                mesh_checked=False,
                unknown_reason="timed out",
            )
        )

        assert "could not be checked" in text
        assert "FAILS" not in text

    def test_pass_flags_an_unrun_mesh_gate(self):
        """A pass with the mesh gate skipped is weaker than a full pass and must say so."""
        text = describe_validity(
            StepValidity(is_valid=True, is_watertight=True, mesh_checked=False)
        )

        assert "tessellation gate not run" in text


class TestRepairStep:
    def test_repair_reports_failure_rather_than_inventing_success(self, tmp_path):
        """The whole point of the ladder is that it verifies through a STEP round-trip.

        On v3 sample 202 ``BRepAlgoAPI_Common(s, s)`` makes the in-memory analyzer return
        True at 0.000000% volume change, and the defect returns on re-read. A ladder that
        trusted the in-memory answer would report a repair that does not exist.
        """
        out = tmp_path / "repaired.step"
        result = repair_step(_open_shell_step(tmp_path), out)

        assert not result.is_valid
        assert not out.exists(), "a failed repair must not leave a file behind to be shipped"

    def test_repair_of_a_valid_solid_succeeds_without_moving_volume(self, tmp_path):
        out = tmp_path / "repaired.step"
        result = repair_step(_box_step(tmp_path), out)

        assert result.is_valid
        assert out.exists()


def test_volume_preserving_check_rejects_a_geometry_changing_rung(tmp_path):
    """A rung that changes volume is replacing the candidate, not repairing it.

    Measured: ShapeUpgrade_UnifySameDomain moves v3 sample 240 by ~4000 mm3. Accepting that
    would silently swap in different geometry, which is worse than shipping the original.
    """
    from cad_gen import step_validity

    step = _box_step(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(Path(step_validity.__file__)), "check", "--step", str(step)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert json.loads(proc.stdout)["is_valid"] is True
