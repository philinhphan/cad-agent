"""Tests for the sandbox executor.

These spawn real subprocesses running real CadQuery — slow-ish (~3s each,
OCCT import) but they are the contract the whole agent loop stands on.
"""

import math
from pathlib import Path

import pytest

from cad_gen.sandbox.executor import run_cad_code

GOOD_BOX = """
import cadquery as cq

result = (
    cq.Workplane("XY")
    .box(20, 30, 10)
    .faces(">Z").workplane()
    .hole(6)
)
"""

SHOW_OBJECT_STYLE = """
import cadquery as cq

box = cq.Workplane("XY").box(5, 5, 5)
show_object(box)
"""

SYNTAX_ERROR = """
result = cq.Workplane("XY".box(
"""

RUNTIME_ERROR = """
import cadquery as cq

result = cq.Workplane("XY").box(undefined_width, 10, 10)
"""

NO_RESULT = """
x = 42
"""

PRINTS_THEN_BUILDS = """
import cadquery as cq

print("building the part now")
result = cq.Workplane("XY").box(2, 2, 2)
"""

INFINITE_LOOP = """
import time

while True:
    time.sleep(0.1)
"""


def test_good_code_produces_artifacts_and_metrics(tmp_path):
    r = run_cad_code(GOOD_BOX, tmp_path)

    assert r.success, r.error
    assert r.stl_path is not None and r.stl_path.exists()
    assert r.step_path is not None and r.step_path.exists()
    assert r.metrics is not None
    assert r.metrics.volume_mm3 == pytest.approx(
        20 * 30 * 10 - math.pi * 3**2 * 10, rel=1e-3
    )
    # exact, not approximately: measuring after STL export picks up the mesh
    # tessellation sag of curved faces and pollutes the critic's dimension checks
    assert r.metrics.bbox_mm == pytest.approx((20.0, 30.0, 10.0), abs=1e-6)
    assert r.metrics.n_solids == 1
    assert r.metrics.is_watertight is True
    assert r.code == GOOD_BOX
    assert r.duration_s > 0


def test_show_object_convention_supported(tmp_path):
    r = run_cad_code(SHOW_OBJECT_STYLE, tmp_path)

    assert r.success, r.error
    assert r.metrics.volume_mm3 == pytest.approx(125, rel=1e-3)


def test_syntax_error_reported(tmp_path):
    r = run_cad_code(SYNTAX_ERROR, tmp_path)

    assert not r.success
    assert "SyntaxError" in r.error
    assert r.metrics is None
    assert r.stl_path is None


def test_runtime_error_reports_traceback(tmp_path):
    r = run_cad_code(RUNTIME_ERROR, tmp_path)

    assert not r.success
    assert "NameError" in r.error
    assert "undefined_width" in r.error


def test_missing_result_object_reported(tmp_path):
    r = run_cad_code(NO_RESULT, tmp_path)

    assert not r.success
    assert "result" in r.error


def test_user_stdout_captured(tmp_path):
    r = run_cad_code(PRINTS_THEN_BUILDS, tmp_path)

    assert r.success, r.error
    assert "building the part now" in r.stdout


def test_relative_out_dir_supported(tmp_path, monkeypatch):
    """Live CLI runs use a relative out_dir (runs/...); the subprocess must
    still find the code file even though its cwd is changed to out_dir."""
    monkeypatch.chdir(tmp_path)

    r = run_cad_code(GOOD_BOX, Path("runs/iter_01/attempt_01"))

    assert r.success, r.error
    assert r.stl_path.exists()


def test_infinite_loop_times_out(tmp_path):
    r = run_cad_code(INFINITE_LOOP, tmp_path, timeout_s=10)

    assert not r.success
    assert "timed out" in r.error.lower()
