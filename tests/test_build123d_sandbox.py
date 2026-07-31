"""Tests for the build123d sandbox harnesses.

The build123d counterpart of test_executor.py / test_introspect.py: these spawn real
subprocesses running real build123d. build123d is an OPTIONAL extra
(`uv sync --extra build123d`), so the whole module skips when it is not installed —
the CadQuery path must stay testable in a default environment.
"""

import math

import pytest

pytest.importorskip("build123d", reason="build123d extra not installed")

from cad_gen.sandbox.executor import (  # noqa: E402 — must follow the importorskip guard
    introspect_cad_code,
    run_cad_code,
)

LIB = {"library": "build123d"}

GOOD_BOX = """
with BuildPart() as part:
    Box(20, 30, 10)
    with Locations((0, 0, 5)):
        Hole(radius=3)
result = part
"""

# The prompt permits assigning either the builder or its `.part`; both must work.
RESULT_IS_PART = """
with BuildPart() as part:
    Box(20, 30, 10)
    with Locations((0, 0, 5)):
        Hole(radius=3)
result = part.part
"""

EXPLICIT_IMPORT = """
from build123d import *

with BuildPart() as part:
    Box(5, 5, 5)
result = part
"""

SHOW_OBJECT_STYLE = """
with BuildPart() as part:
    Box(5, 5, 5)
show_object(part)
"""

SYNTAX_ERROR = """
with BuildPart( as part:
"""

RUNTIME_ERROR = """
with BuildPart() as part:
    Box(undefined_width, 10, 10)
result = part
"""

NO_RESULT = """
x = 42
"""

PRINTS_THEN_BUILDS = """
print("building the part now")
with BuildPart() as part:
    Box(2, 2, 2)
result = part
"""

PLAIN_BOX = """
with BuildPart() as part:
    Box(20, 20, 20)
result = part
"""

EDIT_IMPORTS_BASE = """
base = import_step("input.step")
with BuildPart() as edited:
    add(base)
    Cylinder(radius=3, height=30, mode=Mode.SUBTRACT)
result = edited
"""


def test_good_code_produces_artifacts_and_metrics(tmp_path):
    r = run_cad_code(GOOD_BOX, tmp_path, **LIB)

    assert r.success, r.error
    assert r.stl_path is not None and r.stl_path.exists()
    assert r.step_path is not None and r.step_path.exists()
    assert r.metrics is not None
    assert r.metrics.volume_mm3 == pytest.approx(
        20 * 30 * 10 - math.pi * 3**2 * 10, rel=1e-3
    )
    # Exact, not approximate: the harness must measure BEFORE export, or STL
    # tessellation sag of the curved hole face inflates the bounding box.
    assert r.metrics.bbox_mm == pytest.approx((20.0, 30.0, 10.0), abs=1e-6)
    assert r.metrics.n_solids == 1
    assert r.metrics.is_watertight is True
    assert r.code == GOOD_BOX
    assert r.duration_s > 0


def test_result_may_be_the_part_instead_of_the_builder(tmp_path):
    r = run_cad_code(RESULT_IS_PART, tmp_path, **LIB)

    assert r.success, r.error
    assert r.metrics.n_solids == 1


def test_api_is_preimported_and_explicit_import_still_works(tmp_path):
    """The namespace is pre-seeded, but writing `from build123d import *` is idempotent."""
    seeded = run_cad_code("with BuildPart() as p:\n    Box(5, 5, 5)\nresult = p", tmp_path / "a", **LIB)
    explicit = run_cad_code(EXPLICIT_IMPORT, tmp_path / "b", **LIB)

    assert seeded.success, seeded.error
    assert explicit.success, explicit.error
    assert explicit.metrics.volume_mm3 == pytest.approx(125, rel=1e-3)


def test_show_object_convention_supported(tmp_path):
    r = run_cad_code(SHOW_OBJECT_STYLE, tmp_path, **LIB)

    assert r.success, r.error
    assert r.metrics.volume_mm3 == pytest.approx(125, rel=1e-3)


def test_syntax_error_reported(tmp_path):
    r = run_cad_code(SYNTAX_ERROR, tmp_path, **LIB)

    assert not r.success
    assert "SyntaxError" in r.error
    assert r.metrics is None
    assert r.stl_path is None


def test_runtime_error_reports_traceback(tmp_path):
    r = run_cad_code(RUNTIME_ERROR, tmp_path, **LIB)

    assert not r.success
    assert "NameError" in r.error
    assert "undefined_width" in r.error


def test_missing_result_object_reported(tmp_path):
    r = run_cad_code(NO_RESULT, tmp_path, **LIB)

    assert not r.success
    assert "result" in r.error


def test_user_stdout_captured(tmp_path):
    r = run_cad_code(PRINTS_THEN_BUILDS, tmp_path, **LIB)

    assert r.success, r.error
    assert "building the part now" in r.stdout


def test_seed_files_enables_step_import(tmp_path):
    """The editing seam, build123d flavour: import_step reads the seeded input.step."""
    base = run_cad_code(PLAIN_BOX, tmp_path / "base", **LIB)
    assert base.success, base.error

    r = run_cad_code(
        EDIT_IMPORTS_BASE,
        tmp_path / "edit",
        seed_files={"input.step": base.step_path.read_bytes()},
        **LIB,
    )

    assert r.success, r.error
    assert (r.step_path.parent / "input.step").exists()  # seeded into the working dir
    assert r.metrics.n_solids == 1
    assert r.metrics.volume_mm3 < 20**3  # the cut removed material from the base box


# --- introspection -------------------------------------------------------------------


def test_describe_reports_topology(tmp_path):
    r = introspect_cad_code(GOOD_BOX, {"mode": "describe"}, tmp_path, **LIB)

    assert r.ok, r.error
    assert r.data["bbox_mm"] == pytest.approx([20.0, 30.0, 10.0], abs=1e-6)
    assert r.data["n_solids"] == 1
    assert r.data["n_edges"] > 0
    # Same key shape as the CadQuery probe, so agents/generator.format_describe is shared.
    assert {"bbox_mm", "n_solids", "n_faces", "n_edges", "faces", "edges"} <= set(r.data)
    assert all({"type", "count", "sample", "truncated"} <= set(g) for g in r.data["faces"])


def test_describe_does_not_export(tmp_path):
    introspect_cad_code(GOOD_BOX, {"mode": "describe"}, tmp_path, **LIB)

    assert not (tmp_path / "model.stl").exists()
    assert not (tmp_path / "model.step").exists()


def test_selection_expression_reports_matches(tmp_path):
    query = {"mode": "selection", "expression": "result.edges().filter_by(GeomType.CIRCLE)"}
    r = introspect_cad_code(GOOD_BOX, query, tmp_path, **LIB)

    assert r.ok, r.error
    assert r.data["count"] == 2  # top and bottom rim of the through-hole
    assert r.data["selection_error"] is None
    assert all(m["type"] == "CIRCLE" for m in r.data["matches"])
    assert r.data["matches"][0]["radius"] == pytest.approx(3.0)


def test_selection_accepts_a_single_shape(tmp_path):
    """Indexing a ShapeList yields one Shape, not a list — that must still be reportable."""
    query = {"mode": "selection", "expression": "result.faces().sort_by(Axis.Z)[-1]"}
    r = introspect_cad_code(GOOD_BOX, query, tmp_path, **LIB)

    assert r.ok, r.error
    assert r.data["count"] == 1


def test_bad_expression_is_a_result_not_a_crash(tmp_path):
    """Mirrors introspect.py: a bad selector is the diagnostic the model asked for."""
    query = {"mode": "selection", "expression": "result.edges().filter_by(Axis.NOPE)"}
    r = introspect_cad_code(GOOD_BOX, query, tmp_path, **LIB)

    assert r.ok  # the CODE built fine; only the expression was wrong
    assert r.data["count"] == 0
    assert "AttributeError" in r.data["selection_error"]


def test_non_shape_expression_reported(tmp_path):
    query = {"mode": "selection", "expression": "result.volume"}
    r = introspect_cad_code(GOOD_BOX, query, tmp_path, **LIB)

    assert r.ok
    assert r.data["count"] == 0
    assert "float" in r.data["selection_error"]


def test_broken_code_fails_the_probe(tmp_path):
    """A failure to BUILD is a code failure (ok=False), unlike a bad expression."""
    query = {"mode": "selection", "expression": "result.edges()"}
    r = introspect_cad_code(RUNTIME_ERROR, query, tmp_path, **LIB)

    assert not r.ok
    assert "NameError" in r.error
