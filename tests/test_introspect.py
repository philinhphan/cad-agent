"""Tests for the read-only introspection harness.

Like test_executor.py these spawn real subprocesses running real CadQuery — slow-ish
but they are the contract the generator's selector-verification stands on.
"""

from cad_gen.sandbox.executor import introspect_cad_code

BOX = """
import cadquery as cq

result = cq.Workplane("XY").box(20, 30, 10)
"""

BOX_WITH_HOLE = """
import cadquery as cq

result = cq.Workplane("XY").box(20, 30, 10).faces(">Z").workplane().hole(6)
"""

CRASHES = """
import cadquery as cq

result = cq.Workplane("XY").box(undefined_width, 10, 10)
"""


def test_describe_reports_topology(tmp_path):
    r = introspect_cad_code(BOX, {"mode": "describe"}, tmp_path)

    assert r.ok, r.error
    data = r.data
    assert data["bbox_mm"] == [20.0, 30.0, 10.0]
    assert data["n_solids"] == 1
    assert data["n_faces"] == 6
    assert data["n_edges"] == 12
    # a plain box is all planar faces and straight edges
    assert {g["type"] for g in data["faces"]} == {"PLANE"}
    assert {g["type"] for g in data["edges"]} == {"LINE"}


def test_describe_distinguishes_curved_features(tmp_path):
    r = introspect_cad_code(BOX_WITH_HOLE, {"mode": "describe"}, tmp_path)

    assert r.ok, r.error
    edge_types = {g["type"] for g in r.data["edges"]}
    assert "CIRCLE" in edge_types  # the drilled hole adds circular edges


def test_selector_matches_expected_edges(tmp_path):
    r = introspect_cad_code(
        BOX, {"mode": "selector", "target": "edges", "selector": "|Z"}, tmp_path
    )

    assert r.ok, r.error
    assert r.data["count"] == 4  # the four vertical corner edges of a plain box
    assert r.data["selector_error"] is None
    assert all(e["dir"] == [0.0, 0.0, 1.0] or e["dir"] == [0.0, 0.0, -1.0]
               for e in r.data["matches"])


def test_empty_but_valid_selector_reports_zero(tmp_path):
    r = introspect_cad_code(
        BOX, {"mode": "selector", "target": "edges", "selector": "%CIRCLE"}, tmp_path
    )

    assert r.ok, r.error  # the CODE built fine
    assert r.data["count"] == 0  # a box has no circular edges
    assert r.data["selector_error"] is None


def test_malformed_selector_returns_diagnostic_not_crash(tmp_path):
    r = introspect_cad_code(
        BOX, {"mode": "selector", "target": "edges", "selector": ">>X[99]"}, tmp_path
    )

    assert r.ok, "a bad selector is a result, not a code failure"
    assert r.data["count"] == 0
    assert "IndexError" in r.data["selector_error"]


def test_code_failure_reported_as_not_ok(tmp_path):
    r = introspect_cad_code(CRASHES, {"mode": "describe"}, tmp_path)

    assert not r.ok
    assert r.data is None
    assert "NameError" in r.error
    assert "undefined_width" in r.error


def test_no_geometry_reported(tmp_path):
    r = introspect_cad_code("x = 42", {"mode": "describe"}, tmp_path)

    assert not r.ok
    assert "result" in r.error
