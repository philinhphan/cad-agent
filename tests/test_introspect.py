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


# ── query mode: the imported-B-rep feature finder ──────────────────────────
#
# Motivation: describe() samples 12 entities per geomType in traversal order. On a
# benchmark base model carrying 334-2157 faces that is effectively a random 3%, so the
# generator could not locate the feature an edit instruction named and fell back to writing
# its own face scans with hand-tuned thresholds.

PLATE_WITH_BORES = """
import cadquery as cq

result = (
    cq.Workplane("XY")
    .box(60, 40, 10)
    .faces(">Z")
    .workplane()
    .pushPoints([(-20, 0), (20, 0)])
    .hole(6)
)
"""


def test_query_filters_faces_by_geometry_type(tmp_path):
    r = introspect_cad_code(
        PLATE_WITH_BORES, {"mode": "query", "target": "faces", "where": {"type": "CYLINDER"}}, tmp_path
    )

    assert r.ok, r.error
    assert r.data["count"] == 2  # two through-holes
    assert r.data["total"] > r.data["count"]  # ...out of all the plate's faces
    assert all(m["type"] == "CYLINDER" for m in r.data["matches"])


def test_query_reports_face_radius_so_a_bore_can_be_found_by_size(tmp_path):
    """CadQuery exposes radius on edges but not faces; this is what "the largest-diameter
    bore" resolves against."""
    r = introspect_cad_code(
        PLATE_WITH_BORES,
        {"mode": "query", "target": "faces", "where": {"type": "CYLINDER", "radius_min": 2.9}},
        tmp_path,
    )

    assert r.ok, r.error
    assert r.data["count"] == 2
    assert all(m["radius"] == 3.0 for m in r.data["matches"])
    assert all("axis" in m for m in r.data["matches"])


def test_query_filters_by_normal_and_region(tmp_path):
    top = introspect_cad_code(
        BOX, {"mode": "query", "target": "faces", "where": {"normal": [0, 0, 1]}}, tmp_path
    )
    right_half = introspect_cad_code(
        BOX,
        {"mode": "query", "target": "faces", "where": {"center_box": [0, -100, -100, 100, 100, 100]}},
        tmp_path,
    )

    assert top.ok and top.data["count"] == 1
    # 5 of the box's 6 faces have a centre at x >= 0; only the -X face does not.
    assert right_half.ok and right_half.data["count"] == 5


def test_query_returns_largest_first_and_reports_what_it_hid(tmp_path):
    r = introspect_cad_code(
        BOX, {"mode": "query", "target": "faces", "where": {}, "limit": 2}, tmp_path
    )

    assert r.ok, r.error
    assert r.data["count"] == 6 and len(r.data["matches"]) == 2
    assert r.data["truncated"] is True
    areas = [m["area"] for m in r.data["matches"]]
    assert areas == sorted(areas, reverse=True)


def test_query_reports_absolute_model_bounds(tmp_path):
    """Positioning a cutting primitive by coordinate needs where the part IS, not just its
    size — describe() only reports extents."""
    r = introspect_cad_code(BOX, {"mode": "query", "target": "faces", "where": {}}, tmp_path)

    assert r.ok, r.error
    assert r.data["model_bbox_min_mm"] == [-10.0, -15.0, -5.0]
    assert r.data["model_bbox_max_mm"] == [10.0, 15.0, 5.0]


def test_query_with_no_matches_is_a_result_not_an_error(tmp_path):
    r = introspect_cad_code(
        BOX, {"mode": "query", "target": "faces", "where": {"type": "CYLINDER"}}, tmp_path
    )

    assert r.ok, r.error
    assert r.data["count"] == 0
    assert r.data["query_error"] is None


def test_describe_output_is_unchanged_by_the_query_addition(tmp_path):
    """Generation mode reads describe() every run; its shape must not drift."""
    r = introspect_cad_code(BOX_WITH_HOLE, {"mode": "describe"}, tmp_path)

    assert r.ok, r.error
    assert set(r.data) == {
        "bbox_mm", "n_solids", "n_faces", "n_edges", "faces", "edges", "mode",
    }
    # ...and no radius/axis enrichment leaked into the face samples.
    cylinders = [g for g in r.data["faces"] if g["type"] == "CYLINDER"]
    assert cylinders and all("radius" not in f for g in cylinders for f in g["sample"])
