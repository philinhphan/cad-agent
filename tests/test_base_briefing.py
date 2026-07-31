"""Tests for the base-model feature inventory.

Motivation: editing instructions name features in engineering language ("the largest-diameter
bore", "the wall further in the -X direction"), and `inspect_geometry` returns 12 faces per
geometry type in traversal order. On a benchmark base carrying 334-2157 faces that is a
near-random sample, so the generator was locating features by writing its own face scans with
hand-tuned area thresholds — and landing in the wrong place.
"""

import cadquery as cq
import pytest

from cad_gen.base_briefing import build_base_briefing, format_briefing


@pytest.fixture(scope="module")
def plate_step(tmp_path_factory):
    """A 60x40x10 plate with two 6 mm through-holes and one 8 mm blind hole."""
    path = tmp_path_factory.mktemp("briefing") / "plate.step"
    plate = (
        cq.Workplane("XY")
        .box(60, 40, 10)
        .faces(">Z")
        .workplane()
        .pushPoints([(-20, 0), (20, 0)])
        .hole(6)
        .faces(">Z")
        .workplane()
        .pushPoints([(0, 0)])
        .cboreHole(8, 8.001, 0.001, depth=4)
    )
    cq.exporters.export(plate, str(path))
    return path


class TestBuildBaseBriefing:
    def test_reports_bulk_properties(self, plate_step):
        briefing = build_base_briefing(plate_step)

        assert briefing is not None
        assert briefing.n_solids == 1
        assert briefing.extent_mm == pytest.approx((60.0, 40.0, 10.0), rel=1e-6)
        assert briefing.bbox_min_mm == pytest.approx((-30.0, -20.0, -5.0), rel=1e-6)

    def test_finds_the_through_holes_by_radius_and_axis(self, plate_step):
        briefing = build_base_briefing(plate_step)

        through = [b for b in briefing.bores if b.radius_mm == pytest.approx(3.0, abs=1e-6)]
        assert len(through) == 2, "two 6 mm holes, and they must not merge into one feature"
        for bore in through:
            assert bore.full_circle
            assert bore.extent_mm == pytest.approx(10.0, rel=1e-3)
            # Canonicalised to one hemisphere so a hole authored either way clusters once.
            assert bore.axis == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)

    def test_two_coaxial_faces_of_one_hole_become_one_bore(self, plate_step):
        """A cylinder split by its seam is one feature; listing it twice is what made the
        old output unusable."""
        briefing = build_base_briefing(plate_step)
        radii = [round(b.radius_mm, 3) for b in briefing.bores]

        assert len(radii) == len(set(radii)) or radii.count(3.0) == 2

    def test_bores_are_sorted_largest_first(self, plate_step):
        briefing = build_base_briefing(plate_step)
        full = [b.radius_mm for b in briefing.bores if b.full_circle]

        assert full == sorted(full, reverse=True)

    def test_groups_the_plate_faces_by_normal_and_offset(self, plate_step):
        briefing = build_base_briefing(plate_step)
        top = [
            p
            for p in briefing.planes
            if p.normal == pytest.approx((0.0, 0.0, 1.0), abs=1e-6)
            and p.offset_mm == pytest.approx(5.0, abs=1e-6)
        ]

        assert len(top) == 1, "the top face is one wall, however many faces it is built from"

    def test_unreadable_file_returns_none_rather_than_raising(self, tmp_path):
        """The briefing is prompt enrichment; a base it cannot parse must not fail the run."""
        broken = tmp_path / "broken.step"
        broken.write_text("not a step file")

        assert build_base_briefing(broken) is None


class TestFormatBriefing:
    def test_renders_absolute_coordinates(self, plate_step):
        text = format_briefing(build_base_briefing(plate_step))

        assert "BASE MODEL BRIEFING" in text
        assert "bounding box: x -30.000..30.000" in text
        assert "HOLE/BORE" in text
        assert "Planar walls" in text
