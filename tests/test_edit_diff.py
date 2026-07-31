"""Tests for the boolean before/after edit diff.

Motivation: the existing volume/bbox no-op guard only answers "did anything change?".
The v3 audit shows that is not the question that matters — samples 201 and 209 changed
plenty (200% and 46% of the base volume respectively) and are both catastrophic, while
sample 205 changed only 0.14% and grew a 116 mm spike out of the side of the part.

These spawn real subprocesses running real OCCT booleans, like test_executor.py.
"""

import cadquery as cq
import pytest

from cad_gen.edit_diff import describe_edit_diff, edit_diff_report
from cad_gen.models import EditDiff, EditRegion

BOX = (40.0, 30.0, 20.0)


@pytest.fixture(scope="module")
def base_step(tmp_path_factory):
    path = tmp_path_factory.mktemp("diff") / "base.step"
    cq.exporters.export(cq.Workplane("XY").box(*BOX), str(path))
    return path


def _export(tmp_path, name, workplane):
    path = tmp_path / name
    cq.exporters.export(workplane, str(path))
    return path


class TestEditDiff:
    def test_pocket_cut_is_measured_as_removed_material(self, base_step, tmp_path):
        """A 10x10x5 pocket in the top face: 500 mm3 removed, nothing added."""
        pocket = (
            cq.Workplane("XY")
            .box(*BOX)
            .faces(">Z")
            .workplane()
            .rect(10, 10)
            .cutBlind(-5)
        )
        diff = edit_diff_report(base_step, _export(tmp_path, "pocket.step", pocket))

        assert diff.evaluated, diff.skipped_reason
        assert diff.removed_volume_mm3 == pytest.approx(500.0, rel=1e-3)
        assert diff.added_volume_mm3 == pytest.approx(0.0, abs=1e-3)
        assert len(diff.removed) == 1
        assert diff.changed_fraction == pytest.approx(500.0 / 24000.0, rel=1e-3)

    def test_added_boss_is_measured_and_flagged_as_outside_the_envelope(
        self, base_step, tmp_path
    ):
        """Material stuck on top of the part reaches past the base bounding box."""
        boss = cq.Workplane("XY").box(*BOX).faces(">Z").workplane().rect(10, 10).extrude(5)
        diff = edit_diff_report(base_step, _export(tmp_path, "boss.step", boss))

        assert diff.evaluated, diff.skipped_reason
        assert diff.added_volume_mm3 == pytest.approx(500.0, rel=1e-3)
        assert diff.outside_base_bbox_mm3 == pytest.approx(500.0, rel=1e-3)
        assert diff.overshoot_mm == pytest.approx(5.0, rel=1e-2)
        # 5 mm past a 20 mm extent. The FRACTION is what separates a legitimate "raise this
        # wall by 5 mm" from v3 sample 205's 116 mm spike; the raw presence of overshoot
        # cannot, because both have it.
        assert diff.overshoot_fraction == pytest.approx(0.25, rel=1e-2)

    def test_identical_solid_reports_no_change(self, base_step, tmp_path):
        same = cq.Workplane("XY").box(*BOX)
        diff = edit_diff_report(base_step, _export(tmp_path, "same.step", same))

        assert diff.evaluated, diff.skipped_reason
        assert diff.removed_volume_mm3 == pytest.approx(0.0, abs=1e-3)
        assert diff.added_volume_mm3 == pytest.approx(0.0, abs=1e-3)
        assert diff.locality == 0.0

    def test_rebuilt_part_reads_as_a_whole_part_change(self, base_step, tmp_path):
        """The from-scratch-rebuild signature: both directions return most of the part.

        This is v3 sample 201, which removed 2 548 445 mm3 AND added 2 548 505 mm3 — roughly
        the whole part, twice — while passing the volume/bbox no-op guard. Rotating the box
        45 degrees reproduces the *shape* of that failure in miniature: material lost and
        gained in both directions at once, spread over the entire part. The magnitudes here
        are milder (a rotated box still overlaps itself substantially, ~20% each way, where
        201 was ~100%), so this pins the signature, not a threshold.
        """
        rotated = cq.Workplane("XY").box(*BOX).rotate((0, 0, 0), (0, 0, 1), 45)
        diff = edit_diff_report(base_step, _export(tmp_path, "rotated.step", rotated))

        assert diff.evaluated, diff.skipped_reason
        # Both directions substantial at once is what no genuine local edit ever produces:
        # a real edit adds material or removes it, in one place.
        assert diff.removed_volume_mm3 > 0.1 * 24000.0
        assert diff.added_volume_mm3 > 0.1 * 24000.0
        assert diff.locality > 0.9

    def test_missing_file_withholds_rather_than_rejects(self, base_step, tmp_path):
        diff = edit_diff_report(base_step, tmp_path / "nope.step")

        assert not diff.evaluated
        assert diff.skipped_reason is not None


class TestDescribeEditDiff:
    def test_digest_reports_overshoot_when_the_part_grew(self):
        text = describe_edit_diff(
            EditDiff(
                added=[
                    EditRegion(
                        volume_mm3=500.0, bbox_mm=(10, 10, 5), center_mm=(0, 0, 12)
                    )
                ],
                added_volume_mm3=500.0,
                base_volume_mm3=24000.0,
                changed_fraction=0.0208,
                locality=0.4,
                outside_base_bbox_mm3=500.0,
                overshoot_mm=5.0,
                overshoot_fraction=0.25,
            )
        )

        assert "outside the base envelope" in text
        assert "5.00 mm" in text

    def test_digest_is_silent_about_overshoot_when_there_is_none(self):
        text = describe_edit_diff(
            EditDiff(removed_volume_mm3=500.0, base_volume_mm3=24000.0, changed_fraction=0.02)
        )

        assert "outside the base envelope" not in text
