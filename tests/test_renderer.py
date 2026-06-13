import pytest
import trimesh
from PIL import Image

from cad_gen.models import GeometryMetrics
from cad_gen.rendering.renderer import render_views


@pytest.fixture
def box_stl(tmp_path):
    mesh = trimesh.creation.box(extents=(20, 30, 10))
    path = tmp_path / "box.stl"
    mesh.export(path)
    return path


def test_renders_composite_png_with_metrics_banner(box_stl, tmp_path):
    out = tmp_path / "views.png"
    metrics = GeometryMetrics(
        volume_mm3=6000.0,
        bbox_mm=(20.0, 30.0, 10.0),
        center_of_mass=(0.0, 0.0, 0.0),
        n_solids=1,
        n_faces=6,
        is_watertight=True,
    )

    result = render_views(box_stl, out, metrics=metrics)

    assert result == out
    assert out.exists()
    img = Image.open(out)
    assert img.width >= 1000, "composite should be high-res enough for vision critique"
    assert img.height >= 700
    # Shaded geometry produces many gray levels; a blank canvas produces ~1.
    levels = img.convert("L").getcolors(maxcolors=1_000_000)
    assert len(levels) > 20, "image looks blank — no geometry rendered"


def test_renders_without_metrics(box_stl, tmp_path):
    out = tmp_path / "plain.png"

    render_views(box_stl, out)

    assert out.exists()
    assert out.stat().st_size > 10_000
