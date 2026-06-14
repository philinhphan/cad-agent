import numpy as np
import pytest
import trimesh
from PIL import Image

from cad_gen.models import GeometryMetrics
from cad_gen.rendering.renderer import _rasterize_view, render_views


@pytest.fixture
def box_stl(tmp_path):
    mesh = trimesh.creation.box(extents=(20, 30, 10))
    path = tmp_path / "box.stl"
    mesh.export(path)
    return path


def test_rasterizer_resolves_hidden_surfaces():
    """Z-buffer correctness: the near face must win every overlapped pixel.

    (The previous painter's-algorithm renderer drew phantom rims/recesses
    that misled the vision critic.)
    """
    mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))

    rgb, depth, extent = _rasterize_view(
        mesh, elev=0, azim=-90, center=np.zeros(3), half=8.0, res=200
    )

    # camera sits on -Y; the near face is y=-5, i.e. view-axis depth +5
    assert depth[100, 100] == pytest.approx(5.0, abs=1e-6)


def test_rasterizer_renders_flat_faces_uniformly():
    """Top view of a plain plate shows exactly one color inside the outline —
    any tonal band would read as a phantom rim/recess to the critic."""
    mesh = trimesh.creation.box(extents=(60.0, 40.0, 8.0))

    rgb, depth, extent = _rasterize_view(
        mesh, elev=90, azim=-90, center=np.zeros(3), half=40.0, res=400
    )

    hit = depth > -np.inf
    assert hit.sum() > 1000, "part must actually be rendered"
    colors = rgb[hit]
    assert (colors == colors[0]).all(), "flat top face must be perfectly uniform"


def test_rasterizer_view_extent_in_world_mm():
    mesh = trimesh.creation.box(extents=(10.0, 10.0, 10.0))

    rgb, depth, extent = _rasterize_view(
        mesh, elev=90, azim=-90, center=np.zeros(3), half=8.0, res=100
    )

    assert extent == pytest.approx((-8.0, 8.0, -8.0, 8.0))


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


def test_render_views_with_edges_still_renders_geometry(box_stl, tmp_path):
    from PIL import Image

    out = tmp_path / "edged.png"
    render_views(box_stl, out, with_edges=True)

    assert out.exists()
    levels = Image.open(out).convert("L").getcolors(maxcolors=1_000_000)
    assert len(levels) > 20, "edge overlay must not blank the render"


def test_render_sections_produces_png(box_stl, tmp_path):
    from cad_gen.rendering.renderer import render_sections

    out = tmp_path / "sections.png"
    result = render_sections(box_stl, out)

    assert result == out
    assert out.exists() and out.stat().st_size > 5_000


def test_render_sections_returns_none_for_empty_mesh(tmp_path):
    from cad_gen.rendering.renderer import render_sections

    fake = tmp_path / "empty.stl"
    fake.write_bytes(b"solid x\nendsolid x\n")

    assert render_sections(fake, tmp_path / "s.png") is None


def test_render_sections_accepts_target_and_metrics(box_stl, tmp_path):
    """New signature is back-compatible: the orchestrator passes target/metrics hints,
    but planes derive from the mesh, so a plain box with no target still renders."""
    from cad_gen.rendering.renderer import render_sections

    out = render_sections(box_stl, tmp_path / "s.png", target=None, metrics=None)
    assert out is not None and out.exists()


def test_sections_reveal_each_layer_of_a_stepped_part(tmp_path):
    """A raised pad / step produces a distinct horizontal slab; the feature-aligned
    sections take a plan cut inside each layer (what a single mid-plane cut would miss)."""
    from cad_gen.rendering.renderer import _slab_levels, render_sections

    base = trimesh.creation.box(extents=(40, 40, 10))  # z ∈ [-5, 5]
    top = trimesh.creation.box(extents=(20, 20, 10))
    top.apply_translation([0, 0, 10])                  # z ∈ [5, 15]
    mesh = trimesh.util.concatenate([base, top])

    # two slabs (z≈0 and z≈10) → the pad layer gets its own plan section
    assert len(_slab_levels(mesh)) >= 2

    path = tmp_path / "stepped.stl"
    mesh.export(path)
    out = render_sections(path, tmp_path / "sec.png")
    assert out is not None and out.exists()
