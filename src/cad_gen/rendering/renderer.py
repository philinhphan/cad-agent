"""Render an STL into a 2x2 multi-view composite PNG for the vision critic.

Triangles are rasterized with a software z-buffer (numpy): exact hidden
surfaces, headless-safe everywhere (no OpenGL), no extra dependencies.
matplotlib (Agg) is used only for figure layout, axes, and labels.

A z-buffer matters here: painter's-algorithm polygon sorting draws phantom
rims/recesses on flat faces, which the vision critic then reports as real
geometry errors.
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh

from cad_gen.models import GeometryMetrics

# (title, elev, azim, x-axis label, y-axis label); labels None = screen-space view
_VIEWS = [
    ("isometric", 30, -55, None, None),
    ("front (X-Z)", 0, -90, "X (mm)", "Z (mm)"),
    ("top (X-Y)", 90, -90, "X (mm)", "Y (mm)"),
    ("right (Y-Z)", 0, 0, "Y (mm)", "Z (mm)"),
]

_BASE_COLOR = np.array([0.36, 0.56, 0.80])
_LIGHT_DIR = np.array([0.4, -0.6, 0.8]) / np.linalg.norm([0.4, -0.6, 0.8])
_BACKGROUND = np.array([1.0, 1.0, 1.0])
_RES = 560


def _view_basis(elev: float, azim: float):
    """Orthographic camera basis (right, up, toward-camera) for a matplotlib
    elev/azim pair."""
    e, a = math.radians(elev), math.radians(azim)
    forward = np.array(
        [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]
    )
    world_up = (
        np.array([0.0, 1.0, 0.0])
        if abs(forward[2]) > 0.999
        else np.array([0.0, 0.0, 1.0])
    )
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    return right, up, forward


def _rasterize_view(
    mesh: trimesh.Trimesh,
    elev: float,
    azim: float,
    center: np.ndarray,
    half: float,
    res: int = _RES,
):
    """Rasterize an orthographic view with a z-buffer.

    Returns (rgb image res*res*3 with row 0 at the bottom, depth buffer,
    extent (x0, x1, y0, y1) in world mm along the view's screen axes).
    """
    right, up, forward = _view_basis(elev, azim)

    vertices = mesh.vertices - center
    sx = vertices @ right
    sy = vertices @ up
    sz = vertices @ forward  # larger = closer to camera

    scale = (res - 1) / (2.0 * half)
    px = (sx + half) * scale
    py = (sy + half) * scale

    shade = 0.35 + 0.65 * np.clip(mesh.face_normals @ _LIGHT_DIR, 0.0, 1.0)
    colors = np.clip(shade[:, None] * _BASE_COLOR[None, :], 0.0, 1.0)

    rgb = np.tile(_BACKGROUND, (res, res, 1))
    depth = np.full((res, res), -np.inf)

    tx = px[mesh.faces]  # (n, 3)
    ty = py[mesh.faces]
    tz = sz[mesh.faces]

    for i in range(len(mesh.faces)):
        x0, x1, x2 = tx[i]
        y0, y1, y2 = ty[i]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue  # degenerate or perfectly edge-on

        cx0 = max(int(np.floor(min(x0, x1, x2))), 0)
        cx1 = min(int(np.ceil(max(x0, x1, x2))), res - 1)
        cy0 = max(int(np.floor(min(y0, y1, y2))), 0)
        cy1 = min(int(np.ceil(max(y0, y1, y2))), res - 1)
        if cx0 > cx1 or cy0 > cy1:
            continue

        gx, gy = np.meshgrid(
            np.arange(cx0, cx1 + 1), np.arange(cy0, cy1 + 1), indexing="xy"
        )
        w0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        w1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue

        z = w0 * tz[i, 0] + w1 * tz[i, 1] + w2 * tz[i, 2]
        patch_depth = depth[cy0 : cy1 + 1, cx0 : cx1 + 1]
        update = inside & (z > patch_depth)
        if not update.any():
            continue
        patch_depth[update] = z[update]
        rgb[cy0 : cy1 + 1, cx0 : cx1 + 1][update] = colors[i]

    cr = float(center @ right)
    cu = float(center @ up)
    extent = (cr - half, cr + half, cu - half, cu + half)
    return rgb, depth, extent


_EDGE_COLOR = np.array([0.07, 0.09, 0.12])


def _silhouette(hit: np.ndarray) -> np.ndarray:
    """Boundary pixels: hit pixels with a non-hit 4-neighbour (object outline)."""
    interior = hit.copy()
    interior[:-1, :] &= hit[1:, :]
    interior[1:, :] &= hit[:-1, :]
    interior[:, :-1] &= hit[:, 1:]
    interior[:, 1:] &= hit[:, :-1]
    return hit & ~interior


def _edge_mask(depth: np.ndarray) -> np.ndarray:
    """Silhouette + internal depth-discontinuity edges (crisp feature boundaries)."""
    hit = np.isfinite(depth)
    if not hit.any():
        return np.zeros_like(hit)
    d = depth.copy()
    dmin = float(d[hit].min())
    d[~hit] = dmin
    span = max(float(d[hit].max()) - dmin, 1e-6)
    thr = span * 0.06
    gx = np.zeros_like(d)
    gx[:, 1:] = np.abs(d[:, 1:] - d[:, :-1])
    gy = np.zeros_like(d)
    gy[1:, :] = np.abs(d[1:, :] - d[:-1, :])
    both_x = np.zeros_like(hit)
    both_x[:, 1:] = hit[:, 1:] & hit[:, :-1]
    both_y = np.zeros_like(hit)
    both_y[1:, :] = hit[1:, :] & hit[:-1, :]
    internal = (np.maximum(gx, gy) > thr) & (both_x | both_y)
    return internal | _silhouette(hit)


def _projected_span(depth: np.ndarray, extent: tuple) -> tuple[float, float] | None:
    """Width × height (world mm) of the rendered silhouette in this view's axes."""
    hit = np.isfinite(depth)
    if not hit.any():
        return None
    rows = np.where(hit.any(axis=1))[0]
    cols = np.where(hit.any(axis=0))[0]
    x0, x1, y0, y1 = extent
    res = depth.shape[0]
    wx = (cols[-1] - cols[0] + 1) / res * (x1 - x0)
    wy = (rows[-1] - rows[0] + 1) / res * (y1 - y0)
    return float(wx), float(wy)


def render_views(
    stl_path: Path,
    out_png: Path,
    metrics: GeometryMetrics | None = None,
    target=None,
    with_edges: bool = True,
) -> Path:
    """Render iso/front/top/right shaded views of `stl_path` into `out_png`.

    Edge outlines (silhouette + internal steps) and a per-view measured-span label make
    holes, counterbores and step heights legible to the vision critic.
    """
    mesh = trimesh.load(stl_path, force="mesh")
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    half = float((bounds[1] - bounds[0]).max()) * 0.6 + 1e-9

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (title, elev, azim, xlabel, ylabel) in zip(axes.flat, _VIEWS):
        rgb, depth, extent = _rasterize_view(mesh, elev, azim, center, half)
        if with_edges:
            rgb = rgb.copy()
            rgb[_edge_mask(depth)] = _EDGE_COLOR
        ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")
        ax.set_title(title)
        if xlabel is None:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)
            span = _projected_span(depth, extent)
            if span is not None:
                ax.text(
                    0.02,
                    0.98,
                    f"Δ{xlabel[0].lower()} {span[0]:.1f} · Δ{ylabel[0].lower()} {span[1]:.1f} mm",
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="#1b2433",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7),
                )

    if metrics is not None:
        x, y, z = metrics.bbox_mm
        watertight = {True: "yes", False: "NO", None: "?"}[metrics.is_watertight]
        mass = f"   ·   mass {metrics.mass_g:,.1f} g" if metrics.mass_g is not None else ""
        fig.suptitle(
            f"bbox {x:.1f} × {y:.1f} × {z:.1f} mm   ·   volume {metrics.volume_mm3:,.0f} mm³"
            f"   ·   solids {metrics.n_solids}   ·   watertight {watertight}{mass}",
            fontsize=13,
        )

    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def _slab_levels(mesh: trimesh.Trimesh, max_levels: int = 3) -> list[float]:
    """Z heights, one inside each horizontal 'slab' between distinct horizontal-face levels.

    A plan section taken inside each layer reveals outline changes a single mid-plane cut
    misses — a raised pad, a step, or a cutter that overshoots into a layer it shouldn't.
    """
    normals = mesh.face_normals
    z_centers = mesh.triangles_center[:, 2]
    horizontal = np.abs(normals[:, 2]) > 0.99
    if not horizontal.any():
        return []
    zs = np.unique(np.round(np.sort(z_centers[horizontal]), 1))
    if len(zs) < 2:
        return []
    slabs = [(a + b) / 2.0 for a, b in zip(zs[:-1], zs[1:]) if (b - a) > 0.5]
    if len(slabs) > max_levels:
        idx = sorted({int(j) for j in np.linspace(0, len(slabs) - 1, max_levels).round()})
        slabs = [slabs[i] for i in idx]
    return slabs


def _section_specs(mesh: trimesh.Trimesh):
    """Section planes: two vertical mid-plane profiles + a horizontal plan section inside
    each layer. Tuple is (title, origin, normal, u-axis, v-axis, xlabel, ylabel)."""
    cx, cy, _ = mesh.bounds.mean(axis=0)
    specs = [
        (f"profile X-Z @ Y={cy:.0f}", (cx, cy, 0.0), (0.0, 1.0, 0.0),
         (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), "X (mm)", "Z (mm)"),
        (f"profile Y-Z @ X={cx:.0f}", (cx, cy, 0.0), (1.0, 0.0, 0.0),
         (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), "Y (mm)", "Z (mm)"),
    ]
    for z in _slab_levels(mesh):
        specs.append(
            (f"plan X-Y @ Z={z:.0f}", (cx, cy, z), (0.0, 0.0, 1.0),
             (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), "X (mm)", "Y (mm)")
        )
    return specs


def render_sections(
    stl_path: Path, out_png: Path, target=None, metrics=None
) -> Path | None:
    """Render feature-aligned cross-sections into `out_png`.

    Two vertical profile cuts plus a horizontal plan cut inside each geometric layer (the
    layers are read off the mesh's horizontal faces). Sections expose hole depth/type
    (THRU vs blind vs counterbore), wall thickness, step/pad heights, and internal features
    the shaded exterior hides. `target`/`metrics` are accepted as optional hints; the planes
    are derived from the mesh so this works with or without a typed target. Returns None
    (writing nothing) if the mesh is empty or sectioning fails — sections are optional.
    """
    try:
        mesh = trimesh.load(stl_path, force="mesh")
    except Exception:
        return None
    if mesh is None or not hasattr(mesh, "faces") or len(mesh.faces) == 0:
        return None

    specs = _section_specs(mesh)
    ncols = min(3, len(specs))
    nrows = math.ceil(len(specs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    panels = axes.flat
    drew = False
    for ax, (title, origin, normal, u_axis, v_axis, xlabel, ylabel) in zip(panels, specs):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25)
        try:
            section = mesh.section(
                plane_origin=np.array(origin), plane_normal=np.array(normal)
            )
            if section is None or len(section.entities) == 0:
                continue
            # Project each section polyline onto the plane's in-axis basis (avoids the
            # to_planar()/networkx path); plot crisp contour lines incl. inner hole loops.
            basis = np.column_stack([np.array(u_axis), np.array(v_axis)])
            verts = section.vertices
            for entity in section.entities:
                uv = verts[entity.points] @ basis
                ax.plot(uv[:, 0], uv[:, 1], color="#1b2433", lw=1.5)
            drew = True
        except Exception:
            continue
    for ax in panels:  # blank any unused grid cells
        ax.set_axis_off()
    if not drew:
        plt.close(fig)
        return None

    fig.suptitle(
        "feature-aligned cross-sections — profiles + per-layer plan outlines "
        "(hole depth/type, wall thickness, step/pad heights)",
        fontsize=13,
    )
    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png
