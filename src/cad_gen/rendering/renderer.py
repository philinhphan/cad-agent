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


def render_views(
    stl_path: Path, out_png: Path, metrics: GeometryMetrics | None = None
) -> Path:
    """Render iso/front/top/right shaded views of `stl_path` into `out_png`."""
    mesh = trimesh.load(stl_path, force="mesh")
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    half = float((bounds[1] - bounds[0]).max()) * 0.6 + 1e-9

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (title, elev, azim, xlabel, ylabel) in zip(axes.flat, _VIEWS):
        rgb, _, extent = _rasterize_view(mesh, elev, azim, center, half)
        ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")
        ax.set_title(title)
        if xlabel is None:
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)

    if metrics is not None:
        x, y, z = metrics.bbox_mm
        watertight = {True: "yes", False: "NO", None: "?"}[metrics.is_watertight]
        fig.suptitle(
            f"bbox {x:.1f} × {y:.1f} × {z:.1f} mm   ·   volume {metrics.volume_mm3:,.0f} mm³"
            f"   ·   solids {metrics.n_solids}   ·   watertight {watertight}",
            fontsize=13,
        )

    fig.tight_layout()
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png
