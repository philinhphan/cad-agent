"""Render an STL into a 2x2 multi-view composite PNG for the vision critic.

matplotlib Agg + trimesh: headless-safe everywhere (no OpenGL context).
Known limit: painter's-algorithm depth artifacts on overlapping bodies —
acceptable for shape-level critique.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from cad_gen.models import GeometryMetrics

# (title, elev, azim)
_VIEWS = [
    ("isometric", 30, -55),
    ("front (X-Z)", 0, -90),
    ("top (X-Y)", 90, -90),
    ("right (Y-Z)", 0, 0),
]

_BASE_COLOR = np.array([0.36, 0.56, 0.80])
_LIGHT_DIR = np.array([0.4, -0.6, 0.8]) / np.linalg.norm([0.4, -0.6, 0.8])


def render_views(
    stl_path: Path, out_png: Path, metrics: GeometryMetrics | None = None
) -> Path:
    """Render iso/front/top/right shaded views of `stl_path` into `out_png`."""
    mesh = trimesh.load(stl_path, force="mesh")
    triangles = mesh.triangles
    shade = 0.35 + 0.65 * np.clip(mesh.face_normals @ _LIGHT_DIR, 0.0, 1.0)
    face_colors = np.clip(shade[:, None] * _BASE_COLOR[None, :], 0.0, 1.0)

    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    half = float((bounds[1] - bounds[0]).max()) * 0.6 + 1e-9

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), subplot_kw={"projection": "3d"})
    for ax, (title, elev, azim) in zip(axes.flat, _VIEWS):
        ax.add_collection3d(
            Poly3DCollection(
                triangles,
                facecolors=face_colors,
                edgecolors=(0.0, 0.0, 0.0, 0.08),
                linewidths=0.2,
            )
        )
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")

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
