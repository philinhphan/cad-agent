"""Reprojection check: prove a generated STEP matches a drawing, deterministically.

A VLM (`locator`) proposes where each orthographic view sits; the deterministic geometric
scorer (`check`, run as a subprocess by `adapter`) reprojects the STEP into those view
boxes and scores the overlap. A wrong box yields a low score — never a false pass. See
README.md for the standalone tool and DECISIONS.md for the design rationale.
"""

from cad_gen.reproject.adapter import ReprojectorFn, reproject_report
from cad_gen.reproject.locator import (
    ViewBox,
    ViewLayout,
    build_view_locator_agent,
    locate_drawing_views,
)

__all__ = [
    "ReprojectorFn",
    "ViewBox",
    "ViewLayout",
    "build_view_locator_agent",
    "locate_drawing_views",
    "reproject_report",
]
