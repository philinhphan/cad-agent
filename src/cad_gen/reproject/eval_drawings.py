#!/usr/bin/env python3
"""Drawing-side-only check of the NEW reproject pipeline (no STEP, no HLR, no overlap).

For each drawing it runs only the "drawing -> views" half:
  1. VLM locate_drawing_views -> normalized front/top/side boxes (the new approach;
     the old CV locate_views heuristic was removed)
  2. load_geometry_mask       -> the edge image ("Kantenbild")
  3. _regions_to_boxes        -> snap the VLM boxes onto the actual line work

Output per drawing (out_reproj_check/<name>/):
  geometry_mask.png  -- the edge image (drawing -> lines)
  vlm_overlay.png    -- the RAW VLM boxes on the original
  assigned.png       -- the SNAPPED (tightened) boxes on the edge image
Plus, at the root: composite_<n>.png (original+VLM | edges | snapped) and summary.json.

Run from the repo root:
  uv run python -m cad_gen.reproject.eval_drawings   (uses CAD_GEN_VIEW_MODEL / GEMINI key)
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from cad_gen.imaging import media_type_for
from cad_gen.models import DrawingAttachment, RunConfig
from cad_gen.reproject.check import _regions_to_boxes, load_geometry_mask
from cad_gen.reproject.locator import build_view_locator_agent, locate_drawing_views

DRAWINGS = Path("exampledrawings")
OUTROOT = Path("out_reproj_check")
COLORS = {"front": (0, 0, 255), "top": (0, 190, 0), "side": (255, 90, 0)}  # BGR
PANEL_H = 700


def _draw(img_bgr: np.ndarray, boxes_px: dict, dash: bool = False) -> np.ndarray:
    out = img_bgr.copy()
    for name, (x, y, w, h) in boxes_px.items():
        c = COLORS.get(name, (0, 255, 255))
        cv2.rectangle(out, (x, y), (x + w, y + h), c, 3)
        cv2.putText(out, name, (x + 2, max(18, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, c, 3)
    return out


def _labeled(img: np.ndarray, text: str) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    img = cv2.resize(img, (int(w * PANEL_H / h), PANEL_H), interpolation=cv2.INTER_AREA)
    bar = np.full((36, img.shape[1], 3), 30, np.uint8)
    cv2.putText(bar, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return np.vstack([bar, img])


def _hstack(panels: list[np.ndarray]) -> np.ndarray:
    maxh = max(p.shape[0] for p in panels)
    panels = [np.vstack([p, np.full((maxh - p.shape[0], p.shape[1], 3), 30, np.uint8)])
              for p in panels]
    sep = np.full((maxh, 6, 3), 200, np.uint8)
    comp = panels[0]
    for p in panels[1:]:
        comp = np.hstack([comp, sep, p])
    return comp


async def main() -> None:
    load_dotenv(Path.cwd() / ".env")
    OUTROOT.mkdir(parents=True, exist_ok=True)
    agent = build_view_locator_agent(RunConfig().view_model)

    summary = []
    for img_path in sorted(DRAWINGS.glob("*.jpeg")):
        name = img_path.stem
        outdir = OUTROOT / name
        outdir.mkdir(parents=True, exist_ok=True)
        data = img_path.read_bytes()
        drawing = DrawingAttachment(
            filename=img_path.name, media_type=media_type_for(img_path.name, data), data=data
        )

        try:
            layout = await locate_drawing_views(agent, drawing=drawing)
            regions = {v.label: [v.x, v.y, v.w, v.h] for v in layout.views}
        except Exception as exc:  # noqa: BLE001 - log + keep going
            regions = {}
            print(f"{img_path.name:50s} -> VLM ERROR: {type(exc).__name__}: {exc}")

        orig = cv2.imread(str(img_path))
        H, W = orig.shape[:2]
        mask = load_geometry_mask(str(img_path), color_filter=True)
        cv2.imwrite(str(outdir / "geometry_mask.png"), mask)

        raw_px = {n: (int(x * W), int(y * H), int(w * W), int(h * H))
                  for n, (x, y, w, h) in regions.items()}
        snapped = _regions_to_boxes(mask, regions) if regions else {}

        cv2.imwrite(str(outdir / "vlm_overlay.png"), _draw(orig, raw_px))
        cv2.imwrite(str(outdir / "assigned.png"),
                    _draw(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), snapped))

        comp = _hstack([
            _labeled(_draw(orig, raw_px), "ORIGINAL + VLM boxes"),
            _labeled(mask, "EDGE IMAGE"),
            _labeled(_draw(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), snapped), "SNAPPED views"),
        ])
        cv2.imwrite(str(OUTROOT / f"composite_{name[-3:]}.png"), comp)

        summary.append({
            "drawing": img_path.name,
            "views_found": sorted(regions.keys()),
            "regions_norm": {n: [round(c, 3) for c in b] for n, b in regions.items()},
            "snapped_px": {n: list(map(int, b)) for n, b in snapped.items()},
        })
        print(f"{img_path.name:50s} -> {sorted(regions.keys())}")

    (OUTROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUTROOT}/summary.json + composite_*.png")


if __name__ == "__main__":
    asyncio.run(main())
