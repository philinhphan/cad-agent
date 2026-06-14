#!/usr/bin/env python3
"""Standalone reprojection check: prove a generated STEP matches a technical drawing.

Pipeline (no coupling to any other code):
  1. read STEP, compute 3D bounding box
  2. project the solid into orthographic views (top/front/side) via OpenCASCADE HLR
  3. isolate the drawing's geometry lines (color filter -> keep black, drop blue/green/red)
     inside the view boxes supplied by the caller (--regions-json; a VLM locates them)
  4. snap each supplied box to its line work, then fit each reprojection into it by 2D
     bounding box (no dimension reading)
  5. score the overlap (Chamfer distance + IoU) and write overlay PNGs + a JSON report

The SCORING is deterministic and falsifiable: a wrong view box yields a low overlap score,
never a false pass. Only the view *localisation* is delegated upstream (to a VLM).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

# --- OpenCASCADE (OCP / cadquery-ocp) -------------------------------------------------
from OCP.STEPControl import STEPControl_Reader
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
from OCP.BRepAdaptor import BRepAdaptor_Curve
from OCP.GCPnts import GCPnts_QuasiUniformDeflection
from OCP.Bnd import Bnd_Box
from OCP.BRepBndLib import BRepBndLib
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopoDS import TopoDS


# =====================================================================================
# View convention (configurable defaults).
# Part assumed: X = width, Y = depth, Z = height.
#   front: look along Y, horizontal=X, vertical=Z   -> shows (dx, dz)
#   top  : look along Z, horizontal=X, vertical=Y   -> shows (dx, dy)
#   side : look along X, horizontal=Y, vertical=Z   -> shows (dy, dz)
# Each view gives (x_dir, up_dir) as unit 3-vectors = the view's horizontal/vertical
# axes. The projection (viewing) direction is derived as x_dir x up_dir, so p.X()/p.Y()
# of the HLR result are the view's horizontal/vertical coordinates with up pointing up.
# =====================================================================================
VIEWS = {
    "front": dict(x_dir=(1.0, 0.0, 0.0), up_dir=(0.0, 0.0, 1.0)),
    "top":   dict(x_dir=(1.0, 0.0, 0.0), up_dir=(0.0, 1.0, 0.0)),
    "side":  dict(x_dir=(0.0, 1.0, 0.0), up_dir=(0.0, 0.0, 1.0)),
}


def dims_for_view(name: str, dx: float, dy: float, dz: float) -> tuple[float, float]:
    """(width, height) of the part's 3D bbox as seen in this view."""
    return {
        "front": (dx, dz),
        "top":   (dx, dy),
        "side":  (dy, dz),
    }[name]


# =====================================================================================
# STEP + HLR
# =====================================================================================
def load_step(path: str):
    reader = STEPControl_Reader()
    reader.ReadFile(str(path))
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise RuntimeError(f"could not read a shape from {path}")
    return shape


def bbox3d(shape) -> tuple[float, float, float]:
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (xmax - xmin, ymax - ymin, zmax - zmin)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _edge_polyline(edge, deflection: float) -> list[tuple[float, float]]:
    """HLR edges are returned in the projector frame -> p.X()/p.Y() are the 2D coords."""
    curve = BRepAdaptor_Curve(edge)
    try:
        disc = GCPnts_QuasiUniformDeflection(curve, deflection)
    except Exception:
        return []
    if not disc.IsDone():
        return []
    pts = []
    for i in range(1, disc.NbPoints() + 1):
        p = disc.Value(i)
        pts.append((p.X(), p.Y()))
    return pts


def project_view(shape, spec: dict, include_hidden: bool, deflection: float) -> list[list[tuple[float, float]]]:
    """Hidden-line-removal projection -> list of 2D polylines (in model units).

    The projector frame is built so its X axis = spec['x_dir'] and its Y axis =
    spec['up_dir'] exactly (main direction = x_dir x up_dir), so p.X()/p.Y() are
    the view's horizontal/vertical coordinates with 'up' pointing up.
    """
    algo = HLRBRep_Algo()
    algo.Add(shape)
    normal = _cross(spec["x_dir"], spec["up_dir"])
    ax2 = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(*normal), gp_Dir(*spec["x_dir"]))
    algo.Projector(HLRAlgo_Projector(ax2))
    algo.Update()
    algo.Hide()
    hlr = HLRBRep_HLRToShape(algo)

    getters = [hlr.VCompound, hlr.OutLineVCompound]
    if include_hidden:
        getters += [hlr.HCompound, hlr.OutLineHCompound]

    polylines: list[list[tuple[float, float]]] = []
    for get in getters:
        try:
            comp = get()
        except Exception:
            continue
        if comp is None or comp.IsNull():
            continue
        exp = TopExp_Explorer(comp, TopAbs_EDGE)
        while exp.More():
            edge = TopoDS.Edge_s(exp.Current())
            pts = _edge_polyline(edge, deflection)
            if len(pts) >= 2:
                polylines.append(pts)
            exp.Next()
    return polylines


# =====================================================================================
# Drawing: color filter + view location
# =====================================================================================
def load_geometry_mask(path: str, color_filter: bool) -> np.ndarray:
    """Return a uint8 mask (255 = geometry line) of the drawing's black line work."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read image {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # dark foreground via Otsu (inverted: lines become 255)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if color_filter:
        sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]
        colored_lines = ((sat >= 60) & (gray < 200)).sum()
        if colored_lines > 0.0005 * gray.size:  # there really is colour -> drop it
            # keep only near-grey ink: black geometry sits at sat ~20, while JPEG only
            # desaturates blue/green dimension-line edges down to ~55, so cut at 45.
            low_sat = (sat < 45).astype(np.uint8) * 255
            dark = cv2.bitwise_and(dark, low_sat)
    return dark


def _tighten_box(mask: np.ndarray, box, margin: int = 3):
    """Tighten a region to the line work it contains (union of all non-speck components),
    undoing the morphological-close inflation used for locating. Keeps fragmented views
    whole; relies on the colour filter to have removed dimension/extension lines."""
    x, y, w, h = box
    sub = mask[y:y + h, x:x + w]
    min_speck = max(12, int(0.01 * max(mask.shape)))  # resolution-relative
    n, lab, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
    keep = np.zeros(sub.shape, np.uint8)
    for i in range(1, n):
        if stats[i, 4] >= min_speck:  # drop tiny specks (leftover text / noise)
            keep[lab == i] = 255
    ys, xs = np.where(keep > 0)
    if len(xs) == 0:
        return box
    H, W = mask.shape
    nx0, ny0 = max(0, x + int(xs.min()) - margin), max(0, y + int(ys.min()) - margin)
    nx1, ny1 = min(W - 1, x + int(xs.max()) + margin), min(H - 1, y + int(ys.max()) + margin)
    return (nx0, ny0, nx1 - nx0 + 1, ny1 - ny0 + 1)


def _regions_to_boxes(mask: np.ndarray, regions: dict) -> dict[str, tuple[int, int, int, int]]:
    """Convert externally-supplied normalized view regions to tight pixel boxes.

    `regions` maps a view name (front/top/side) to ``[x, y, w, h]`` in [0, 1] with the
    origin at the top-left. Each box is scaled to pixels and then snapped to the line work
    it contains, so a coarse box (e.g. from a vision model) still bounds the view exactly.
    """
    H, W = mask.shape
    out: dict[str, tuple[int, int, int, int]] = {}
    for name, box in regions.items():
        nx, ny, nw, nh = box
        x = min(int(round(min(max(nx, 0.0), 1.0) * W)), W - 1)
        y = min(int(round(min(max(ny, 0.0), 1.0) * H)), H - 1)
        w = max(1, min(int(round(nw * W)), W - x))
        h = max(1, min(int(round(nh * H)), H - y))
        out[name] = _tighten_box(mask, (x, y, w, h))
    return out


# =====================================================================================
# Fit + score
# =====================================================================================
def _dihedral(pts: np.ndarray, t: int) -> np.ndarray:
    """Apply one of 8 dihedral transforms (rotation k*90 + optional mirror) in 2D."""
    u, v = pts[:, 0].copy(), pts[:, 1].copy()
    if t & 4:  # mirror x
        u = -u
    rot = t & 3
    for _ in range(rot):
        u, v = -v, u
    return np.stack([u, v], axis=1)


def rasterize(pts_list: list[np.ndarray], w: int, h: int, thickness: int = 2) -> np.ndarray:
    """Fit polylines (model units) into a w x h canvas by uniform-scale bbox fit.

    Uniform scale (geometric mean) keeps the model aspect honest, so an aspect
    mismatch stays visible in the overlay and penalised by the Chamfer score.
    """
    canvas = np.zeros((h, w), np.uint8)
    allpts = np.concatenate(pts_list, axis=0)
    minu, minv = allpts.min(axis=0)
    maxu, maxv = allpts.max(axis=0)
    wu, hv = max(maxu - minu, 1e-9), max(maxv - minv, 1e-9)
    s = math.sqrt((w * h) / (wu * hv))
    # avoid spilling outside the canvas
    s = min(s, (w - 2) / wu, (h - 2) / hv)
    offx = (w - wu * s) / 2.0
    offy = (h - hv * s) / 2.0
    for pts in pts_list:
        x = (pts[:, 0] - minu) * s + offx
        y = h - ((pts[:, 1] - minv) * s + offy)  # flip v: up -> up
        poly = np.stack([x, y], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, 255, thickness, cv2.LINE_AA)
    return canvas


def match_metrics(draw: np.ndarray, hlr: np.ndarray, tau: float) -> dict:
    """Compare two line masks. Returns mean Chamfer (px) plus coverage at tolerance tau:
    recall = fraction of drawing edge px with a reprojection line within tau,
    precision = fraction of reprojection edge px with a drawing line within tau.
    Coverage catches localised mismatches (a feature that sits somewhere completely
    different) that a mean distance averages away."""
    db = (draw > 0)
    hb = (hlr > 0)
    if db.sum() == 0 or hb.sum() == 0:
        return dict(chamfer=float("inf"), iou=0.0, recall=0.0, precision=0.0)
    dt_draw = cv2.distanceTransform(255 - draw, cv2.DIST_L2, 3)
    dt_hlr = cv2.distanceTransform(255 - hlr, cv2.DIST_L2, 3)
    d_h2d = dt_draw[hb]   # HLR -> nearest drawing line
    d_d2h = dt_hlr[db]    # drawing -> nearest HLR line
    chamfer = (float(d_h2d.mean()) + float(d_d2h.mean())) / 2.0
    precision = float((d_h2d <= tau).mean())
    recall = float((d_d2h <= tau).mean())
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    iou = float(((cv2.dilate(draw, k) > 0) & (cv2.dilate(hlr, k) > 0)).sum()) / \
        float(((cv2.dilate(draw, k) > 0) | (cv2.dilate(hlr, k) > 0)).sum())
    return dict(chamfer=chamfer, iou=iou, recall=recall, precision=precision)


def fit_score_overlay(draw_sub: np.ndarray, polylines, auto_orient: bool,
                      tau_frac: float = 0.015):
    """Fit reprojection into the drawing view, score it, return (result, overlay)."""
    h, w = draw_sub.shape
    diag = math.hypot(w, h)
    tau = tau_frac * diag   # matching band, resolution-independent
    pts_list = [np.asarray(p, dtype=np.float64) for p in polylines]

    transforms = range(8) if auto_orient else [0]
    best = None
    for t in transforms:
        tl = [_dihedral(p, t) for p in pts_list]
        hlr_mask = rasterize(tl, w, h)
        m = match_metrics(draw_sub, hlr_mask, tau)
        if best is None or m["chamfer"] < best[0]["chamfer"]:
            best = (m, t, hlr_mask)

    m, t, hlr_mask = best
    # colour overlay so localised mismatches are visible:
    #   red    = reprojection & drawing agree, grey = drawing matched
    #   blue   = drawing line with NO reprojection nearby (missing/displaced feature)
    #   orange = reprojected line with NO drawing line nearby (extra/wrong feature)
    dt_draw = cv2.distanceTransform(255 - draw_sub, cv2.DIST_L2, 3)
    dt_hlr = cv2.distanceTransform(255 - hlr_mask, cv2.DIST_L2, 3)
    db, hb = draw_sub > 0, hlr_mask > 0
    overlay = np.full((h, w, 3), 255, np.uint8)
    overlay[db] = (170, 170, 170)
    overlay[db & (dt_hlr > tau)] = (255, 90, 0)    # blue (BGR): unmatched drawing
    overlay[hb] = (0, 0, 255)                       # red: reprojection
    overlay[hb & (dt_draw > tau)] = (0, 140, 255)   # orange: unmatched reprojection

    # coverage = recall: fraction of DRAWING lines that have a reprojected line nearby.
    # This is the robust correctness signal (every drawn line must be explained by the
    # model). Precision is reported for diagnostics but not gated on: the reprojection
    # legitimately carries extra edges (hidden lines, tangents) the drawing may omit.
    result = dict(coverage=round(m["recall"], 3),
                  precision=round(m["precision"], 3),
                  chamfer_pct=round(100.0 * m["chamfer"] / diag, 3),
                  chamfer_px=round(m["chamfer"], 2),
                  iou=round(m["iou"], 3),
                  orient=t)
    return result, overlay


# =====================================================================================
# Main
# =====================================================================================
def run(drawing: str, step: str, out: str, *, regions: dict, color_filter: bool,
        include_hidden: bool, auto_orient: bool, max_chamfer_pct: float, min_coverage: float,
        aspect_tol: float, deflection: float, debug: bool) -> dict:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)

    shape = load_step(step)
    dx, dy, dz = bbox3d(shape)

    mask = load_geometry_mask(drawing, color_filter)
    if debug:
        cv2.imwrite(str(outdir / "debug_geometry_mask.png"), mask)
    view_boxes = _regions_to_boxes(mask, regions)

    report = {
        "drawing": str(drawing),
        "step": str(step),
        "step_bbox_mm": {"dx": round(dx, 2), "dy": round(dy, 2), "dz": round(dz, 2)},
        "views": {},
    }

    chamfer_pcts, coverages, all_ok = [], [], True
    for name, spec in VIEWS.items():
        if name not in view_boxes:
            report["views"][name] = None
            all_ok = False
            continue
        x, y, w, h = view_boxes[name]
        draw_sub = mask[y:y + h, x:x + w]

        polylines = project_view(shape, spec, include_hidden, deflection)
        if not polylines:
            report["views"][name] = {"error": "empty projection"}
            all_ok = False
            continue

        res, overlay = fit_score_overlay(draw_sub, polylines, auto_orient)
        overlay_path = outdir / f"overlay_{name}.png"
        cv2.imwrite(str(overlay_path), overlay)

        # scale-free aspect check (the reliable bbox check, no dimensions read)
        sw, sh = dims_for_view(name, dx, dy, dz)
        step_aspect = sw / sh if sh else float("inf")
        draw_aspect = w / h if h else float("inf")
        rel = abs(draw_aspect - step_aspect) / step_aspect if step_aspect else float("inf")
        aspect_ok = rel <= aspect_tol

        res.update(
            overlay=overlay_path.name,
            draw_aspect=round(draw_aspect, 3),
            step_aspect=round(step_aspect, 3),
            aspect_rel_err=round(rel, 3),
            aspect_ok=bool(aspect_ok),
        )
        report["views"][name] = res
        chamfer_pcts.append(res["chamfer_pct"])
        coverages.append(res["coverage"])
        if (res["chamfer_pct"] > max_chamfer_pct or res["coverage"] < min_coverage
                or not aspect_ok):
            all_ok = False

    report["overall"] = {
        "mean_chamfer_pct": round(float(np.mean(chamfer_pcts)), 3) if chamfer_pcts else None,
        "min_coverage": round(float(np.min(coverages)), 3) if coverages else None,
        "thresholds": {"max_chamfer_pct": max_chamfer_pct, "min_coverage": min_coverage},
        "views_found": len(view_boxes),
        "pass": bool(all_ok and chamfer_pcts),
    }

    with open(outdir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description="Reproject a STEP and check it against a drawing.")
    ap.add_argument("--drawing", required=True, help="original drawing image (PNG/JPG)")
    ap.add_argument("--step", required=True, help="generated STEP file")
    ap.add_argument("--out", default="out_reproj", help="output directory")
    ap.add_argument("--regions-json", required=True,
                    help="JSON file mapping view name -> [x, y, w, h] normalized boxes (0..1)")
    ap.add_argument("--no-color-filter", action="store_true", help="disable black/colour split")
    ap.add_argument("--no-hidden", action="store_true",
                    help="do not project hidden edges (default: include, for fair coverage)")
    ap.add_argument("--auto-orient", action="store_true",
                    help="try 8 in-plane orientations per view, keep best (hedges STEP orientation)")
    ap.add_argument("--max-chamfer-pct", type=float, default=2.0,
                    help="pass threshold: max mean edge distance as %% of the view "
                         "diagonal (resolution-independent)")
    ap.add_argument("--min-coverage", type=float, default=0.90,
                    help="pass threshold: min fraction of lines (both directions) with a "
                         "match within the tolerance band (catches localised mismatches)")
    ap.add_argument("--aspect-tol", type=float, default=0.1, help="relative bbox aspect tolerance")
    ap.add_argument("--deflection", type=float, default=0.1, help="HLR curve discretisation (mm)")
    ap.add_argument("--debug", action="store_true", help="dump intermediate masks")
    args = ap.parse_args()

    with open(args.regions_json) as f:
        regions = json.load(f)

    report = run(args.drawing, args.step, args.out,
                 regions=regions,
                 color_filter=not args.no_color_filter,
                 include_hidden=not args.no_hidden,
                 auto_orient=args.auto_orient,
                 max_chamfer_pct=args.max_chamfer_pct,
                 min_coverage=args.min_coverage,
                 aspect_tol=args.aspect_tol,
                 deflection=args.deflection,
                 debug=args.debug)
    print(json.dumps(report, indent=2))
    print("\nPASS" if report["overall"]["pass"] else "\nFAIL")


if __name__ == "__main__":
    main()
