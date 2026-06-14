"""Lightweight OpenCV primitive extraction for technical drawings.

The goal is not full drawing understanding. It produces a structured, deterministic hint
surface: line/circle/centerline/arrow/text-region candidates that can be persisted and
fed to agents alongside the original image and reprojection metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel


class DrawingPrimitive(BaseModel):
    kind: Literal["line", "circle", "arc", "centerline", "dimension_arrow", "text_region"]
    bbox_norm: tuple[float, float, float, float]
    confidence: float
    data: dict = {}


class DrawingPrimitiveSet(BaseModel):
    primitives: list[DrawingPrimitive]
    counts: dict[str, int]
    digest: str
    image_size: tuple[int, int] | None = None  # width, height in pixels


_PROMPT_KIND_ORDER = {
    "circle": 0,
    "arc": 1,
    "centerline": 2,
    "line": 3,
    "dimension_arrow": 4,
    "text_region": 5,
}


def extract_drawing_primitives(path: str | Path) -> DrawingPrimitiveSet:
    """Extract simple drawing primitives from a PNG/JPEG using OpenCV only."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"could not read image {path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    primitives: list[DrawingPrimitive] = []

    primitives.extend(_line_primitives(cv2, dark))
    primitives.extend(_circle_primitives(cv2, gray))
    primitives.extend(_arc_primitives(cv2, dark))
    primitives.extend(_centerline_primitives(cv2, img))
    primitives.extend(_arrow_primitives(cv2, dark))
    primitives.extend(_text_region_primitives(cv2, dark))
    _assign_text_regions(primitives)

    counts: dict[str, int] = {}
    for p in primitives:
        counts[p.kind] = counts.get(p.kind, 0) + 1
    digest = "Primitive extraction: " + ", ".join(
        f"{kind}={count}" for kind, count in sorted(counts.items())
    )
    if not counts:
        digest = "Primitive extraction: no primitives detected"
    H, W = gray.shape
    return DrawingPrimitiveSet(
        primitives=primitives, counts=counts, digest=digest, image_size=(W, H)
    )


def primitive_prompt_summary(
    drawing_name: str,
    primitive_set: DrawingPrimitiveSet,
    *,
    image_size: tuple[int, int] | None = None,
    max_items: int = 18,
) -> str:
    """Return a compact coordinate-rich primitive block for the generator prompt.

    Coordinates are normalized to the drawing image (origin top-left). Pixel-derived
    radius/endpoints are included when the extractor has them; otherwise bbox-derived
    values are used. These are hints, not dimensions.
    """
    if not primitive_set.primitives:
        return f"{drawing_name}: {primitive_set.digest}"
    image_size = image_size or primitive_set.image_size

    ordered = sorted(
        enumerate(primitive_set.primitives),
        key=lambda item: (
            _PROMPT_KIND_ORDER.get(item[1].kind, 99),
            -item[1].confidence,
            item[0],
        ),
    )
    lines = [
        f"{drawing_name} primitive hints ({primitive_set.digest}):",
        "Use these as feature-location hints only; read dimensions from the original drawing callouts.",
    ]
    for idx, primitive in ordered[:max_items]:
        lines.append(f"- {_primitive_line(idx, primitive, image_size)}")
    if len(ordered) > max_items:
        lines.append(f"- ... {len(ordered) - max_items} lower-confidence primitives omitted")
    return "\n".join(lines)


def _line_primitives(cv2, mask) -> list[DrawingPrimitive]:
    lines = cv2.HoughLinesP(
        mask, rho=1, theta=3.14159 / 180, threshold=35, minLineLength=25, maxLineGap=5
    )
    if lines is None:
        return []
    H, W = mask.shape
    out = []
    for x1, y1, x2, y2 in lines[:, 0][:80]:
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length < 20:
            continue
        out.append(
            DrawingPrimitive(
                kind="line",
                bbox_norm=_bbox_norm((min(x1, x2), min(y1, y2), abs(x2 - x1) + 1, abs(y2 - y1) + 1), W, H),
                confidence=min(1.0, length / max(W, H)),
                data={"p1": [int(x1), int(y1)], "p2": [int(x2), int(y2)]},
            )
        )
    return out


def _circle_primitives(cv2, gray) -> list[DrawingPrimitive]:
    import numpy as np

    H, W = gray.shape
    blurred = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=80,
        param2=18,
        minRadius=5,
        maxRadius=max(8, min(H, W) // 3),
    )
    if circles is None:
        return []
    out = []
    for x, y, r in np.round(circles[0, :30]).astype(int):
        out.append(
            DrawingPrimitive(
                kind="circle",
                bbox_norm=_bbox_norm((x - r, y - r, 2 * r, 2 * r), W, H),
                confidence=0.8,
                data={"center_px": [int(x), int(y)], "radius_px": int(r)},
            )
        )
    return out


def _centerline_primitives(cv2, img) -> list[DrawingPrimitive]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Green/blue drafting centerlines are common in the challenge examples; this also
    # catches synthetic dashed green centerlines used by tests.
    colored = cv2.inRange(hsv, (35, 35, 20), (130, 255, 245))
    lines = cv2.HoughLinesP(
        colored, rho=1, theta=3.14159 / 180, threshold=8, minLineLength=8, maxLineGap=18
    )
    if lines is None:
        return []
    H, W = colored.shape
    out = []
    for x1, y1, x2, y2 in lines[:, 0][:30]:
        out.append(
            DrawingPrimitive(
                kind="centerline",
                bbox_norm=_bbox_norm((min(x1, x2), min(y1, y2), abs(x2 - x1) + 1, abs(y2 - y1) + 1), W, H),
                confidence=0.7,
                data={"p1": [int(x1), int(y1)], "p2": [int(x2), int(y2)]},
            )
        )
    return out


def _arc_primitives(cv2, mask) -> list[DrawingPrimitive]:
    import math

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    H, W = mask.shape
    out = []
    for c in contours:
        if len(c) < 20:
            continue
        peri = cv2.arcLength(c, False)
        if peri < 30:
            continue
        (cx, cy), radius = cv2.minEnclosingCircle(c)
        if radius < 8:
            continue
        coverage = peri / max(2 * math.pi * radius, 1.0)
        x, y, w, h = cv2.boundingRect(c)
        if 0.12 <= coverage <= 0.9 and min(w, h) >= 6:
            out.append(
                DrawingPrimitive(
                    kind="arc",
                    bbox_norm=_bbox_norm((x, y, w, h), W, H),
                    confidence=round(min(0.85, coverage), 3),
                    data={
                        "center_px": [round(float(cx), 1), round(float(cy), 1)],
                        "radius_px": round(float(radius), 1),
                        "coverage_turns": round(float(coverage), 3),
                    },
                )
            )
    return out[:40]


def _arrow_primitives(cv2, mask) -> list[DrawingPrimitive]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = mask.shape
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > 1500:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)
        if len(approx) == 3:
            x, y, w, h = cv2.boundingRect(c)
            out.append(
                DrawingPrimitive(
                    kind="dimension_arrow",
                    bbox_norm=_bbox_norm((x, y, w, h), W, H),
                    confidence=0.75,
                    data={"area_px": float(area)},
                )
            )
    return out


def _text_region_primitives(cv2, mask) -> list[DrawingPrimitive]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    H, W = mask.shape
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 20 or area > 900:
            continue
        if h < 5 or w < 3 or h > 60 or w > 120:
            continue
        # Exclude long drafting strokes; text glyphs are compact components.
        if max(w / max(h, 1), h / max(w, 1)) > 8:
            continue
        out.append(
            DrawingPrimitive(
                kind="text_region",
                bbox_norm=_bbox_norm((int(x), int(y), int(w), int(h)), W, H),
                confidence=0.45,
                data={"area_px": int(area), "text": None},
            )
        )
    return out[:80]


def _assign_text_regions(primitives: list[DrawingPrimitive]) -> None:
    anchors = [
        (idx, p)
        for idx, p in enumerate(primitives)
        if p.kind not in {"text_region", "dimension_arrow"}
    ]
    if not anchors:
        return
    for p in primitives:
        if p.kind != "text_region":
            continue
        pcx, pcy = _center(p.bbox_norm)
        best_idx, best, best_dist = -1, None, float("inf")
        for idx, candidate in anchors:
            ccx, ccy = _center(candidate.bbox_norm)
            dist = ((pcx - ccx) ** 2 + (pcy - ccy) ** 2) ** 0.5
            if dist < best_dist:
                best_idx, best, best_dist = idx, candidate, dist
        if best is not None:
            p.data = {
                **p.data,
                "nearest_primitive": {
                    "index": best_idx,
                    "kind": best.kind,
                    "distance_norm": round(best_dist, 4),
                },
            }


def _primitive_line(
    index: int, primitive: DrawingPrimitive, image_size: tuple[int, int] | None
) -> str:
    kind = primitive.kind
    base = f"{kind}#{index} conf={primitive.confidence:.2f} bbox={_fmt_box(primitive.bbox_norm)}"
    if kind == "circle":
        return base + " " + _circle_hint(primitive, image_size)
    if kind == "arc":
        coverage = primitive.data.get("coverage_turns")
        suffix = _circle_hint(primitive, image_size)
        if coverage is not None:
            suffix += f" coverage={coverage}"
        return base + " " + suffix
    if kind in {"line", "centerline"}:
        return base + " " + _line_hint(primitive, image_size)
    if kind == "text_region":
        nearest = primitive.data.get("nearest_primitive") or {}
        if nearest:
            return (
                base
                + f" nearest={nearest.get('kind')}#{nearest.get('index')}"
                + f" distance_norm={nearest.get('distance_norm')}"
            )
    return base


def _circle_hint(primitive: DrawingPrimitive, image_size: tuple[int, int] | None) -> str:
    center = _norm_point(primitive.data.get("center_px"), image_size)
    radius = _norm_radius(primitive.data.get("radius_px"), image_size, primitive.bbox_norm)
    return f"center_norm={center} radius_norm~{radius:.3f}"


def _line_hint(primitive: DrawingPrimitive, image_size: tuple[int, int] | None) -> str:
    p1 = _norm_point(primitive.data.get("p1"), image_size)
    p2 = _norm_point(primitive.data.get("p2"), image_size)
    return f"p1_norm={p1} p2_norm={p2}"


def _norm_point(point: object, image_size: tuple[int, int] | None) -> str:
    if (
        image_size is not None
        and isinstance(point, list)
        and len(point) >= 2
        and image_size[0] > 0
        and image_size[1] > 0
    ):
        return f"({float(point[0]) / image_size[0]:.3f},{float(point[1]) / image_size[1]:.3f})"
    return "(?,?)"


def _norm_radius(
    radius_px: object,
    image_size: tuple[int, int] | None,
    bbox_norm: tuple[float, float, float, float],
) -> float:
    if image_size is not None and isinstance(radius_px, (int, float)) and image_size[0] > 0:
        return float(radius_px) / image_size[0]
    return max(bbox_norm[2], bbox_norm[3]) / 2


def _fmt_box(box: tuple[float, float, float, float]) -> str:
    x, y, w, h = box
    return f"({x:.3f},{y:.3f},{w:.3f},{h:.3f})"


def _center(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return x + w / 2, y + h / 2


def _bbox_norm(box: tuple[int, int, int, int], width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = box
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return (
        round(x / width, 4),
        round(y / height, 4),
        round(w / width, 4),
        round(h / height, 4),
    )
