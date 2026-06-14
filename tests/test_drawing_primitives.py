import pytest

from cad_gen.reproject.drawing_primitives import (
    DrawingPrimitive,
    DrawingPrimitiveSet,
    extract_drawing_primitives,
    primitive_prompt_summary,
)


def test_extracts_lines_circles_centerlines_arrows_and_text_regions(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    img = np.full((220, 260, 3), 255, np.uint8)
    cv2.line(img, (20, 180), (220, 180), (0, 0, 0), 2)
    cv2.line(img, (40, 60), (40, 180), (0, 0, 0), 2)
    cv2.circle(img, (150, 120), 28, (0, 0, 0), 2)
    cv2.ellipse(img, (205, 75), (22, 18), 0, 20, 210, (0, 0, 0), 2)
    for x in range(45, 210, 24):
        cv2.line(img, (x, 100), (x + 12, 100), (0, 180, 0), 1)  # dashed centerline
    cv2.fillPoly(img, [np.array([[70, 45], [88, 39], [88, 51]])], (0, 0, 0))
    cv2.rectangle(img, (105, 32), (114, 45), (0, 0, 0), -1)
    cv2.rectangle(img, (119, 32), (128, 45), (0, 0, 0), -1)
    path = tmp_path / "drawing.png"
    cv2.imwrite(str(path), img)

    primitives = extract_drawing_primitives(path)

    kinds = {p.kind for p in primitives.primitives}
    assert "line" in kinds
    assert "circle" in kinds
    assert "arc" in kinds
    assert "centerline" in kinds
    assert "dimension_arrow" in kinds
    assert "text_region" in kinds
    text_regions = [p for p in primitives.primitives if p.kind == "text_region"]
    assert any("nearest_primitive" in p.data for p in text_regions)
    assert primitives.counts["line"] >= 2
    assert "primitive extraction" in primitives.digest.lower()


def test_prompt_summary_exposes_coordinates_and_text_links_to_agent():
    primitives = DrawingPrimitiveSet(
        counts={"circle": 1, "line": 1, "arc": 1, "text_region": 1},
        digest="Primitive extraction: arc=1, circle=1, line=1, text_region=1",
        primitives=[
            DrawingPrimitive(
                kind="circle",
                bbox_norm=(0.40, 0.30, 0.10, 0.12),
                confidence=0.82,
                data={"center_px": [150, 120], "radius_px": 28},
            ),
            DrawingPrimitive(
                kind="line",
                bbox_norm=(0.10, 0.80, 0.70, 0.01),
                confidence=0.9,
                data={"p1": [20, 180], "p2": [220, 180]},
            ),
            DrawingPrimitive(
                kind="arc",
                bbox_norm=(0.70, 0.20, 0.15, 0.18),
                confidence=0.6,
                data={"center_px": [205, 75], "radius_px": 22, "coverage_turns": 0.42},
            ),
            DrawingPrimitive(
                kind="text_region",
                bbox_norm=(0.37, 0.24, 0.08, 0.04),
                confidence=0.45,
                data={"nearest_primitive": {"index": 0, "kind": "circle", "distance_norm": 0.07}},
            ),
        ],
    )

    summary = primitive_prompt_summary("drawing_01.png", primitives, image_size=(260, 220))

    assert "drawing_01.png primitive hints" in summary
    assert "circle#0" in summary
    assert "center_norm=(0.577,0.545)" in summary
    assert "radius_norm~0.108" in summary
    assert "line#1" in summary and "p1_norm=(0.077,0.818)" in summary
    assert "arc#2" in summary and "coverage=0.42" in summary
    assert "text_region#3" in summary and "nearest=circle#0" in summary
    assert "Use these as feature-location hints" in summary
