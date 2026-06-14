"""Unit tests for the reproject adapter: report -> LLM artifacts, and the subprocess
wrapper's failure handling. The deterministic reproject_check itself (OCCT/cv2) is not
exercised here — the subprocess is mocked — so these stay fast and hermetic."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cad_gen.models import ReprojectionView, RunConfig
from cad_gen.reproject import adapter as reproject_adapter


def make_view(
    coverage: float,
    *,
    chamfer_pct: float = 1.0,
    aspect_ok: bool = True,
    draw_aspect: float = 1.5,
    step_aspect: float = 1.5,
    aspect_rel_err: float = 0.0,
    overlay: str = "overlay.png",
) -> dict:
    """A located-view entry shaped like reproject_check's report.json output."""
    return {
        "coverage": coverage,
        "precision": 0.91,
        "chamfer_pct": chamfer_pct,
        "chamfer_px": 2.3,
        "iou": 0.77,
        "orient": 0,
        "overlay": overlay,
        "draw_aspect": draw_aspect,
        "step_aspect": step_aspect,
        "aspect_rel_err": aspect_rel_err,
        "aspect_ok": aspect_ok,
    }


def make_report(views: dict, *, views_found: int | None = None, passed: bool = False) -> dict:
    found = (
        views_found
        if views_found is not None
        else sum(1 for v in views.values() if isinstance(v, dict) and "coverage" in v)
    )
    return {
        "step_bbox_mm": {"dx": 10, "dy": 10, "dz": 10},
        "views": views,
        "overall": {
            "mean_chamfer_pct": 1.2,
            "min_coverage": 0.0,
            "thresholds": {},
            "views_found": found,
            "pass": passed,
        },
    }


def build(report: dict, tmp_path: Path, config: RunConfig | None = None):
    return reproject_adapter._build_from_report(report, tmp_path, config or RunConfig())


REGIONS = {"front": [0.1, 0.6, 0.3, 0.3], "top": [0.1, 0.1, 0.3, 0.3]}


# --------------------------------------------------------------------------- #
# Digest construction
# --------------------------------------------------------------------------- #
def test_digest_verbalises_coverage_and_drops_diagnostics(tmp_path):
    report = make_report({"front": make_view(0.78), "top": make_view(0.97), "side": None})
    rep = build(report, tmp_path)

    assert rep.evaluated is True
    assert rep.views_found == 2
    assert "front view" in rep.digest and "78%" in rep.digest
    assert "fully matched" in rep.digest  # top at 97%
    assert reproject_adapter._ADVISORY in rep.digest
    # diagnostic-only fields never leak into the LLM-facing digest
    for noise in ("precision", "iou", "chamfer_px", "orient"):
        assert noise not in rep.digest


def test_localised_interpretation_points_at_weakest_view(tmp_path):
    report = make_report({"front": make_view(0.70), "top": make_view(0.98)})
    rep = build(report, tmp_path)

    assert "front view is the weakest" in rep.interpretation
    assert rep.interpretation in rep.digest


def test_single_view_is_a_local_hint_not_withheld(tmp_path):
    report = make_report({"front": make_view(0.40), "top": None, "side": None})
    rep = build(report, tmp_path)

    # one low view alone is NOT treated as a global orientation miss
    assert rep.evaluated is True
    assert "Only the front view" in rep.interpretation


def test_aspect_sign_verbalised(tmp_path):
    report = make_report(
        {
            "front": make_view(0.97, aspect_ok=False, draw_aspect=1.0, step_aspect=2.0,
                               aspect_rel_err=0.5),  # part wider than drawing
            "top": make_view(0.97, aspect_ok=False, draw_aspect=2.0, step_aspect=1.0,
                             aspect_rel_err=0.5),  # part taller than drawing
        }
    )
    rep = build(report, tmp_path)

    assert "too wide for its height" in rep.digest
    assert "too tall for its width" in rep.digest


# --------------------------------------------------------------------------- #
# "Nicht bestrafen" — withholding heuristics
# --------------------------------------------------------------------------- #
def test_no_views_found_is_not_evaluated(tmp_path):
    report = make_report({"front": None, "top": None, "side": None}, views_found=0)
    rep = build(report, tmp_path)

    assert rep.evaluated is False
    assert rep.digest == ""
    assert rep.skipped_reason


def test_global_low_coverage_withheld_only_when_aspects_match(tmp_path):
    # Low coverage everywhere BUT proportions still match (make_view default aspect_ok=True)
    # = a correct-but-rotated part -> withheld as orientation.
    report = make_report({"front": make_view(0.42), "top": make_view(0.40)})
    rep = build(report, tmp_path)

    assert rep.evaluated is False
    assert "orientation" in rep.skipped_reason


def test_global_low_with_bad_aspect_is_surfaced_as_wrong(tmp_path):
    # The runs/20260614_050528 case: a flat plate vs a tall bracket — low coverage AND
    # grossly wrong aspects. This must NOT be masked as orientation; it must surface so the
    # generator learns the shape is fundamentally wrong.
    report = make_report(
        {
            "front": make_view(0.06, aspect_ok=False, draw_aspect=2.0, step_aspect=12.0,
                               aspect_rel_err=5.0),
            "side": make_view(0.07, aspect_ok=False, draw_aspect=1.3, step_aspect=12.0,
                              aspect_rel_err=8.0),
        }
    )
    rep = build(report, tmp_path)

    assert rep.evaluated is True
    assert "missing" in rep.interpretation or "wrong" in rep.interpretation
    assert "rebuild" in rep.interpretation


def test_mixed_coverage_is_evaluated(tmp_path):
    report = make_report({"front": make_view(0.40), "top": make_view(0.95)})
    rep = build(report, tmp_path)

    assert rep.evaluated is True  # max coverage clears the orientation gate


# --------------------------------------------------------------------------- #
# Subprocess wrapper
# --------------------------------------------------------------------------- #
def test_subprocess_nonzero_returns_not_evaluated(tmp_path, monkeypatch):
    def fake_run(argv, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="boom traceback")

    monkeypatch.setattr(reproject_adapter.subprocess, "run", fake_run)
    rep = reproject_adapter.reproject_report(
        tmp_path / "m.step", tmp_path / "d.png", tmp_path / "out", RunConfig(), regions=REGIONS
    )
    assert rep.evaluated is False
    assert "boom" in rep.skipped_reason


def test_subprocess_timeout_returns_not_evaluated(tmp_path, monkeypatch):
    def fake_run(argv, **kwargs):
        raise reproject_adapter.subprocess.TimeoutExpired(cmd=argv, timeout=1)

    monkeypatch.setattr(reproject_adapter.subprocess, "run", fake_run)
    rep = reproject_adapter.reproject_report(
        tmp_path / "m.step", tmp_path / "d.png", tmp_path / "out", RunConfig(),
        regions=REGIONS, timeout_s=1,
    )
    assert rep.evaluated is False
    assert "timed out" in rep.skipped_reason


def test_reproject_report_reads_written_report(tmp_path, monkeypatch):
    def fake_run(argv, **kwargs):
        out_dir = Path(argv[argv.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(
            json.dumps(make_report({"front": make_view(0.97), "top": make_view(0.96)}))
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(reproject_adapter.subprocess, "run", fake_run)
    rep = reproject_adapter.reproject_report(
        tmp_path / "m.step", tmp_path / "d.png", tmp_path / "out", RunConfig(), regions=REGIONS
    )
    assert rep.evaluated is True
    assert rep.views_found == 2


def test_no_regions_returns_not_evaluated(tmp_path):
    rep = reproject_adapter.reproject_report(
        tmp_path / "m.step", tmp_path / "d.png", tmp_path / "out", RunConfig(), regions=None
    )
    assert rep.evaluated is False
    assert "view layout" in rep.skipped_reason


def test_regions_written_and_passed_to_subprocess(tmp_path, monkeypatch):
    captured: dict = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        out_dir = Path(argv[argv.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.json").write_text(
            json.dumps(make_report({"front": make_view(0.97), "top": make_view(0.96)}))
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(reproject_adapter.subprocess, "run", fake_run)
    rep = reproject_adapter.reproject_report(
        tmp_path / "m.step", tmp_path / "d.png", tmp_path / "out", RunConfig(), regions=REGIONS
    )
    assert rep.evaluated is True
    argv = captured["argv"]
    assert "--regions-json" in argv
    regions_path = Path(argv[argv.index("--regions-json") + 1])
    assert json.loads(regions_path.read_text()) == REGIONS


# --------------------------------------------------------------------------- #
# Composite (requires opencv)
# --------------------------------------------------------------------------- #
def test_build_composite_skips_missing_view(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    from PIL import Image

    for name in ("front", "side"):  # 'top' intentionally absent
        img = np.full((40, 30, 3), 255, np.uint8)
        img[10:30, 5] = (255, 90, 0)  # a blue error line (BGR)
        cv2.imwrite(str(tmp_path / f"overlay_{name}.png"), img)

    def view(name: str) -> ReprojectionView:
        return ReprojectionView(
            coverage=0.8, chamfer_pct=1.0, aspect_ok=True, aspect_rel_err=0.0,
            aspect_signed=0.0, overlay_path=tmp_path / f"overlay_{name}.png",
        )

    out = reproject_adapter._build_composite(
        {"front": view("front"), "side": view("side")}, tmp_path / "composite.png"
    )
    assert out is not None and out.exists()
    assert Image.open(out).width > 0
