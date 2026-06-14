"""Adapter: run the deterministic reproject_check on (drawing, STEP) and turn its
human-facing report + overlays into LLM-facing artifacts.

ADVISORY ONLY (see ReprojectionReport): the result never gates the score. The
reproject_check module itself is left untouched and invoked as a subprocess (mirroring
sandbox/executor.py), so an OCCT HLR crash or hang cannot take down the run; its files
are post-processed here into a compact text digest (for the critic) and a labelled
multi-view overlay composite (for the generator).

cv2/matplotlib are imported lazily inside the functions that need them, so importing
this module (hence the whole package) stays cheap and robust even before
opencv-python-headless is installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from cad_gen.models import ReprojectionReport, ReprojectionView, RunConfig

ReprojectorFn = Callable[..., ReprojectionReport]

# Run the geometric scorer by file path (like executor.py's HARNESS): this runs the
# script directly without importing the cad_gen package into the subprocess.
_REPROJECT = Path(__file__).resolve().parent / "check.py"
_STDERR_TAIL = 3000
_VIEW_ORDER = ("front", "top", "side")
_ADVISORY = (
    "This is a deterministic geometric check, advisory only; the original drawing "
    "remains the source of truth for all dimensions."
)


def reproject_report(
    step_path: Path,
    drawing_path: Path,
    out_dir: Path,
    config: RunConfig,
    regions: dict | None = None,
    timeout_s: float = 120,
) -> ReprojectionReport:
    """Run reproject_check on (drawing, step) in a subprocess and build LLM artifacts.

    `regions` are the VLM-located orthographic view boxes ({name: [x, y, w, h]} normalized
    0..1); without them there is nothing to reproject against. Returns an advisory
    ReprojectionReport — any failure (no view layout, subprocess error/timeout, unparseable
    drawing, or a global orientation/scale mismatch) yields ``evaluated=False`` so the
    caller withholds the signal instead of penalising the model.
    """
    if not regions:
        return _not_evaluated("no view layout (VLM locator returned no views)")
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    regions_file = out_dir / "regions.json"
    regions_file.write_text(json.dumps(regions))
    argv = [
        sys.executable,
        str(_REPROJECT),
        "--drawing", str(Path(drawing_path).resolve()),
        "--step", str(Path(step_path).resolve()),
        "--out", str(out_dir),
        "--regions-json", str(regions_file),
        "--auto-orient",  # hedge in-plane orientation; color_filter/hidden are CLI-default-on
    ]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout_s, cwd=out_dir
        )
    except subprocess.TimeoutExpired:
        return _not_evaluated(f"reproject timed out after {timeout_s:g}s")
    if proc.returncode != 0:
        return _not_evaluated(proc.stderr[-_STDERR_TAIL:] or f"exit code {proc.returncode}")

    report_file = out_dir / "report.json"
    if not report_file.exists():
        return _not_evaluated("reproject produced no report.json")
    try:
        report = json.loads(report_file.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return _not_evaluated(f"unreadable report.json: {exc}")
    return _build_from_report(report, out_dir, config)


def _not_evaluated(reason: str) -> ReprojectionReport:
    return ReprojectionReport(evaluated=False, skipped_reason=reason)


def _build_from_report(report: dict, out_dir: Path, config: RunConfig) -> ReprojectionReport:
    """Turn a reproject_check report dict into an advisory ReprojectionReport.

    Withholds the signal (``evaluated=False``) when no views were located, or when
    every located view has low coverage together — the latter almost always means a
    global orientation/scale difference (auto-orient only hedges in-plane rotation),
    not a feature error, so penalising on it would mislead the generator.
    """
    overall = report.get("overall") or {}
    views_found = int(overall.get("views_found") or 0)
    raw_views = report.get("views") or {}

    valid = {
        name: v
        for name, v in raw_views.items()
        if isinstance(v, dict) and "coverage" in v
    }
    if views_found == 0 or not valid:
        return _not_evaluated("drawing views could not be located")

    covs = [v["coverage"] for v in valid.values()]
    # Global-low coverage is only attributed to orientation/scale (and withheld) when the
    # bbox aspects still MATCH — a correct-but-rotated part keeps its proportions. If the
    # aspects are also grossly off, the part is genuinely wrong, and saying so is the most
    # useful signal we can give, so we surface it instead of masking it.
    aspects_match = all(v.get("aspect_ok") for v in valid.values())
    if (
        len(covs) >= 2
        and max(covs) < config.reproject_orientation_coverage
        and aspects_match
    ):
        return _not_evaluated(
            "all views' coverage is low but bbox aspects still match — likely a global "
            "orientation/scale difference rather than a feature error"
        )

    views: dict[str, ReprojectionView] = {}
    for name, v in valid.items():
        draw_a = float(v.get("draw_aspect", 0.0))
        step_a = float(v.get("step_aspect", 0.0))
        rel = float(v.get("aspect_rel_err", 0.0))
        overlay = v.get("overlay")
        views[name] = ReprojectionView(
            coverage=float(v["coverage"]),
            chamfer_pct=float(v.get("chamfer_pct", 0.0)),
            aspect_ok=bool(v.get("aspect_ok", True)),
            aspect_rel_err=rel,
            aspect_signed=rel if step_a > draw_a else -rel,  # + => part too wide
            overlay_path=(out_dir / overlay) if overlay else None,
        )

    digest, interpretation = _build_digest(views, config)
    return ReprojectionReport(
        evaluated=True,
        passed=bool(overall.get("pass", False)),
        views_found=views_found,
        mean_chamfer_pct=overall.get("mean_chamfer_pct"),
        views=views,
        digest=digest,
        interpretation=interpretation,
        composite_path=_safe_composite(views, out_dir / "overlay_composite.png"),
    )


def _build_digest(views: dict[str, ReprojectionView], config: RunConfig) -> tuple[str, str]:
    """Build the critic-facing text digest (+ the standalone interpretation line).

    Only action-relevant numbers: per-view coverage (verbalised) and aspect deviation.
    The raw report's precision/iou/chamfer_px/orient are diagnostic and dropped here.
    """
    lines: list[str] = []
    for name in (n for n in _VIEW_ORDER if n in views):
        v = views[name]
        pct = round(v.coverage * 100)
        if v.coverage >= 0.95:
            lines.append(f"- {name} view: {pct}% of the drawing's lines reproduced — fully matched.")
        elif v.coverage >= config.reproject_low_coverage:
            lines.append(
                f"- {name} view: {pct}% of the drawing's lines reproduced — mostly matched, "
                "minor gaps."
            )
        else:
            lines.append(
                f"- {name} view: only {pct}% of the drawing's lines reproduced — geometry is "
                "missing or misplaced here (shown blue in the overlay)."
            )
        if not v.aspect_ok:
            apct = round(v.aspect_rel_err * 100)
            side = "wide for its height" if v.aspect_signed > 0 else "tall for its width"
            lines.append(f"  - proportions are off: the part is ~{apct}% too {side} here.")

    interpretation = _interpretation(views, config)
    digest = "\n".join(lines) + "\n\n" + interpretation + "\n\n" + _ADVISORY
    return digest, interpretation


def _interpretation(views: dict[str, ReprojectionView], config: RunConfig) -> str:
    """One line distinguishing a global (orientation/scale) miss from a localised one."""
    order = [n for n in _VIEW_ORDER if n in views]
    covs = {n: views[n].coverage for n in order}
    vals = list(covs.values())
    low = config.reproject_low_coverage

    if len(vals) == 1:
        return (
            f"Only the {order[0]} view could be located, so treat this as a local hint, "
            "not a full check."
        )
    if all(c >= low for c in vals):
        return "All checked views match the drawing well."
    if all(c < low for c in vals):
        return (
            "Almost none of the drawing is reproduced in any view — the overall shape is "
            "wrong or major features are missing; rebuild the part's overall form to match "
            "the drawing before refining details."
        )
    worst = min(order, key=lambda n: covs[n])
    return (
        f"The {worst} view is the weakest ({round(covs[worst] * 100)}%) while the others match "
        f"— this points to a localised feature error in the {worst} view; use the overlay to "
        "locate it, then read the correct value off the original drawing."
    )


def _safe_composite(views: dict[str, ReprojectionView], out_png: Path) -> Path | None:
    """Best-effort composite; a failure here must not lose the (text) digest signal."""
    try:
        return _build_composite(views, out_png)
    except Exception:
        return None


def _build_composite(views: dict[str, ReprojectionView], out_png: Path) -> Path | None:
    """Combine the per-view overlays into one labelled, legend-bearing PNG.

    Error lines (blue=missing, orange=extra) are dilated so a vision model actually
    sees them; the colour legend is baked into the figure. Missing views are omitted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    images = []
    for name in (n for n in _VIEW_ORDER if n in views):
        path = views[name].overlay_path
        if path is None:
            continue
        emphasized = _load_emphasized(path)
        if emphasized is not None:
            images.append((name, emphasized))
    if not images:
        return None

    fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5), squeeze=False)
    for ax, (name, img) in zip(axes[0], images):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"{name} view")
        ax.set_xticks([])
        ax.set_yticks([])
    handles = [
        Patch(color=(0.0, 0.0, 1.0), label="blue = drawing line missing in model"),
        Patch(color=(1.0, 0.55, 0.0), label="orange = model line not in drawing"),
        Patch(color=(1.0, 0.0, 0.0), label="red = match"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(
        "Reprojection overlay — diagnostic locator only; read dimensions off the drawing"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    return out_png


def _load_emphasized(path: Path):
    """Load an overlay (BGR) and thicken its blue/orange error lines for VLM salience."""
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    out = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for color in ((255, 90, 0), (0, 140, 255)):  # blue, orange error lines (BGR, exact)
        arr = np.array(color, dtype=np.uint8)
        mask = cv2.dilate(cv2.inRange(img, arr, arr), kernel)
        out[mask > 0] = color
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
