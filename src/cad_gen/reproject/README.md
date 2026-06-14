# reproject

Check that a generated **STEP** really matches a technical **drawing**. The **scoring** is
a falsifiable, deterministic geometric arbiter; only the *view localisation* is delegated
to a VLM upstream (`cad_gen`), which supplies the per-view boxes via `--regions-json`. A
wrong box yields a low overlap score — never a false pass.

It reprojects the solid into orthographic views (OpenCASCADE hidden-line removal), isolates
the drawing's geometry lines inside the supplied view boxes, snaps each box to its line
work, fits each reprojection into it by bounding box (no dimension reading), and scores the
overlap. Output: a colour-coded overlay PNG per view plus a JSON report with line coverage,
Chamfer distance and a scale-free bounding-box aspect check.

All scores are **resolution-independent** (normalised by view size), so the same
thresholds work regardless of the drawing's pixel resolution.

The scorer (`check.py`) is self-contained — stdlib + OpenCV + cadquery-ocp only, no
`cad_gen` imports — so `adapter.py` runs it as an isolated subprocess. Dependencies are
declared in the project's `pyproject.toml`; there is no separate install step.

This folder (`cad_gen.reproject`):
- `check.py` — the deterministic geometric scorer (this README).
- `locator.py` — the VLM that proposes the per-view boxes.
- `adapter.py` — runs `check.py` and turns its report/overlays into LLM-facing artifacts.
- `eval_drawings.py` — drawing-only eval of the locator over `exampledrawings/`.
- `DECISIONS.md` — the design rationale for the whole reprojection signal.

## Usage (standalone scorer)

```bash
python src/cad_gen/reproject/check.py \
    --drawing      path/to/drawing.png \
    --step         path/to/model.step \
    --regions-json path/to/regions.json \
    --out          out_reproj/
```

`regions.json` maps each present view to a normalized box (origin top-left, values in
`[0, 1]`), e.g. `{"front": [0.13, 0.66, 0.23, 0.39], "top": [0.13, 0.18, 0.13, 0.30]}`.

Outputs in `--out`:
- `overlay_front.png`, `overlay_top.png`, `overlay_side.png` — colour-coded overlay:
  **red** = reprojection and drawing agree, **grey** = matched drawing line,
  **blue** = drawing line with no reprojection nearby (missing / displaced feature),
  **orange** = reprojected line with no drawing line nearby (extra / wrong feature).
- `report.json` — per-view `coverage`, `precision`, `chamfer_pct`, aspect check, and an
  overall `pass`.

### Options

| flag | meaning |
|------|---------|
| `--no-color-filter` | keep all dark pixels (use for pure black/white scans) |
| `--no-hidden`       | do not project hidden edges (default: include, for fair coverage) |
| `--auto-orient`     | try 8 in-plane orientations per view, keep the best fit |
| `--max-chamfer-pct` | gross-error threshold: max mean edge distance as % of view diagonal (default 2.0) |
| `--min-coverage`    | pass threshold: min fraction of drawing lines matched within the tolerance band (default 0.90) |
| `--aspect-tol`      | relative bbox aspect tolerance (default 0.1) |
| `--deflection`      | HLR curve discretisation in mm (default 0.1) |
| `--debug`           | dump the intermediate geometry-line mask |

## Conventions & assumptions

- **Part orientation:** X = width, Y = depth, Z = height. Views:
  `front` looks along Y (shows X×Z), `top` along Z (shows X×Y), `side` along X (shows Y×Z).
  If the generated STEP comes out in a different axis frame, the fixed view mapping can
  misassign views — use `--auto-orient` to absorb in-plane rotation/mirroring.
- **View boxes are supplied, not detected:** the front/top/side regions come in via
  `--regions-json` (a VLM in `cad_gen` locates them, ignoring iso-renders, section/detail
  views and the title block). Each supplied box is snapped to its line work, so a coarse
  box still bounds the view. A view absent from the JSON → `null` in the report.
- **Colour coding:** geometry = black, dimensions = blue, centrelines = green,
  iso-renders = red. The colour filter keeps only black, dropping the rest for free.
  On non-colour-coded scans it auto-falls back to plain thresholding.

## What "pass" means

`pass = true` only if, for **every** located view:
- **coverage ≥ `--min-coverage`** — almost every drawing line has a reprojected line
  within the tolerance band. This is the main signal: a wrong solid leaves drawing lines
  uncovered (highlighted blue in the overlay). Coverage is recall, not a mean distance —
  a localised error (one feature in the wrong place) is caught even on a large view where
  a mean distance would average it away.
- **chamfer_pct ≤ `--max-chamfer-pct`** — a coarse guard against gross global misfit.
- **aspect ratios match** — the 3D bounding-box proportions agree with the drawing views.

The metric is **falsifiable**: it can (and does) fail, which is what makes a pass
meaningful. Coverage is reported as recall only; `precision` (reprojected lines absent
from the drawing) is reported for diagnostics but not gated, since a correct model
legitimately carries extra edges (hidden lines, tangents) the drawing may not show.

### Calibration note

Defaults (`min-coverage 0.90`, `max-chamfer-pct 2.0`) were set with a clear margin
between a known-correct part (min coverage ~0.98) and several wrong generations (≤0.80).
A genuinely correct but very complex part may sit a little lower; treat these as
sensitivity knobs, and if you have a known-correct reference for your part family,
confirm it passes and adjust if needed.
