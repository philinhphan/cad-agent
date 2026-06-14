# cad-gen — Pipeline Flow (Inputs & Outputs)

Agentic text/drawing → CAD via a self-refine loop. Entry point:
`cad_gen.orchestrator.generate_cad(spec, config, *, drawings, interpretation, …)`.

```
                         ┌─────────────────────────────────────────────────────────────┐
 INPUT                   │  spec (text)  and/or  drawings (JPEG/PNG)  +  RunConfig       │
                         └─────────────────────────────────────────────────────────────┘
                                                   │
   ── run setup ─────────  run_dir/  spec.txt · config.json · input/drawing_NN.*
                                                   │
   ── once, drawing mode ─ ① interpret drawing  → drawing_interpretation.md  (dimension digest)
                           ①b structure/check   → drawing_constraints.json   (dimension assertions)
                           ①c CV primitives     → drawing_primitives_NN.json (lines/circles/text regions)
                           ② locate views (VLM) → view_layout_NN.json        (front/top/side boxes)
                                                   │
   ┌───────────────────────────── self-refine loop (≤ max_iterations) ───────────────────────────┐
   │  ③ build prompt  =  spec + feedback + interpretation + drawing img(s) + champion overlay img  │
   │  ④ GENERATOR (LLM) ──writes CadQuery──▶ ⑤ SANDBOX exec (subprocess) ──▶ STL · STEP · metrics  │
   │  ⑥ RENDER (z-buffer) ──▶ views.png (iso/front/top/right)                                       │
   │  ⑦ REPROJECT each drawing (subprocess) ─▶ report.json · overlays · mismatch clusters · digest  │
   │  ⑧ CRITIC (vision LLM) = spec + metrics + code + reproject digest/verdict + [render,drawings,  │
   │                          overlay] ──▶ Critique{score 0-10, matches_spec, issues, suggestions}  │
   │  ⑨ score ≥ threshold? ── yes ▶ break ── no ▶ feedback(champion) + champion overlay ▶ next iter │
   └───────────────────────────────────────────────────────────────────────────────────────────────┘
                                                   │
   ── finalize ──────────  best = max(score, index) ▶ final/ · report.md · run_result.json
                                                   │
 OUTPUT                   RunResult{accepted, best, iterations, run_dir, …}
```

## Stages

| # | Stage | Code | Input | Output |
|---|-------|------|-------|--------|
| 0 | Run setup | `generate_cad`, `_new_run_dir`, `_persist_drawings` | spec, drawings, `RunConfig` | `run_dir/` + `spec.txt`, `config.json`, `input/drawing_NN.{png,jpg}` |
| ① | Drawing interpret (once, drawing mode) | `agents/drawing_parser.py` (`interpret_drawing`, model `config.model`) | drawing image(s) + spec | Markdown dimension digest → `drawing_interpretation.md`. Skipped if caller passed `interpretation` (CLI/web human-review gate). |
| ①b | Structured constraints | `drawing_constraints.py` (`derive_drawing_constraints`, `validate_drawing_constraints`) | dimension digest + later kernel metrics | `drawing_constraints.json` + per-iteration `ConstraintValidation` digest. Bbox is checked deterministically; feature callouts are tracked as unverified until topology-specific validators exist. |
| ①c | Drawing primitive extraction | `reproject/drawing_primitives.py` (`extract_drawing_primitives`, `primitive_prompt_summary`) | each persisted drawing image | `drawing_primitives_NN.json` plus generator prompt hints: top line/circle/arc/centerline/arrow/text-region candidates with normalized coordinates, radius/endpoint hints, and nearest text-region links. |
| ② | View localization (once per drawing+reproject) | `reproject/locator.py` (`locate_drawing_views`, **VLM** `config.view_model`) | each drawing image | `ViewLayout` = normalized front/top/side boxes → `view_layout_NN.json` (plus `view_layout.json` alias for the first sheet) → `view_regions` dict. Failure → `None` (that sheet's reproject withholds). |
| ③ | Build prompt | `orchestrator._build_prompt` | spec, `feedback` (champion), `interpretation`, drawings, champion overlay bytes | `str` (text-only) **or** `[text, overlay_img?, *drawing_imgs]` |
| ④ | Generator | `agents/generator.py` (`build_generator_agent`, `config.model`, `config.reasoning_effort`) | prompt | CadQuery code; validated by calling the `execute_cad_code` tool (≤ `max_exec_attempts_per_iteration`) |
| ⑤ | Sandbox execution | `sandbox/executor.py` + `sandbox/harness.py` (subprocess) | CadQuery code | `ExecutionResult{success, code, metrics, stl_path, step_path, error}`; writes `attempt_NN/{model.py, model.stl, model.step, metrics.json}`. Metrics = volume, bbox, COM, n_solids, n_faces, watertight. |
| ⑥ | Render | `rendering/renderer.py` (`render_views`, numpy z-buffer) | STL + metrics | `iter_NN/views.png` (2×2 iso/front/top/right, mm axes) |
| ⑦ | Reproject check (drawing mode, **before** critic) | `reproject/adapter.py` (`reproject_report`, `combine_reprojection_reports`) → subprocess `reproject/check.py` | STEP + every drawing + each drawing's `view_regions` | Per drawing: `report.json`, overlays, structured mismatch clusters. Multi-sheet runs aggregate into one `ReprojectionReport{children=[...]}` and optional composite. See sub-pipeline below. |
| ⑧ | Critic | `agents/critic.py` (`run_critique`, **vision LLM** `config.critic_model`) | spec + metrics + code + structured constraints + constraint validation + reproject digest/verdict + images `[render, *drawings, overlay_composite]` | `Critique{matches_spec, score 0-10, issues, suggestions, summary}` |
| — | Persist iteration | `orchestrator` | record | `iter_NN/iteration.json`; fires `on_iteration` callback (web SSE) |
| ⑨ | Accept / refine | `orchestrator`, `_build_feedback` | `effective_score` (= critic score) | `≥ score_threshold` → break; else champion-anchored `feedback` + champion `overlay_composite` bytes carried to next ③ |
| ⑩ | Finalize | `_persist_final`, `_write_report` | all iterations | `final/{model.py, model.stl, model.step, views.png, critique.json}`, `report.md`, `run_result.json`; returns `RunResult` |

## ⑦ Reproject sub-pipeline (the deterministic geometric arbiter)

A **VLM proposes** the view boxes (②); a **deterministic check judges** the overlap — so it can
only produce false-negatives, never a fake pass. Advisory to the score (no hard gate), but the
critic must respect a failed check (soft-cap, see below).

```
reproject_report(step, drawing, out_dir, config, regions)         [adapter.py, in-process]
  regions empty/None ───────────────────────────────▶ evaluated=False ("no view layout")
  else: write regions.json → subprocess: check.py --drawing --step --regions-json --auto-orient
        check.py [subprocess, OCP+OpenCV, no cad_gen imports]:
          load STEP → HLR-project into front/top/side  ·  load drawing → geometry-line mask
          snap each region to its ink (_tighten_box)   ·  fit reprojection by bbox, score
          ▶ report.json {per-view coverage, chamfer_pct, aspect} + overlay_{view}.png
        adapter post-processes report.json →
          ▶ structured mismatch clusters from overlay blue/orange connected components
          ▶ overlay_composite.png  (labelled front|top|side, blue=missing/orange=extra/red=match)
          ▶ digest (per-view coverage verbalised) + interpretation line
          ▶ ReprojectionReport{evaluated, passed, views, digest, composite_path, …}
  withheld (evaluated=False) when: subprocess error/timeout · no views located ·
                                   all views low coverage AND aspects still match (≈ orientation)
  multiple input drawings ─────────▶ combine_reprojection_reports(children=[per-sheet reports])
```

**Routing of the reproject outputs:**
- **overlay_composite.png → generator** (next iteration, via `_build_feedback` + `_build_prompt`): a
  *locator* of where lines are missing (blue) / extra (orange). Dimensionless — the generator must
  read corrected sizes off the **original drawing**, not the overlay.
- **digest text + PASS/FAIL verdict + overlay image → critic**: grounds the score on falsifiable
  evidence. **Soft-cap:** on `passed=False` the critic may not give `matches_spec` / score ≥ 8 unless
  the overlay shows the flagged views are actually correct.

## Inputs

- **spec**: natural-language part description (optional if a drawing is given).
- **drawings**: `list[DrawingAttachment]` (JPEG/PNG; bytes + media type). Drawing mode enables ①②⑦.
- **interpretation** (optional): pre-extracted/edited dimension digest (CLI editor gate or web). If
  omitted in drawing mode, auto-generated by ①.
- **RunConfig** (env-resolvable, see `.env.example`):
  - `model` ← `CAD_GEN_MODEL` (generator) · `reasoning_effort` ← `CAD_GEN_REASONING_EFFORT`
  - `critic_model` ← `CAD_GEN_CRITIC_MODEL` (vision) · `view_model` ← `CAD_GEN_VIEW_MODEL` (vision)
  - `max_iterations` (5) · `score_threshold` (8) · `exec_timeout_s` (60) ·
    `max_exec_attempts_per_iteration` (4) · `out_dir` (`runs`)
  - `reproject` (True) · `reproject_timeout_s` (120) · `reproject_low_coverage` (0.80) ·
    `reproject_orientation_coverage` (0.55)
- Injectable for tests: `generator_model`, `critic_model`, `interpreter_model`, `view_locator_model`,
  `executor`, `renderer`, `reprojector`, `on_iteration`.

## Output: `RunResult` + on-disk run directory

`RunResult{accepted, spec, drawings, interpretation, best, iterations, run_dir}` —
`accepted = best.effective_score ≥ score_threshold`; `best = max(iterations, key=(score, index))`.

```
runs/<timestamp>/
├── spec.txt · config.json
├── input/drawing_NN.{png,jpg}            # persisted input drawings (drawing mode)
├── drawing_interpretation.md             # ① dimension digest
├── drawing_constraints.json              # ①b structured constraints
├── drawing_primitives_NN.json            # ①c OpenCV primitive candidates per drawing
├── view_layout.json                      # ② compatibility alias for first drawing's VLM boxes
├── view_layout_NN.json                   # ② VLM view boxes per drawing
├── iter_NN/
│   ├── attempt_MM/{model.py, model.stl, model.step, metrics.json}   # ⑤ each exec attempt
│   ├── views.png                         # ⑥ render
│   ├── reproject/{report.json, overlay_{front,top,side}.png, overlay_composite.png, regions.json}  # ⑦ single drawing
│   ├── reproject/drawing_NN/{report.json, overlay_*.png, regions.json}  # ⑦ multi drawing
│   └── iteration.json                    # IterationRecord (execution, critique, reprojection)
├── final/{model.py, model.stl, model.step, views.png, critique.json}   # ⑩ best iteration
├── report.md                             # human-readable summary table
└── run_result.json                       # full RunResult
```

## Entry points

- **CLI** (`cli.py`): `cad-gen "<spec>" [--drawing … --model … --threshold …]` → loads `.env`, builds
  `RunConfig`, optional editor review of the dimension digest, runs `generate_cad`, prints per-iteration
  progress + verdict.
- **Web** (`web/server.py`, `web/runs.py`): `POST /api/runs` (multipart spec + drawings) → `generate_cad`
  on a worker thread; `GET /api/runs/{id}/events` streams `started → iteration* → result` over SSE;
  artifacts served read-only (absolute fs paths stripped). `POST /api/drawings/interpret` runs ① alone
  for the review gate.

## Key data models (`models.py`)

`DrawingAttachment` · `GeometryMetrics` · `ExecutionResult` · `Critique` ·
`ReprojectionView`/`ReprojectionReport` · `IterationRecord` (`effective_score` = critic score, else 0) ·
`RunConfig` · `RunResult`.
