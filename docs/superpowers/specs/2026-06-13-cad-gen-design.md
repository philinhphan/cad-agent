# cad-gen — Agentic CAD Generation with Self-Refine Loop

## Context

Build a new application (greenfield, `/Users/philinh/cad-gen` is empty) that turns a natural-language part specification into CAD geometry. An LLM generates CAD code; the code is executed in a subprocess via a tool call; the resulting geometry is rendered and visually inspected by a vision-capable critic that produces critique, a score, and validation; the output is iteratively refined (self-refine loop) until a quality threshold is reached or the iteration budget is exhausted.

**Decisions confirmed with the user:**
- Agentic SDK: **PydanticAI** (LLM-agnostic — provider swap is a model-string change; Gemini models by default)
- CAD library: **CadQuery** (parametric BREP on OCCT; STEP + STL export; best LLM codegen success)
- Interface: **CLI** (`cad-gen "<spec>"`), with the core exposed as an importable async library function
- API key: user will provide `GEMINI_API_KEY` via a gitignored `.env`

**Environment facts (verified):**
- macOS arm64, `uv 0.8.18`, Python 3.13.7; CadQuery 2.7.0 + cadquery-ocp 7.9.3 have cp313 macosx_arm64 wheels → no Python pinning gymnastics
- No LLM API key currently in env; offline tests must not require one
- PydanticAI current API (verified via docs): `Agent("google:gemini-3.5-flash", output_type=Model, instructions=...)`, `BinaryContent(data, media_type="image/png")` for images, `agent.override(model=TestModel()/FunctionModel(fn))` + `models.ALLOW_MODEL_REQUESTS = False` for offline tests. Default model remains overridable via flag/env.

## Approaches considered

1. **Explicit orchestrated self-refine loop (CHOSEN)** — application code owns the iterate/score/stop logic; agents are steps inside it. Generator agent has an `execute_cad_code` tool so runtime errors are fixed *within* an iteration; the vision critic runs after every successful execution and drives the *outer* refinement. Deterministic, guaranteed critique each iteration, enforceable budgets, fully testable offline with `FunctionModel`.
2. *Single autonomous agent with all tools* — agent decides when to execute/render/critique/stop. Less code, but unreliable (may skip critique, stop early, loop), hard to test, budget not enforceable. Rejected.
3. *LangGraph state machine* — equivalent control to (1) but adds a heavy dependency; plain Python loop + PydanticAI achieves the same. Rejected.

## Architecture

```
spec ─► ORCHESTRATOR (async loop, max_iterations, threshold)
          │ iteration i
          ▼
   GENERATOR agent (PydanticAI, output: CadQuery code rationale)
          │ tool: execute_cad_code(code)  ◄─┐ retry w/ traceback
          ▼                                 │ (inner loop, ≤4 attempts)
   SANDBOX executor ── subprocess: harness.py ─► model.stl / model.step / metrics.json
          │ last successful execution = code of record
          ▼
   RENDERER: STL ─► trimesh+matplotlib ─► views.png (iso/front/top/right composite)
          ▼
   CRITIC agent (vision; inputs: spec + metrics JSON + views.png)
          │ structured output: Critique{score 0-10, matches_spec, issues, suggestions}
          ▼
   score ≥ threshold ─► ACCEPT (final artifacts + report)
   else feedback (code + critique) ─► next iteration; at budget: best-scoring iteration wins
```

Key loop semantics:
- **Inner loop (correctness):** generator must call the `execute_cad_code` tool; on failure the tool returns the traceback and the agent retries (≤4 executions/iteration). The tool records every attempt in per-iteration workspace state (PydanticAI `deps`); the *last successful execution* is the code of record — not whatever code string the agent prints.
- **Outer loop (quality):** critic scores renders+metrics against the spec. Feedback prompt for iteration N+1 = spec + current code + critic issues/suggestions (structured re-prompt; no raw message history — cheaper and avoids history bloat).
- **Stop:** `score >= threshold` (default 8) → accept; `iterations == max_iterations` (default 5) → return best-by-score; an iteration with no successful execution scores 0 and its "critique" is the traceback (loop continues).
- Every iteration persisted to `runs/<timestamp>/iter_NN/` (code, STL, STEP, renders, metrics, critique); `final/` + `report.md` written at the end.

## Project layout

```
cad-gen/
  pyproject.toml            # uv-managed; [project.scripts] cad-gen = "cad_gen.cli:app"
  .python-version           # 3.13
  .env.example              # GEMINI_API_KEY=...
  .gitignore                # .env, runs/, __pycache__, .venv
  README.md
  docs/superpowers/specs/2026-06-13-cad-gen-design.md   # this design, committed in Phase 1
  src/cad_gen/
    __init__.py             # exports generate_cad, RunConfig, RunResult
    models.py               # Pydantic schemas (below)
    sandbox/
      harness.py            # standalone script run IN the subprocess (no package imports)
      executor.py           # run_cad_code(code, out_dir, timeout) -> ExecutionResult
    rendering/
      renderer.py           # render_views(stl_path, out_png, metrics) -> Path
    agents/
      prompts.py            # system prompts + compact CadQuery cheatsheet/conventions
      generator.py          # build_generator_agent(model) -> Agent[GenDeps, str]
      critic.py             # build_critic_agent(model) -> Agent[None, Critique]
    orchestrator.py         # async generate_cad(spec, config) -> RunResult; persistence; report.md
    cli.py                  # typer app, rich progress, .env loading (python-dotenv)
  tests/
    conftest.py             # ALLOW_MODEL_REQUESTS=False; tmp run dirs; cube STL fixture
    test_models.py          # schema bounds (score 0-10 etc.)
    test_executor.py        # real cadquery subprocess: good code, syntax error, timeout, no-result
    test_renderer.py        # trimesh-generated cube STL -> composite PNG exists, plausible size
    test_agents.py          # generator tool wiring via FunctionModel; critic schema via TestModel
    test_orchestrator.py    # scripted FunctionModels: fail→fix→accept; budget exhaustion picks best
```

## Core contracts

**Schemas (`models.py`):**
```python
GeometryMetrics: volume_mm3, bbox (x/y/z), center_of_mass, n_solids, n_faces, is_watertight
ExecutionResult: success, code, error (traceback|None), metrics|None, stl_path|None,
                 step_path|None, stdout, duration_s
Critique:        matches_spec: bool, score: int (ge=0, le=10), issues: list[str],
                 suggestions: list[str], summary: str
IterationRecord: index, execution: ExecutionResult, render_path|None, critique|None
RunConfig:       model (default "google:gemini-3.5-flash"), critic_model (defaults to Gemini),
                 max_iterations=5, score_threshold=8, exec_timeout_s=60,
                 max_exec_attempts_per_iteration=4, out_dir="runs"
RunResult:       accepted: bool, best: IterationRecord, iterations: list[IterationRecord],
                 run_dir: Path
```

**Sandbox contract (`sandbox/`):** `executor.run_cad_code()` writes code to `iter_NN/model.py`, then `subprocess.run([sys.executable, harness.py, model.py, out_dir], timeout=...)`. `harness.py` (self-contained, imports only stdlib + cadquery) execs the code in a namespace pre-seeded with `cadquery as cq` and a `show_object()` shim; locates the result solid by convention — variable `result`, else `show_object()`-registered objects, else any Workplane/Shape in the namespace; exports `model.stl` + `model.step`; writes `metrics.json` (OCCT volume/bbox/solids; watertightness computed in parent via trimesh). Failure ⇒ traceback on stderr + nonzero exit. **Note:** subprocess gives crash/timeout/state isolation, not a security boundary — acceptable for a local dev tool; documented in README.

**Renderer (`rendering/renderer.py`):** matplotlib `Agg` (headless-safe on macOS — no OpenGL/EGL flakiness) + trimesh. 2×2 composite PNG (~1400px): isometric, front, top, right; per-face normal shading; equal aspect; suptitle shows measured bbox so the critic can check dimensions. One composite image per critique keeps vision token cost low. Known limit: painter's-algorithm z-order artifacts on overlapping bodies — fine for shape-level critique; pyrender/OSMesa is future work.

**Agents (`agents/`):**
- *Generator* — `Agent(model, deps_type=GenDeps, output_type=str)`; final output is only a short summary — the code of record is always the last successful `execute_cad_code` tool call recorded in the workspace, never the agent's restated text. Instructions: write parametric CadQuery against the harness conventions (assign to `result`, mm units), ALWAYS call `execute_cad_code` and finish only after a successful run; tool returns success+metrics or traceback. `GenDeps` carries the iteration workspace (executor closure + attempt log).
- *Critic* — `Agent(critic_model, output_type=Critique)`; run with `[prompt_text, BinaryContent(views_png, media_type="image/png")]`; prompt includes spec + metrics JSON; instructions: strict spec compliance check (features present, proportions, dimensions vs measured bbox), concrete code-level suggestions, scoring rubric (8+ = accept).

**CLI (`cli.py`):** `cad-gen "<spec>" [--max-iterations 5] [--threshold 8] [--model google:gemini-3.5-flash] [--critic-model M] [--timeout 60] [--out runs]`. Loads `.env`; clear error if `GEMINI_API_KEY`/`GOOGLE_API_KEY` is missing for the default provider. Rich live table: iteration, exec ok/✗, score, top issue. Exit codes: 0 accepted, 1 budget exhausted (best effort still written), 2 hard failure.

## Implementation phases (TDD; superpowers test-driven-development skill)

1. **Scaffold** — `git init`; `uv init --package`; pin 3.13; deps: `pydantic-ai-slim[google]`, `cadquery`, `trimesh`, `numpy`, `matplotlib`, `typer`, `rich`, `python-dotenv`; dev: `pytest`, `pytest-asyncio`, `ruff`. `.env.example`, `.gitignore`, commit design doc to `docs/superpowers/specs/`. Smoke-check `import cadquery` works (earliest risk retirement).
2. **Schemas** — `models.py` + `test_models.py`.
3. **Sandbox executor** — `harness.py` + `executor.py` + `test_executor.py` (good box code; syntax error; runtime error; no `result` var; timeout via `while True`).
4. **Renderer** — `renderer.py` + `test_renderer.py` (cube STL fixture from `trimesh.creation.box` — no LLM/cadquery needed).
5. **Agents** — prompts + generator (tool wiring tested with `FunctionModel` scripting a failing-then-passing tool sequence) + critic (`TestModel` → valid `Critique`).
6. **Orchestrator** — self-refine loop, artifact persistence, `report.md`; `test_orchestrator.py` with scripted `FunctionModel`s: (a) iteration 1 scores 5, iteration 2 scores 9 → accepted, 2 iterations on disk; (b) all low scores → budget exhaustion returns best; (c) unrunnable code → score-0 iteration, loop continues.
7. **CLI + README** — typer entry point, rich progress, docs (usage, architecture sketch, sandbox security note).
8. **Live E2E** — user pastes key into `.env`; verify model name `google:gemini-3.5-flash` against the live API at first call (fall back via `--model` if needed); run the verification specs below; tune prompts if the loop doesn't converge.

## Verification

- **Offline (no key):** `uv run pytest` — full suite green, `ALLOW_MODEL_REQUESTS=False` guarantees no accidental API calls.
- **Live E2E (needs `GEMINI_API_KEY` in `.env`):**
  1. `uv run cad-gen "a 40mm cube with a 10mm diameter centered through-hole"` — expect accept in 1–2 iterations; open `runs/<ts>/final/views.png`, check STL watertight + volume ≈ 40³ − hole.
  2. `uv run cad-gen "rectangular mounting bracket 60x40x8mm with 4x M4 clearance holes (4.5mm) inset 6mm from corners, 3mm filleted vertical edges" --threshold 9` — expect the refine loop to actually iterate (visible score progression in CLI + report.md).
  3. Confirm `report.md` timeline, per-iteration artifacts, and exit codes (0 / 1 paths).

## Non-goals (v1) / risks

- Not a security sandbox (documented); no web viewer (CLI option chosen; three.js viewer is future work); no assemblies/multi-part; no STEP import.
- OCCT subprocess startup ~2–5s per execution — acceptable; matplotlib z-order artifacts on complex overlaps — acceptable for critique; the default Gemini model can be replaced live with `--model` as an escape hatch.
