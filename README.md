# cad-gen — Agentic Drawing → CAD

Turn a **technical drawing** into real, manufacturable CAD geometry (**STEP + STL**)
through a **self-refine loop**: an LLM writes [CadQuery](https://cadquery.readthedocs.io/)
code, the code runs in a sandboxed subprocess, the resulting solid is rendered and
**reprojected against the drawing by a deterministic geometric arbiter**, a **vision-model
critic** scores it, and the code is refined iteratively until it passes a quality
threshold. An optional text description can be supplied alongside the drawing to
disambiguate. The same loop also **edits an existing model** from a text instruction.

The design principle throughout: the LLM proposes, and something deterministic decides. Every
signal the critic is given — kernel-measured metrics, the reprojection overlap, the OCCT
validity verdict, the before/after edit diff — is computed from the geometry, not
self-reported, and the ones that are unambiguous (an invalid solid, an unchanged model)
override the critic outright rather than arguing with it.

Built on [PydanticAI](https://ai.pydantic.dev/), so it is **LLM-agnostic** — any
supported provider works by changing one model string (OpenAI by default). Ships with a
CLI, a Python library API, and a **Next.js + FastAPI web dashboard** that streams every
iteration live and renders the generated solid in 3D in the browser.

> **New here?** [`docs/report/README.md`](docs/report/README.md) is the project report: the
> **GEARS** framework the loop is built on, worked demos with real renders, the measured
> results, the limitations, and where to pick the project up. This file is the operator manual.

```
drawing (+ optional text) ─► ORCHESTRATOR (outer loop: quality)
                    │
                    ├─ interpret drawing → dimension digest + structured constraints
                    │  locate views (VLM) → front/top/side boxes
                    ▼
   GENERATOR agent ──── execute_cad_code tool ──► subprocess sandbox
          │    ▲                                  (CadQuery / OpenCASCADE:
          │    └── traceback retry (inner loop:    STL, STEP, volume, bbox, watertight)
          │                        correctness)
          ▼
   RENDERER (iso / front / top / right composite PNG)
          ▼
   REPROJECT check ─► deterministic overlap score + overlay (missing/extra lines)
          ▼
   VALIDITY gate ─► OCCT BRepCheck + closed shells + manifold mesh (invalid ⇒ score 0)
          ▼
   CRITIC agent (vision) ─► Critique{score 0-10, matches_spec, issues, suggestions}
          │
          ├─ score ≥ threshold ─► ACCEPT: final/ + report.md
          └─ else: feedback + overlay → next iteration (budget-capped, best VALID one wins)
```

The same loop also runs in **editing mode** — given an existing STEP and an instruction,
modify that model instead of building one from scratch. The drawing stages fall away and the
base model takes their place: it is seeded into the sandbox, inventoried into a feature
briefing for the generator, and used as the reference the candidate is measured against.

---

## Table of contents

- [Tech stack — APIs, frameworks & tools](#tech-stack--apis-frameworks--tools)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick start (CLI)](#quick-start-cli)
- [Web dashboard](#web-dashboard)
- [Library API](#library-api)
- [Configuration reference](#configuration-reference)
- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Output artifacts](#output-artifacts)
- [Benchmarking with CADGenBench](#benchmarking-with-cadgenbench)
- [Testing & development](#testing--development)
- [Deployment](#deployment)
- [Security notes](#security-notes)
- [Known limitations](#known-limitations)
- [Future work](#future-work)

---

## Tech stack — APIs, frameworks & tools

### Core / backend (Python ≥ 3.12)

| Area | Tool | Role |
|---|---|---|
| **Agent framework** | [PydanticAI](https://ai.pydantic.dev/) (`pydantic-ai-slim[google,openai]`) | Provider-agnostic LLM agents (generator, critic, drawing interpreter, view locator) with typed tool calling and structured outputs |
| **LLM provider (default)** | [OpenAI](https://platform.openai.com/) (`openai-responses:gpt-5.6-luna`) | Generator, vision critic, and drawing view-locator. Swappable per-agent to any PydanticAI provider (e.g. `google:`, `anthropic:`) |
| **CAD kernel** | [CadQuery](https://cadquery.readthedocs.io/) (default) or [build123d](https://build123d.readthedocs.io/) — both on [OpenCASCADE / OCP](https://github.com/CadQuery/OCP) | Build solids from generated Python; export STEP, compute ground-truth metrics (volume, bbox, COM, face/solid count, watertightness). Pick per run with `--library` |
| **Mesh / geometry** | [trimesh](https://trimesh.org/) | STL export and mesh handling |
| **Rendering** | [NumPy](https://numpy.org/) z-buffer renderer | Headless 4-view composite PNG (iso/front/top/right) — no GPU/OSMesa required |
| **Computer vision** | [OpenCV](https://opencv.org/) (`opencv-python-headless`) | Drawing line masks, primitive extraction, distance transforms, reprojection overlays |
| **Plotting** | [Matplotlib](https://matplotlib.org/) | Axis-annotated render composites |
| **CLI** | [Typer](https://typer.tiangolo.com/) | `cad-gen` command-line entry point |
| **Terminal UI** | [Rich](https://rich.readthedocs.io/) | Live progress / formatted output |
| **Config / secrets** | [python-dotenv](https://github.com/theskumar/python-dotenv) | `.env` loading |
| **Packaging** | [uv](https://docs.astral.sh/uv/) (`uv_build`) | Locked, reproducible installs; project build backend |
| **Lint** | [Ruff](https://docs.astral.sh/ruff/) | Linting / formatting |
| **Tests** | [pytest](https://docs.pytest.org/) + `pytest-asyncio` | Offline test suite (no API calls) |

### Web backend (optional `web` extra)

| Tool | Role |
|---|---|
| [FastAPI](https://fastapi.tiangolo.com/) | HTTP API wrapping `generate_cad` |
| [Uvicorn](https://www.uvicorn.org/) | ASGI server |
| [sse-starlette](https://github.com/sysid/sse-starlette) | Server-Sent Events to stream each iteration live |
| `python-multipart` | Spec + drawing file uploads |
| [httpx](https://www.python-httpx.org/) | Async HTTP client |
| [fal-client](https://fal.ai/) | Optional product-style "showcase" image generation (opt-in) |

### Frontend (`web/`, Node.js)

| Tool | Role |
|---|---|
| [Next.js 16](https://nextjs.org/) (App Router) + [React 19](https://react.dev/) | Dashboard UI |
| [TypeScript](https://www.typescriptlang.org/) | Type-safe frontend |
| [Three.js](https://threejs.org/) + [@react-three/fiber](https://r3f.docs.pmnd.rs/) + [drei](https://github.com/pmndrs/drei) + [three-stdlib](https://github.com/pmndrs/three-stdlib) | Interactive 3D orbit view of the generated STL |
| [Tailwind CSS v4](https://tailwindcss.com/) | Styling |
| [Motion](https://motion.dev/) | Animations |
| [Shiki](https://shiki.style/) | Syntax-highlighted CadQuery code blocks |
| [pnpm](https://pnpm.io/) | Package manager |

---

## Prerequisites

- **Python ≥ 3.12**
- **[uv](https://docs.astral.sh/uv/)** — the Python package/dependency manager used here.
  Install: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- An **LLM API key** — a single **OpenAI** key covers all default agents
  ([get one here](https://platform.openai.com/api-keys)). Any agent can be pointed at
  Google/Gemini or Anthropic instead by changing its `provider:` prefix; see
  [Configuration reference](#configuration-reference).
- *(Web dashboard only)* **Node.js ≥ 20** and **[pnpm](https://pnpm.io/installation)**.
- *(Docker deploy only)* **Docker**.

> A pre-configured **VS Code Dev Container** is included (`.devcontainer/`) if you prefer
> a ready-made environment.

---

## Installation

```bash
# 1. clone
git clone <repo-url> cad-agent && cd cad-agent

# 2. install Python deps into a locked virtualenv (.venv/)
uv sync

# 3. configure secrets
cp .env.example .env        # then edit .env and paste your OPENAI_API_KEY
```

That's it for the CLI and library. The CAD kernel (CadQuery / OpenCASCADE) and the
headless renderer are pure Python wheels — **no system OpenGL/OSMesa needed**.

For the **web dashboard**, also install the optional backend extra and the frontend:

```bash
uv sync --extra web         # FastAPI, uvicorn, SSE, multipart, fal-client
cd web && pnpm install      # frontend deps
```

---

## Quick start (CLI)

```bash
# from a technical drawing (JPEG/PNG) — opens the dimension digest for review first
uv run cad-gen "bracket" --drawing exampledrawings/"WhatsApp Image 2026-06-13 at 18.10.05 (2).jpeg"

# higher quality bar and bigger iteration budget
uv run cad-gen "mounting bracket" \
  --drawing path/to/drawing.jpeg \
  --threshold 9 --max-iterations 6
```

The first positional argument is an optional text description that disambiguates the
drawing; `--drawing` (repeatable for multi-sheet) is the authoritative input.

### CLI options

| option | default | meaning |
|---|---|---|
| `--max-iterations, -n` | 5 | outer self-refine iteration budget |
| `--threshold, -t` | 8 | critic score (0–10) required to accept |
| `--model, -m` | `openai-responses:gpt-5.6-luna` | generator model (`provider:name`) |
| `--critic-model` | `openai-responses:gpt-5.6-luna` | vision critic model |
| `--library, -l` | `cadquery` | CAD library the generator writes (`cadquery` \| `build123d`) |
| `--drawing` | – | path to a technical drawing (repeatable for multi-sheet) |
| `--timeout` | 60 | sandbox seconds per execution attempt |
| `--out, -o` | `runs/` | artifacts directory |

`CAD_GEN_MODEL` / `CAD_GEN_CRITIC_MODEL` / `CAD_GEN_VIEW_MODEL` env vars (or `.env`) set
the same defaults; the CLI flags override them. `--library` has no env var — it is a
per-run choice.

**Exit codes:** `0` accepted · `1` budget exhausted (best effort still written) ·
`2` configuration error.

### Choosing a CAD library

The generator writes either **CadQuery** (default) or **build123d**. Both are Python BREP
frameworks on the same OpenCASCADE kernel and both emit STEP + STL, so everything
downstream — rendering, the reprojection check, metrics, the benchmark adapter — is
identical either way. What changes is the code the LLM writes, the cheat sheet it is given,
and the geometry-probe tool it gets:

| | CadQuery | build123d |
|---|---|---|
| style | method chaining on `cq.Workplane` | `with BuildPart()` builder blocks |
| selection | string selectors (`.faces(">Z")`) | `ShapeList` methods (`.faces().sort_by(Axis.Z)[-1]`) |
| probe tool | `check_selector(code, target, selector)` | `check_selection(code, expression)` |
| install | core dependency | `uv sync --extra build123d` |

```bash
uv sync --extra build123d          # one-time
cad-gen "a 40x30x10mm plate with a 6mm centred hole" --library build123d
```

Selecting `build123d` without installing the extra fails immediately with an install hint
rather than erroring inside the sandbox. The library is also selectable in the web
dashboard under **model & sandbox**, and is recorded in each run's `config.json`.

> **Note on pinning:** the extra pins `build123d>=0.9,<0.10`, which shares the
> `cadquery-ocp` 7.8.x already locked for CadQuery — so installing it leaves the CadQuery
> path untouched. build123d ≥0.10 moves to `cadquery-ocp-novtk` ≥7.9 and would upgrade the
> kernel underneath CadQuery and `reproject/check.py`. See the comment in `pyproject.toml`
> before relaxing either bound.

---

## Web dashboard

A Next.js dashboard drives the loop from the browser: type a spec (or drop a drawing),
watch each iteration stream in live over SSE, **orbit the real generated geometry in 3D**,
inspect the generated code and critique, and browse run history. It talks to a thin FastAPI
service that wraps `generate_cad`.

Run both processes locally (needs `OPENAI_API_KEY` in `.env`):

```bash
# one-time: install the web extra + frontend deps
uv sync --extra web
cd web && pnpm install && cd ..

# terminal 1 — backend (FastAPI + SSE) on :8000
# --reload-dir scopes the watcher to source so runs writing to runs/ don't restart the server mid-run
uv run --extra web uvicorn cad_gen.web.server:app --reload --reload-dir src/cad_gen

# terminal 2 — frontend (Next.js) on :3000
cd web && pnpm dev
```

Open <http://localhost:3000>.

**Drawing workflow:** drop a JPEG/PNG on the form, review/edit the auto-extracted
dimensions, optionally add a clarifying text note, then generate. When a run finishes, the
result header includes an optional **fal.ai showcase** button (set `FAL_KEY` on the backend
to enable) that turns the final render into a product-style image.

### Web API surface (FastAPI)

| Endpoint | Purpose |
|---|---|
| `POST /api/runs` | start a run (multipart: drawings + optional spec text) |
| `GET /api/runs/{id}/events` | SSE stream: `started → iteration* → result` |
| `GET /api/runs` / `GET /api/runs/{id}` | run history / single run |
| `POST /api/drawings/interpret` | run drawing interpretation alone (the human-review gate) |
| artifact routes | serve run files read-only (absolute fs paths stripped) |

---

## Library API

```python
import asyncio
from pathlib import Path
from cad_gen import RunConfig, generate_cad
from cad_gen.models import DrawingAttachment

async def main():
    drawing = DrawingAttachment(
        data=Path("path/to/drawing.jpeg").read_bytes(),
        media_type="image/jpeg",
    )
    result = await generate_cad(
        "mounting bracket",  # optional clarifying note
        RunConfig(max_iterations=6, score_threshold=8),
        drawings=[drawing],
    )
    print(result.accepted, result.best.effective_score, result.run_dir)

asyncio.run(main())
```

`generate_cad(spec, config, *, drawings=..., interpretation=..., base_step=...,
reference_images=..., on_iteration=...)` returns a
`RunResult{accepted, best, iterations, run_dir, edit_diff, ...}`. Most collaborators
(models, executor, renderer, reprojector, `on_iteration` callback) are injectable for
testing.

Passing `base_step` (the bytes of an existing model) switches the run to **editing mode** —
modify that model per the instruction instead of building from scratch:

```python
result = await generate_cad(
    "Remove the groove inside the largest-diameter bore.",
    RunConfig(max_iterations=5),
    base_step=Path("input.step").read_bytes(),
)
print(result.edit_diff.digest)  # what material the edit actually moved, and where
```

Two `RunConfig` knobs gate the deterministic checks, both on by default and neither
env-configurable: `validity_gate` (per-iteration OCCT validity, with `validity_timeout_s`)
and `edit_diff` (the post-loop boolean diff, with `edit_diff_timeout_s` and
`edit_diff_max_candidates`). Turn them off only for runs that will never be submitted —
the validity gate costs one subprocess per iteration.

---

## Configuration reference

All configuration is via environment variables (or `.env`); see **`.env.example`** for the
annotated source of truth. Summary:

| Variable | Default | Applies to |
|---|---|---|
| `OPENAI_API_KEY` | – | **required** — generator, critic, view-locator |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) / `ANTHROPIC_API_KEY` | – | only if you point an agent at that provider |
| `CAD_GEN_MODEL` | `openai-responses:gpt-5.6-luna` | generator model |
| `CAD_GEN_CRITIC_MODEL` | `openai-responses:gpt-5.6-luna` | vision critic model |
| `CAD_GEN_VIEW_MODEL` | (critic default) | drawing view-locator (vision) |
| `CAD_GEN_REASONING_EFFORT` | provider default | generator thinking effort (`minimal…xhigh`) |
| `CAD_GEN_CRITIC_REASONING_EFFORT` | provider default | critic thinking effort |
| `CAD_GEN_BMW` | – | set to `1` to route **all** agents through the BMW LLM gateway |
| `CAD_GEN_WEB_ORIGINS` | `http://localhost:3000` | web CORS allow-list (comma-separated) |
| `CAD_GEN_RUNS_DIR` | `runs` | artifacts root |
| `FAL_KEY` | – | optional fal.ai showcase image |
| `CAD_GEN_SHOWCASE_MODEL` | `fal-ai/flux-pro/kontext` | showcase image model |
| `NEXT_PUBLIC_API_BASE_URL` *(frontend, `web/.env`)* | `http://localhost:8000` | backend URL the UI calls |

Switching provider = change the `provider:` prefix and set the matching API key. Note: the
CLI's key preflight checks openai/google/anthropic; other providers fail at call time if the
key is missing.

### BMW LLM API (BMW PCs only)

On a BMW PC the public provider endpoints are blocked — only BMW's OpenAI-spec-compatible
gateway is reachable. Point cad-gen at it either way:

- **One switch:** `CAD_GEN_BMW=1` forces every agent through the gateway (default model
  `openai/gpt-5-mini`). An explicit `bmw:<model>` env var still overrides per agent.
- **Per agent:** prefix any model string with `bmw:`, e.g. `CAD_GEN_MODEL=bmw:openai/gpt-5-mini`
  (model ids follow the BMW catalog: `openai/gpt-5-mini`, `openai/gpt-4o`,
  `anthropic/claude-sonnet-4-5`, …).

Set the credentials (`LLM_API_PROD_KEY`, plus `CLIENT_ID`+`CLIENT_SECRET` or a pre-fetched
`LLM_ACCESS_TOKEN`) in `.env`; region and CA-cert overrides are documented in `.env.example`.
cad-gen fetches/refreshes the WebEAM bearer token and downloads BMW's CA bundle automatically.

---

## How it works

The orchestrator runs an **outer self-refine loop** (quality) wrapping an **inner
correctness loop** (the generator retries on tracebacks):

1. **Run setup** — create `runs/<timestamp>/`, persist the drawing(s), config, and any
   optional text note.
2. **Interpret (once)** — interpret the drawing into a dimension digest + structured
   constraints (deterministic bbox assertions), extract OpenCV line/circle primitives, and
   a **VLM locates the front/top/side view boxes**.
3. **Generate** — the generator agent writes CadQuery code and validates it by calling the
   `execute_cad_code` tool (retrying on errors within the iteration).
4. **Execute** — code runs in an isolated subprocess sandbox → STL, STEP, and
   **ground-truth metrics from the OCCT kernel** (volume, bbox, COM, n_solids, n_faces,
   watertight — never self-reported).
5. **Render** — a headless NumPy z-buffer renderer produces a 4-view composite PNG.
6. **Reproject** — the solid is reprojected into orthographic views
   (OpenCASCADE hidden-line removal) and overlaid on the drawing's line work inside the
   located boxes. A **deterministic, resolution-independent overlap score** judges it — a
   VLM only *proposes* the boxes, so the check can only false-negative, never fake a pass.
   It emits a colour-coded overlay (blue = missing, orange = extra, red = match).
7. **Check validity** — the exported STEP is run through CADGenBench's own validity gate
   (OCCT `BRepCheck_Analyzer`, closed shells, manifold tessellation), in a subprocess with a
   timeout because OCCT will not honour a Python signal mid-call. Like the reprojection check
   this runs *before* the critic, so its verdict grounds the critique instead of competing
   with it. A gate that could not run is "no verdict", never a rejection.
8. **Critique** — the vision critic receives the spec, measured metrics, code, the reproject
   and validity digests, and the images, returning a structured `Critique` with a 0–10 score.
   A **soft-cap** forbids a passing score when the reproject check failed unless the overlay
   shows the flagged views are actually correct. Then the deterministic overrides apply: an
   invalid solid scores 0 on the benchmark whatever else is right about it, so its critique is
   forced to 0 with the OCCT reason attached as the top-priority issue.
9. **Accept or refine** — score ≥ threshold accepts; otherwise champion-anchored feedback
   plus the overlay are carried into the next iteration. Champion choice is
   **validity-first**: a valid iteration scoring 5 outranks an invalid one scoring 9, because
   the latter is worth zero. If the budget runs out, the best iteration is returned (exit 1).

**Editing mode** (`generate_cad(..., base_step=...)`) follows the same loop with the drawing
pipeline dormant and three substitutions: the base model is seeded into every sandbox dir as
`input.step`, a measured **feature briefing** of it goes into every prompt, and the
drawing-vs-solid arbiter is replaced by two base-vs-candidate ones — a per-iteration
volume/bbox **no-op guard** and, once after the loop, a boolean **edit diff** that measures
exactly what material the edit added and removed, and where. Both hard-override the critic:
a measured no-op or invalid solid cannot be accepted regardless of what the renders looked
like.

The **reprojection signal** is the key correctness anchor for drawings — see
[`src/cad_gen/reproject/README.md`](src/cad_gen/reproject/README.md) and
[`src/cad_gen/reproject/DECISIONS.md`](src/cad_gen/reproject/DECISIONS.md) for the full
design. The end-to-end I/O of every stage is documented in **[`pipeline.md`](pipeline.md)**
(with a visual version in `pipeline_visualization.html`).

---

## Project structure

```
.
├── src/cad_gen/
│   ├── orchestrator.py        # the self-refine loop; generate_cad() entry point
│   ├── cli.py                 # Typer CLI (cad-gen)
│   ├── models.py              # Pydantic data models (RunConfig, RunResult, Critique, …)
│   ├── imaging.py             # image helpers
│   ├── drawing_constraints.py # structured dimension assertions from the digest
│   ├── step_metrics.py        # volume/bbox of a STEP via raw OCP; editing no-op guard
│   ├── step_validity.py       # CADGenBench's validity gate + OCCT repair ladder
│   ├── edit_diff.py           # boolean before/after diff of an edit against its base
│   ├── base_briefing.py       # feature inventory (bores, walls) of an editing base model
│   ├── agents/                # PydanticAI agents
│   │   ├── generator.py       #   writes CAD code (execute_cad_code, inspect_geometry,
│   │   │                      #   check_selector/check_selection, find_geometry [editing])
│   │   ├── critic.py          #   vision critic → Critique
│   │   ├── drawing_parser.py  #   drawing → dimension digest
│   │   └── prompts.py         #   system prompts (generation + editing variants)
│   ├── sandbox/               # subprocess execution of generated code
│   │   ├── executor.py · harness.py · introspect.py
│   ├── rendering/renderer.py  # headless NumPy z-buffer 4-view render
│   ├── reproject/             # deterministic drawing-vs-solid geometric arbiter
│   │   ├── check.py           #   self-contained scorer (OCP + OpenCV)
│   │   ├── locator.py         #   VLM that proposes view boxes
│   │   ├── adapter.py         #   runs check.py, builds LLM-facing overlays/digest
│   │   ├── drawing_primitives.py · eval_drawings.py
│   │   └── README.md · DECISIONS.md
│   ├── bench/                 # CADGenBench harness (cad-gen-bench, optional `bench` extra)
│   │   ├── dataset.py         #   fetch + parse benchmark samples from the HF Hub
│   │   ├── adapter.py         #   sample → generate_cad → output.step; validity fallback chain
│   │   ├── submission.py      #   assemble the leaderboard submission zip
│   │   └── cli.py             #   Typer CLI: run · package
│   └── web/                   # FastAPI backend (server.py, runs.py, schemas.py)
├── web/                       # Next.js + React 19 frontend (3D viewer, live SSE)
├── tests/                     # offline pytest suite (no API calls)
├── exampledrawings/           # sample technical drawings
├── runs/                      # generated run artifacts (gitignored output)
├── pipeline.md                # full stage-by-stage I/O documentation
├── Dockerfile                 # backend container image
├── pyproject.toml · uv.lock   # deps & lockfile
└── .env.example               # annotated configuration template
```

---

## Output artifacts

Every run writes `runs/<timestamp>/`:

```
spec.txt  config.json  run_result.json  report.md
input/drawing_NN.{png,jpg}              # persisted input drawings
drawing_interpretation.md               # dimension digest
drawing_constraints.json                # structured constraints
view_layout_NN.json                     # located view boxes
iter_NN/
  attempt_MM/{model.py,model.stl,model.step,metrics.json}
  views.png                             # render
  reproject/{report.json,overlay_*.png} # drawing-vs-solid check
  iteration.json                        # incl. the validity verdict + edit delta
final/  model.py  model.stl  model.step  views.png  critique.json
```

An **editing** run (`base_step=...`) adds:

```
input/input.step                        # the base model, as seeded into the sandbox
input/reference_NN.png                  # the base model's renders
input/base_briefing.md                  # measured bore / wall inventory of the base
iter_NN/edit_diff_lumps.stl             # material this candidate moved (per candidate diffed)
final/edit_diff.json                    # the chosen candidate's measured before/after diff
final/edit_diff.png                     # 4-view render of just the changed material
```

---

## Benchmarking with CADGenBench

[CADGenBench](https://github.com/huggingface/cadgenbench) scores text/drawing→CAD
systems against private ground truth (validity gate → shape / interface / topology
metrics). The `cad-gen-bench` harness (in the optional `bench` extra) fetches the
benchmark samples, drives cad-gen over each, and lays the winning STEP out in the
submission structure the leaderboard expects — cad-gen and CADGenBench both speak STEP,
so no format conversion is involved. Both task families are supported: **generation**
(drawing→CAD) and **editing** (modify a provided `input.step` per an instruction).

```bash
# install the bench extra (huggingface-hub + pyyaml)
uv sync --extra bench

# smoke-test: 2 samples, short budget, in parallel — eyeball before scaling up
uv run --extra bench cad-gen-bench run --limit 2 --max-iterations 2 -o results/smoke

# full run: all 81 samples (49 generation + 32 editing), resumable — reuses any output.step
uv run --extra bench cad-gen-bench run -o results/cadgen-v1 -m openai:gpt-5.5 -n 5 -j 4

# narrow to one family (e.g. iterate on editing): --task-type generation|editing|all
uv run --extra bench cad-gen-bench run --task-type editing -o results/edits

# package into a leaderboard submission zip (meta.json + <sample>/output.step)
uv run --extra bench cad-gen-bench package results/cadgen-v1 \
    --submitter "Your Name" --name "cad-gen gpt-5.5 v1" --agree
```

Then upload the zip via the CADGenBench leaderboard Space's **Submit** tab; the Space
runs scoring against the private ground truth and publishes the CAD Score, per-task-type
breakdown, and validity rate.

- **Inputs** are pulled from the public HF dataset (`HuggingAI4Engineering/cadgenbench-data`,
  overridable via `CADGENBENCH_DATA_REPO` / `--data-repo`); the private ground truth is
  never needed locally.
- **Scope:** a plain `run` executes all 81 samples (a complete submission). `--task-type
  generation|editing|all` (default `all`) narrows it. **Editing** samples seed their
  `input.step` into the sandbox so the generator loads it with
  `cq.importers.importStep("input.step")`, applies the requested change, and preserves the
  rest — the shape axis is renormalized against the no-op baseline, so a minimal correct
  edit is what scores. The base model's `renders/` are attached to the generator and a
  before/after-aware critic as reference. The final CAD Score cannot be computed locally
  (the no-op baseline lives in the private ground truth), but the two things that reliably
  zero it can be, and are:
  - **Validity** (`step_validity.py`): every iteration's exported STEP is run through the
    benchmark's own gate — OCCT `BRepCheck_Analyzer`, closed shells, and a manifold
    tessellation. An invalid candidate has its critique forced to 0 with the OCCT reason
    fed back, and validity outranks the critic's score when picking a champion. If nothing
    valid comes out, a repair ladder is tried, and failing that an editing sample ships its
    unmodified `input.step` — worth up to 0.4 where an invalid edit is worth 0. The summary
    prints every fallback in red; they are failures, not successes.
  - **Locality** (`edit_diff.py`): the champion is cut against the base model both ways, so
    what the edit actually added and removed is measured, not eyeballed. This is what
    catches an over-cut or a from-scratch rebuild — both of which change plenty and so slip
    past the volume/bbox no-op guard.
- **Editing generators get a base-model briefing** (`base_briefing.py`): a measured inventory
  of `input.step`'s bores (radius, axis, extent, through/blind) and planar walls (normal,
  offset, area), plus a `find_geometry` probe that filters faces by type, radius, normal or
  region. Instructions name features in engineering language ("the largest-diameter bore"),
  and `inspect_geometry` alone samples too few faces of a 1000-face import to find them.
- Each sample's full self-refine trace lands beside its candidate under
  `results/<run>/<sample>/cadgen/` for debugging; only `output.step` is packaged.
- **Where the benchmark framing lives:** `sample_to_request` / `sample_to_edit_request` in
  `src/cad_gen/bench/adapter.py` compose the cad-gen spec (millimetres, single watertight
  solid; drawing-authoritative for generation, minimal-change for editing). The benchmark
  descriptions are terse, so this preamble is the main textual lever on quality — edit it
  to taste.

---

## Testing & development

```bash
uv run pytest        # offline: scripted models, no API calls (enforced)
uv run ruff check .  # lint
```

Tests use PydanticAI's `TestModel`/`FunctionModel` with `ALLOW_MODEL_REQUESTS=False`, so
the entire loop — agents, sandbox, renderer, reprojection — is exercised without any API
key or network access.

The geometric checks are tested against **real OCCT**, not mocks: the validity gate, the
boolean edit diff and the base-model briefing all run on STEP files the tests export with
CadQuery. That is what makes them worth having — a mocked `BRepCheck` would agree with
whatever the code believed. The generator's system prompts are pinned to snapshots under
`tests/snapshots/`, so rewording one is a deliberate act (regenerate them in the same
commit) rather than an accident.

---

## Deployment

- **Frontend → Vercel:** set the project root to `web/` and `NEXT_PUBLIC_API_BASE_URL` to
  your backend URL.
- **Backend → container:** the backend **can't** run on Vercel (native OCCT, subprocess
  execution, persistent run artifacts). Build the included `Dockerfile` and host it on
  Fly.io / Render / a VM:

  ```bash
  docker build -t cad-gen-api .
  docker run -p 8000:8000 -e OPENAI_API_KEY=... \
    -e CAD_GEN_WEB_ORIGINS=https://your-app.vercel.app \
    -v cadgen-runs:/data/runs cad-gen-api
  ```

---

## Security notes

- The subprocess sandbox provides **crash/timeout/state isolation, not a security
  boundary** — generated code runs with your user's privileges. The web backend makes this
  reachable over HTTP, so bind it to localhost in development and **never expose it
  publicly without container isolation and auth**.
- **Never commit real secrets.** `.env` is gitignored; only `.env.example` (no secrets)
  belongs in version control. If a key has ever been committed or shared, rotate it.

---

## Known limitations

**Generation.** Simple prismatic parts (plates, brackets, blocks, holes, fillets) converge
reliably. Parts needing a swept feature *fused* to a body — e.g. a mug handle — are at the
edge of current model capability: the model often leaves the feature as a separate, unfused
solid, which the critic correctly rejects via the ground-truth `n_solids` check, so the run
returns its best effort rather than a wrong "accepted". Refining from the best-so-far
iteration keeps these hard cases from diverging but does not guarantee they solve within the
budget.

**Editing** is harder, and the deterministic checks bound the damage rather than removing it:

- The checks catch a bad edit; they do not produce a good one. A rejected candidate becomes a
  retry, and if the budget runs out the run still ships its best effort (flagged).
- **Not every invalid solid is repairable.** The OCCT repair ladder is verified through a STEP
  round-trip, because a shape can pass `BRepCheck_Analyzer` in memory and fail again once
  written and re-read — measured on a real candidate at 0.000000 % volume change. On the seven
  invalid candidates of the audited v3 submission the ladder fixes **none**; it is kept as a
  cheap rung for failure modes that run did not exhibit, and the fallback does the work.
- **Some benchmark base models are themselves invalid.** Three of the 32 CADGenBench editing
  inputs fail the validity gate as shipped, so for those there is no valid no-op to fall back
  to and an inherited defect must be regenerated by the edit itself (a boolean *through* the
  offending region usually does it) or the sample scores 0 either way.
- `is_plausible_local_edit()` in `edit_diff.py` currently accepts every measured diff, so the
  locality gate reports numbers without yet acting on them. Its docstring carries the
  calibration data from the v3 audit.

---

## Future work

pyrender/OSMesa renderer (true hidden-surface removal); multi-part assemblies;
topology-specific dimension-assertion validators beyond the bbox check; auth + persistent
run store for a hosted web deployment.
