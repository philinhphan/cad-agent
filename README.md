# cad-gen — Agentic Drawing → CAD

Turn a **technical drawing** into real, manufacturable CAD geometry (**STEP + STL**)
through a **self-refine loop**: an LLM writes [CadQuery](https://cadquery.readthedocs.io/)
code, the code runs in a sandboxed subprocess, the resulting solid is rendered and
**reprojected against the drawing by a deterministic geometric arbiter**, a **vision-model
critic** scores it, and the code is refined iteratively until it passes a quality
threshold. An optional text description can be supplied alongside the drawing to
disambiguate.

Built on [PydanticAI](https://ai.pydantic.dev/), so it is **LLM-agnostic** — any
supported provider works by changing one model string (OpenAI by default). Ships with a
CLI, a Python library API, and a **Next.js + FastAPI web dashboard** that streams every
iteration live and renders the generated solid in 3D in the browser.

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
   CRITIC agent (vision) ─► Critique{score 0-10, matches_spec, issues, suggestions}
          │
          ├─ score ≥ threshold ─► ACCEPT: final/ + report.md
          └─ else: feedback + overlay → next iteration (budget-capped, best effort wins)
```

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
| **LLM provider (default)** | [OpenAI](https://platform.openai.com/) (`openai:gpt-5-mini`) | Generator, vision critic, and drawing view-locator. Swappable per-agent to any PydanticAI provider (e.g. `google:`, `anthropic:`) |
| **CAD kernel** | [CadQuery](https://cadquery.readthedocs.io/) + [OpenCASCADE / OCP](https://github.com/CadQuery/OCP) | Build solids from generated Python; export STEP, compute ground-truth metrics (volume, bbox, COM, face/solid count, watertightness) |
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
- An **LLM API key** — a single **Google/Gemini** key covers all default agents.
  ([Get one here.](https://aistudio.google.com/apikey))
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
| `--model, -m` | `openai:gpt-5-mini` | generator model (`provider:name`) |
| `--critic-model` | `openai:gpt-5-mini` | vision critic model |
| `--drawing` | – | path to a technical drawing (repeatable for multi-sheet) |
| `--timeout` | 60 | sandbox seconds per execution attempt |
| `--out, -o` | `runs/` | artifacts directory |

`CAD_GEN_MODEL` / `CAD_GEN_CRITIC_MODEL` / `CAD_GEN_VIEW_MODEL` env vars (or `.env`) set
the same defaults; the CLI flags override them.

**Exit codes:** `0` accepted · `1` budget exhausted (best effort still written) ·
`2` configuration error.

---

## Web dashboard

A Next.js dashboard drives the loop from the browser: type a spec (or drop a drawing),
watch each iteration stream in live over SSE, **orbit the real generated geometry in 3D**,
inspect the CadQuery code and critique, and browse run history. It talks to a thin FastAPI
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

`generate_cad(spec, config, *, drawings=..., interpretation=..., on_iteration=...)`
returns a `RunResult{accepted, best, iterations, run_dir, ...}`. Most collaborators
(models, executor, renderer, reprojector, `on_iteration` callback) are injectable for
testing.

---

## Configuration reference

All configuration is via environment variables (or `.env`); see **`.env.example`** for the
annotated source of truth. Summary:

| Variable | Default | Applies to |
|---|---|---|
| `OPENAI_API_KEY` | – | **required** — generator, critic, view-locator |
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) / `ANTHROPIC_API_KEY` | – | only if you point an agent at that provider |
| `CAD_GEN_MODEL` | `openai:gpt-5-mini` | generator model |
| `CAD_GEN_CRITIC_MODEL` | `openai:gpt-5-mini` | vision critic model |
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
7. **Critique** — the vision critic receives the spec, measured metrics, code, the
   reproject digest/verdict, and the images, returning a structured `Critique` with a 0–10
   score. A **soft-cap** forbids a passing score when the reproject check failed unless the
   overlay shows the flagged views are actually correct.
8. **Accept or refine** — score ≥ threshold accepts; otherwise champion-anchored feedback
   plus the overlay are carried into the next iteration. If the budget runs out, the
   best-scoring iteration is returned (exit 1).

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
│   ├── agents/                # PydanticAI agents
│   │   ├── generator.py       #   writes CadQuery code (+ execute_cad_code tool)
│   │   ├── critic.py          #   vision critic → Critique
│   │   ├── drawing_parser.py  #   drawing → dimension digest
│   │   └── prompts.py         #   system prompts
│   ├── sandbox/               # subprocess execution of generated code
│   │   ├── executor.py · harness.py · introspect.py
│   ├── rendering/renderer.py  # headless NumPy z-buffer 4-view render
│   ├── reproject/             # deterministic drawing-vs-solid geometric arbiter
│   │   ├── check.py           #   self-contained scorer (OCP + OpenCV)
│   │   ├── locator.py         #   VLM that proposes view boxes
│   │   ├── adapter.py         #   runs check.py, builds LLM-facing overlays/digest
│   │   ├── drawing_primitives.py · eval_drawings.py
│   │   └── README.md · DECISIONS.md
│   ├── eval/checks.py         # evaluation helpers
│   ├── bench/                 # CADGenBench harness (cad-gen-bench, optional `bench` extra)
│   │   ├── dataset.py         #   fetch + parse benchmark samples from the HF Hub
│   │   ├── adapter.py         #   sample → generate_cad → output.step (sample_to_request)
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
  iteration.json
final/  model.py  model.stl  model.step  views.png  critique.json
```

---

## Benchmarking with CADGenBench

[CADGenBench](https://github.com/huggingface/cadgenbench) scores text/drawing→CAD
systems against private ground truth (validity gate → shape / interface / topology
metrics). The `cad-gen-bench` harness (in the optional `bench` extra) fetches the
benchmark's generation samples, drives cad-gen over each drawing, and lays the winning
STEP out in the submission structure the leaderboard expects — cad-gen and CADGenBench
both speak STEP, so no format conversion is involved.

```bash
# install the bench extra (huggingface-hub + pyyaml)
uv sync --extra bench

# smoke-test: 2 samples, short budget, in parallel — eyeball before scaling up
uv run --extra bench cad-gen-bench run --limit 2 --max-iterations 2 -o results/smoke

# full run: all 49 generation samples (resumable — reuses any existing output.step)
uv run --extra bench cad-gen-bench run -o results/cadgen-v1 -m openai:gpt-5.5 -n 5 -j 4

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
- **Scope:** only `generation` samples (49 of them) are run — the drawing→CAD task cad-gen
  is built for. `editing` samples are skipped (recorded "missing" / 0 by the grader).
- Each sample's full self-refine trace lands beside its candidate under
  `results/<run>/<sample>/cadgen/` for debugging; only `output.step` is packaged.
- The `run` summary flags any candidate that isn't a single watertight solid — a cheap
  local proxy for the benchmark's validity gate (which scores non-watertight / multi-solid
  parts 0), using metrics cad-gen already computes.
- **Where the benchmark framing lives:** `sample_to_request` in
  `src/cad_gen/bench/adapter.py` composes the cad-gen spec (millimetres, drawing-authoritative,
  single watertight solid). The benchmark descriptions are terse and identical, so this
  preamble is the main textual lever on quality — edit it to taste.

---

## Testing & development

```bash
uv run pytest        # offline: scripted models, no API calls (enforced)
uv run ruff check .  # lint
```

Tests use PydanticAI's `TestModel`/`FunctionModel` with `ALLOW_MODEL_REQUESTS=False`, so
the entire loop — agents, sandbox, renderer, reprojection — is exercised without any API
key or network access.

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

Simple prismatic parts (plates, brackets, blocks, holes, fillets) converge reliably. Parts
needing a swept feature *fused* to a body — e.g. a mug handle — are at the edge of current
model capability: the model often leaves the feature as a separate, unfused solid, which
the critic correctly rejects via the ground-truth `n_solids` check, so the run returns its
best effort rather than a wrong "accepted". Refining from the best-so-far iteration keeps
these hard cases from diverging but does not guarantee they solve within the budget.

---

## Future work

pyrender/OSMesa renderer (true hidden-surface removal); multi-part assemblies;
topology-specific dimension-assertion validators beyond the bbox check; auth + persistent
run store for a hosted web deployment.
