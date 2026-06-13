# cad-gen

Agentic text-to-CAD: turns a natural-language part specification into solid CAD
geometry (STEP + STL) through a **self-refine loop** — an LLM writes
[CadQuery](https://cadquery.readthedocs.io/) code, the code is executed in a
sandboxed subprocess, the resulting geometry is rendered and **visually
inspected by a vision-model critic** that scores it against the spec, and the
code is refined iteratively until the quality threshold is met.

Built on [PydanticAI](https://ai.pydantic.dev/), so it is LLM-agnostic: any
supported provider works by changing one model string (OpenAI by default).

```
spec ─► ORCHESTRATOR (outer loop: quality)
          │
          ▼
   GENERATOR agent ──── execute_cad_code tool ──► subprocess sandbox
          │    ▲                                  (CadQuery/OCCT: STL, STEP,
          │    └── traceback retry (inner loop:    volume, bbox, watertight)
          │                        correctness)
          ▼
   RENDERER (iso / front / top / right composite PNG)
          ▼
   CRITIC agent (vision) ─► Critique{score 0-10, issues, suggestions}
          │
          ├─ score ≥ threshold ─► ACCEPT: final/ + report.md
          └─ else: feedback → next iteration (budget-capped, best effort wins)
```

## Setup

```bash
uv sync
cp .env.example .env   # paste your OPENAI_API_KEY
```

## Usage

```bash
uv run cad-gen "a 40mm cube with a 10mm diameter centered through-hole"

uv run cad-gen "rectangular mounting bracket 60x40x8mm with 4x M4 clearance \
  holes (4.5mm) inset 6mm from corners, 3mm filleted vertical edges" \
  --threshold 9 --max-iterations 6
```

| option | default | meaning |
|---|---|---|
| `--max-iterations, -n` | 5 | outer self-refine iteration budget |
| `--threshold, -t` | 8 | critic score (0–10) required to accept |
| `--model, -m` | `openai:gpt-5.2` | generator model (`provider:name`) |
| `--critic-model` | same as `--model` | vision critic model |
| `--timeout` | 60 | sandbox seconds per execution attempt |
| `--out, -o` | `runs/` | artifacts directory |

`CAD_GEN_MODEL` / `CAD_GEN_CRITIC_MODEL` env vars (or `.env`) also work.

Exit codes: `0` accepted · `1` budget exhausted (best effort still written) ·
`2` configuration error.

### Artifacts

Every run writes `runs/<timestamp>/`:

```
spec.txt  config.json  run_result.json  report.md
iter_01/  attempt_01/{model.py,model.stl,model.step,metrics.json}
          views.png  iteration.json
iter_02/  ...
final/    model.py  model.stl  model.step  views.png  critique.json
```

### Library API

```python
from cad_gen import RunConfig, generate_cad

result = await generate_cad(
    "a coffee mug, 80mm diameter, 100mm tall, 3mm wall, with handle",
    RunConfig(max_iterations=6, score_threshold=8),
)
result.accepted, result.best.effective_score, result.run_dir
```

## How acceptance works

The critic receives the spec, the **measured** geometry (volume, bounding box,
solid count, watertightness from the OCCT kernel — not self-reported), the
generated code, and a 4-view render. It returns a structured critique with a
0–10 score; ≥ 8 (configurable) accepts. Failed executions score 0 and feed the
traceback back to the generator. If the budget runs out, the best-scoring
iteration is returned and the run exits 1.

## Web view

A Next.js dashboard (in `web/`) drives the loop from the browser: type a spec,
watch each iteration stream in live over SSE, orbit the real generated geometry
in 3D, and browse run history. It talks to a thin FastAPI service that wraps
`generate_cad`.

Run both processes locally (needs `OPENAI_API_KEY` in `.env`):

```bash
# terminal 1 — backend (FastAPI + SSE), serves on :8000
uv run uvicorn cad_gen.web.server:app --reload

# terminal 2 — frontend (Next.js), serves on :3000
cd web && pnpm install && pnpm dev
```

Open <http://localhost:3000>. Backend env (all optional): `CAD_GEN_WEB_ORIGINS`
(comma-separated CORS allow-list, default `http://localhost:3000`),
`CAD_GEN_RUNS_DIR` (artifacts root, default `runs`). Frontend env:
`NEXT_PUBLIC_API_BASE_URL` (default `http://localhost:8000`; see `web/.env.example`).

**Deploying:** the frontend deploys to Vercel (set root directory to `web/` and
`NEXT_PUBLIC_API_BASE_URL` to your backend URL). The backend can't run on Vercel
(native OCCT, subprocess execution, persistent run artifacts) — build the
included `Dockerfile` and host it on Fly.io / Render / a VM, setting
`OPENAI_API_KEY` and `CAD_GEN_WEB_ORIGINS` (your Vercel domain).

## Security note

The subprocess sandbox provides **crash/timeout/state isolation, not a
security boundary** — generated code runs with your user's privileges. The web
backend makes this reachable over HTTP, so bind it to localhost in development
and **never expose it publicly without container isolation and auth**.

## Development

```bash
uv run pytest        # offline: scripted models, no API calls (enforced)
uv run ruff check .
```

Tests use PydanticAI's `TestModel`/`FunctionModel` with
`ALLOW_MODEL_REQUESTS=False`, so the whole loop is testable without a key.

## Known limitations

Simple prismatic parts (plates, brackets, blocks, holes, fillets) converge
reliably. Parts needing a swept feature *fused* to a body — e.g. a mug handle —
are at the edge of current model capability: the model often leaves the feature
as a separate, unfused solid, which the critic correctly rejects via the
ground-truth `n_solids` check, so the run returns its best effort rather than a
wrong "accepted". Refining from the best-so-far iteration keeps these hard cases
from diverging, but does not guarantee they solve within the budget.

## Future work

pyrender/OSMesa renderer (true hidden-surface removal); multi-part assemblies;
dimension-assertion validator parsed from the spec; auth + persistent run store
for a hosted web deployment.
