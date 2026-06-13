"""FastAPI application factory for the cad-gen web backend.

Wraps the existing async ``generate_cad`` loop and streams each self-refine
iteration to the browser over SSE. The OpenAI key stays server-side.

Security note: this server executes LLM-generated CadQuery in a subprocess —
crash/timeout isolation only, NOT a security sandbox (same caveat as the CLI,
now reachable over HTTP). Bind to localhost in development; public hosting
requires container isolation and auth.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from cad_gen.web import runs as runs_mod
from cad_gen.web.schemas import StartRunRequest, StartRunResponse

DEFAULT_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


def _resolve_origins(origins: list[str] | None) -> list[str]:
    if origins is not None:
        return origins
    env = os.environ.get("CAD_GEN_WEB_ORIGINS")
    if env:
        return [o.strip() for o in env.split(",") if o.strip()]
    return DEFAULT_ORIGINS


def _resolve_runs_dir(runs_dir: Path | str | None) -> Path:
    if runs_dir is not None:
        return Path(runs_dir)
    return Path(os.environ.get("CAD_GEN_RUNS_DIR", "runs"))


def create_app(
    *, origins: list[str] | None = None, runs_dir: Path | str | None = None
) -> FastAPI:
    """Build the FastAPI app. Args override the matching env vars (for tests)."""
    load_dotenv(Path.cwd() / ".env")

    app = FastAPI(title="cad-gen", version="0.1.0")
    app.state.runs_dir = _resolve_runs_dir(runs_dir)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=_resolve_origins(origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/runs", response_model=StartRunResponse)
    def start_run(req: StartRunRequest) -> StartRunResponse:
        handle = runs_mod.start_run(req.spec, req.config, app.state.runs_dir)
        return StartRunResponse(run_id=handle.run_id)

    @app.get("/api/runs")
    def list_runs() -> list[dict]:
        return runs_mod.list_runs(app.state.runs_dir)

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict:
        return runs_mod.run_detail(run_id, app.state.runs_dir)

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str, request: Request) -> EventSourceResponse:
        handle = runs_mod.RUNS.get(run_id)
        if handle is None:
            raise HTTPException(status_code=404, detail="unknown run")
        return EventSourceResponse(runs_mod.event_stream(handle))

    @app.get("/api/runs/{run_id}/iterations/{index}/{name}")
    def iteration_artifact(run_id: str, index: int, name: str) -> FileResponse:
        path = runs_mod.iteration_artifact_path(
            run_id, index, name, app.state.runs_dir
        )
        return FileResponse(path)

    return app


app = create_app()
