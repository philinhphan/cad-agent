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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from cad_gen.agents.drawing_parser import build_drawing_parser_agent, interpret_drawing
from cad_gen.imaging import ALLOWED_MEDIA_TYPES, MAX_DRAWING_BYTES, MAX_DRAWINGS, media_type_for
from cad_gen.models import DrawingAttachment, RunConfig
from cad_gen.web import runs as runs_mod
from cad_gen.web.schemas import StartRunResponse

DEFAULT_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]


async def _to_attachments(files: list[UploadFile]) -> list[DrawingAttachment]:
    """Validate uploaded drawing files (type + size + count) into attachments."""
    if len(files) > MAX_DRAWINGS:
        raise HTTPException(
            status_code=422, detail=f"at most {MAX_DRAWINGS} drawings per run"
        )
    attachments: list[DrawingAttachment] = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_DRAWING_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"{f.filename or 'drawing'} exceeds {MAX_DRAWING_BYTES} bytes",
            )
        media = media_type_for(f.filename, data)
        if media not in ALLOWED_MEDIA_TYPES:
            raise HTTPException(
                status_code=422, detail="drawings must be PNG or JPEG images"
            )
        attachments.append(
            DrawingAttachment(filename=f.filename or "drawing", media_type=media, data=data)
        )
    return attachments


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

    @app.post("/api/drawings/interpret")
    async def interpret(
        spec: str = Form(""),
        config: str = Form("{}"),
        drawings: list[UploadFile] = File(default=[]),
    ) -> dict:
        """Extract a dimensions digest from drawing(s) for the user to review/edit."""
        atts = await _to_attachments(drawings)
        if not atts:
            raise HTTPException(status_code=422, detail="no drawings to interpret")
        cfg = RunConfig.model_validate_json(config)
        agent = build_drawing_parser_agent(cfg.model)
        text = await interpret_drawing(agent, spec=spec, drawings=atts)
        return {"interpretation": text}

    @app.post("/api/runs", response_model=StartRunResponse)
    async def start_run(
        spec: str = Form(""),
        config: str = Form("{}"),
        interpretation: str | None = Form(None),
        drawings: list[UploadFile] = File(default=[]),
    ) -> StartRunResponse:
        atts = await _to_attachments(drawings)
        if not spec.strip() and not atts:
            raise HTTPException(
                status_code=422, detail="provide a spec, a drawing, or both"
            )
        cfg = RunConfig.model_validate_json(config)
        handle = runs_mod.start_run(
            spec,
            cfg,
            app.state.runs_dir,
            drawings=atts,
            interpretation=interpretation,
        )
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

    @app.get("/api/runs/{run_id}/input/{name}")
    def input_artifact(run_id: str, name: str):
        return runs_mod.input_artifact(run_id, name, app.state.runs_dir)

    return app


app = create_app()
