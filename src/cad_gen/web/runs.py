"""Run lifecycle: launch generation on a worker thread and stream events.

`generate_cad` is async and drives a *blocking* CadQuery subprocess. Running it
on a dedicated worker thread (via ``asyncio.run``) keeps that blocking work off
the server's event loop. The synchronous ``on_iteration`` callback fires on the
worker thread and pushes events into a thread-safe ``queue.Queue``; the SSE
endpoint drains that queue without blocking the loop.
"""

import asyncio
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from fastapi.responses import FileResponse, Response

from cad_gen.imaging import drawing_filename
from cad_gen.models import DrawingAttachment, IterationRecord, RunConfig, RunResult
from cad_gen.orchestrator import generate_cad

_SENTINEL = object()
# friendly url key -> artifact filename
_ARTIFACTS = {"stl": "model.stl", "step": "model.step", "views": "views.png"}
_ARTIFACT_NAMES = tuple(_ARTIFACTS.values())
_SHOWCASE_CACHE = "showcase.json"
_SHOWCASE_MODEL = "fal-ai/flux-pro/kontext"
_SHOWCASE_TIMEOUT_S = 180


@dataclass
class RunHandle:
    run_id: str
    spec: str
    config: RunConfig
    drawings: list[DrawingAttachment] = field(default_factory=list)
    interpretation: str | None = None
    events: "queue.Queue[tuple[Any, Any]]" = field(default_factory=queue.Queue)
    status: str = "running"  # running | done | error
    run_dir: Path | None = None
    result: RunResult | None = None
    # index -> {artifact name -> absolute path}
    artifacts: dict[int, dict[str, Path]] = field(default_factory=dict)

    @property
    def drawing_names(self) -> list[str]:
        return [drawing_filename(i, d.media_type) for i, d in enumerate(self.drawings, 1)]


# Process-wide registry of live/finished runs (v1: in-memory, never evicted).
RUNS: dict[str, RunHandle] = {}


def start_run(
    spec: str,
    config: RunConfig,
    runs_dir: Path,
    *,
    drawings: list[DrawingAttachment] | None = None,
    interpretation: str | None = None,
) -> RunHandle:
    """Register a run and launch its generation worker thread."""
    run_id = uuid.uuid4().hex[:12]
    config = config.model_copy(update={"out_dir": runs_dir})
    handle = RunHandle(
        run_id=run_id,
        spec=spec,
        config=config,
        drawings=drawings or [],
        interpretation=interpretation,
    )
    RUNS[run_id] = handle
    threading.Thread(target=_worker, args=(handle,), daemon=True).start()
    return handle


def _worker(handle: RunHandle) -> None:
    def on_iteration(record: IterationRecord) -> None:
        _capture_artifacts(handle, record)
        handle.events.put(("iteration", record))

    try:
        result = asyncio.run(
            generate_cad(
                handle.spec,
                handle.config,
                drawings=handle.drawings,
                interpretation=handle.interpretation,
                on_iteration=on_iteration,
            )
        )
        handle.run_dir = result.run_dir
        handle.result = result
        handle.status = "done"
        handle.events.put(("result", result))
    except Exception as exc:  # noqa: BLE001 - surfaced to the client as an event
        handle.status = "error"
        handle.events.put(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        handle.events.put((_SENTINEL, None))


def _capture_artifacts(handle: RunHandle, record: IterationRecord) -> None:
    ex = record.execution
    if ex is None or not ex.success:
        return
    handle.artifacts[record.index] = {
        "model.stl": Path(ex.stl_path) if ex.stl_path else None,
        "model.step": Path(ex.step_path) if ex.step_path else None,
        "views.png": Path(record.render_path) if record.render_path else None,
    }


# --------------------------------------------------------------------------- #
# Event serialization (browser-facing; absolute fs paths stripped → URLs)
# --------------------------------------------------------------------------- #
def started_event(handle: RunHandle) -> dict:
    return {
        "type": "started",
        "run_id": handle.run_id,
        "spec": handle.spec,
        "config": handle.config.model_dump(mode="json"),
        "drawings": handle.drawing_names,
        "interpretation": handle.interpretation,
    }


def _strip_paths(record_data: dict) -> dict:
    """Remove absolute filesystem paths from a serialized IterationRecord."""
    if record_data.get("execution"):
        record_data["execution"].pop("stl_path", None)
        record_data["execution"].pop("step_path", None)
    record_data.pop("render_path", None)
    reprojection = record_data.get("reprojection")
    if reprojection:
        reprojection.pop("composite_path", None)
        for view in (reprojection.get("views") or {}).values():
            if isinstance(view, dict):
                view.pop("overlay_path", None)
    return record_data


def iteration_payload(run_id: str, record: IterationRecord) -> dict:
    """Cleaned record (no absolute fs paths) + browser-facing artifact URLs."""
    data = _strip_paths(record.model_dump(mode="json"))

    urls: dict[str, str] = {}
    ex = record.execution
    if ex is not None and ex.success:
        base = f"/api/runs/{run_id}/iterations/{record.index}"
        urls = {key: f"{base}/{name}" for key, name in _ARTIFACTS.items()}
    return {"record": data, "urls": urls}


def iteration_event(run_id: str, record: IterationRecord) -> dict:
    return {"type": "iteration", **iteration_payload(run_id, record)}


def result_event(run_id: str, result: RunResult) -> dict:
    data = result.model_dump(mode="json")
    data.pop("run_dir", None)
    _strip_paths(data["best"])
    for iteration in data["iterations"]:
        _strip_paths(iteration)
    return {"type": "result", "result": data, "run_dir": result.run_dir.name}


async def event_stream(handle: RunHandle):
    """Async generator of SSE payloads: started → iteration* → result|error."""
    loop = asyncio.get_running_loop()
    yield {"data": json.dumps(started_event(handle))}
    while True:
        kind, payload = await loop.run_in_executor(None, handle.events.get)
        if kind is _SENTINEL:
            break
        if kind == "iteration":
            event = iteration_event(handle.run_id, payload)
        elif kind == "result":
            event = result_event(handle.run_id, payload)
        elif kind == "error":
            event = {"type": "error", "message": payload}
        else:
            continue
        yield {"data": json.dumps(event)}


# --------------------------------------------------------------------------- #
# History + detail (finished runs read from disk; live runs from the registry)
# --------------------------------------------------------------------------- #
def _load_result(run_id: str, runs_dir: Path) -> RunResult:
    handle = RUNS.get(run_id)
    if handle is not None and handle.result is not None:
        return handle.result
    result_file = Path(runs_dir) / run_id / "run_result.json"
    if not result_file.exists():
        raise HTTPException(status_code=404, detail="run not found")
    return RunResult.model_validate_json(result_file.read_text())


def list_runs(runs_dir: Path) -> list[dict]:
    """Summaries of finished runs, newest first (run dirs are timestamp-named)."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    summaries = []
    for d in sorted((p for p in root.iterdir() if p.is_dir()), reverse=True):
        result_file = d / "run_result.json"
        if not result_file.exists():
            continue
        result = RunResult.model_validate_json(result_file.read_text())
        summaries.append(
            {
                "id": d.name,
                "spec": result.spec,
                "accepted": result.accepted,
                "score": result.best.effective_score,
                "n_iterations": len(result.iterations),
                "drawings": result.drawings,
                "created_at": d.stat().st_mtime,
            }
        )
    return summaries


def run_detail(run_id: str, runs_dir: Path) -> dict:
    result = _load_result(run_id, runs_dir)
    return {
        "id": run_id,
        "spec": result.spec,
        "accepted": result.accepted,
        "best_index": result.best.index,
        "drawings": result.drawings,
        "interpretation": result.interpretation,
        "iterations": [iteration_payload(run_id, r) for r in result.iterations],
    }


# --------------------------------------------------------------------------- #
# Optional fal.ai showcase image generation
# --------------------------------------------------------------------------- #
def showcase_detail(run_id: str, runs_dir: Path) -> dict:
    """Return an existing fal.ai showcase image for a run without generating one."""
    result = _load_result(run_id, runs_dir)
    cache = _showcase_cache_path(result, runs_dir)
    if not cache.exists():
        raise HTTPException(status_code=404, detail="showcase not generated")
    data = json.loads(cache.read_text())
    data["cached"] = True
    return data


async def generate_showcase(run_id: str, runs_dir: Path) -> dict:
    """Generate a product-style showcase image from the best CAD render via fal.ai."""
    if not os.environ.get("FAL_KEY"):
        raise HTTPException(status_code=503, detail="FAL_KEY is not configured")

    result = _load_result(run_id, runs_dir)
    cache = _showcase_cache_path(result, runs_dir)
    if cache.exists():
        data = json.loads(cache.read_text())
        data["cached"] = True
        return data

    render_path = _best_render_path(run_id, result, runs_dir)
    model = os.environ.get("CAD_GEN_SHOWCASE_MODEL", _SHOWCASE_MODEL)
    prompt = _showcase_prompt(result)

    try:
        data = await asyncio.to_thread(_call_fal_showcase, model, prompt, render_path)
    except ModuleNotFoundError as exc:
        if exc.name == "fal_client":
            raise HTTPException(
                status_code=503,
                detail="fal-client is not installed; install the web extra dependencies",
            ) from exc
        raise
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001 - external provider error for browser display
        raise HTTPException(status_code=502, detail=f"fal.ai generation failed: {exc}") from exc

    image = _first_image(data)
    payload = {
        "image": image,
        "model": model,
        "prompt": prompt,
        "source_render": f"/api/runs/{run_id}/iterations/{result.best.index}/views.png",
        "created_at": time.time(),
        "cached": False,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(payload, indent=2))
    return payload


def _call_fal_showcase(model: str, prompt: str, render_path: Path) -> dict:
    import fal_client

    image_url = fal_client.upload_file(str(render_path))
    result = fal_client.subscribe(
        model,
        arguments={
            "prompt": prompt,
            "image_url": image_url,
            "aspect_ratio": "16:9",
            "num_images": 1,
            "output_format": "jpeg",
            "guidance_scale": 3.5,
        },
        client_timeout=_SHOWCASE_TIMEOUT_S,
    )
    if not isinstance(result, dict):
        raise RuntimeError("fal.ai returned an unexpected response")
    return result


def _first_image(data: dict) -> dict:
    images = data.get("images")
    if not isinstance(images, list) or not images:
        raise HTTPException(status_code=502, detail="fal.ai returned no image")
    image = images[0]
    if not isinstance(image, dict) or not image.get("url"):
        raise HTTPException(status_code=502, detail="fal.ai returned an invalid image")
    return {
        "url": image["url"],
        "content_type": image.get("content_type"),
        "file_name": image.get("file_name"),
        "width": image.get("width"),
        "height": image.get("height"),
    }


def _showcase_prompt(result: RunResult) -> str:
    spec = (result.spec or "the generated CAD part").strip()
    if len(spec) > 700:
        spec = spec[:697].rstrip() + "..."
    metrics = result.best.execution.metrics if result.best.execution else None
    metric_line = ""
    if metrics is not None:
        bbox = " x ".join(f"{v:.1f}" for v in metrics.bbox_mm)
        metric_line = f" Approximate bounding box: {bbox} mm."
    return (
        "Create one clean product showcase image from this multi-view CAD render. "
        "Preserve the part geometry, proportions, holes, fillets, and mechanical details. "
        "Show a single centered three-quarter studio render on a neutral technical background. "
        "Use matte light-gray material with subtle edge highlights. No text, labels, dimensions, "
        f"extra parts, or logos. Design spec: {spec}.{metric_line}"
    )


def _showcase_cache_path(result: RunResult, runs_dir: Path) -> Path:
    run_dir = _result_run_dir(result, runs_dir)
    return run_dir / "final" / _SHOWCASE_CACHE


def _result_run_dir(result: RunResult, runs_dir: Path) -> Path:
    path = safe_artifact(Path(result.run_dir), runs_dir)
    if not path.is_dir():
        raise HTTPException(status_code=404, detail="run directory not found")
    return path


def _best_render_path(run_id: str, result: RunResult, runs_dir: Path) -> Path:
    if result.best.render_path is not None:
        return safe_artifact(Path(result.best.render_path), runs_dir)
    return safe_artifact(_disk_artifact_path(run_id, result.best.index, "views.png", runs_dir), runs_dir)


# --------------------------------------------------------------------------- #
# Artifact resolution
# --------------------------------------------------------------------------- #
def safe_artifact(path: Path, runs_dir: Path) -> Path:
    """Resolve `path` and ensure it lives under the runs root (traversal guard)."""
    resolved = Path(path).resolve()
    root = Path(runs_dir).resolve()
    if not resolved.is_relative_to(root):
        raise HTTPException(status_code=403, detail="artifact outside runs root")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return resolved


def iteration_artifact_path(run_id: str, index: int, name: str, runs_dir: Path) -> Path:
    """Locate an iteration artifact for a live run (registry) or a past run (disk)."""
    if name not in _ARTIFACT_NAMES:
        raise HTTPException(status_code=404, detail="unknown artifact")

    handle = RUNS.get(run_id)
    if handle is not None and index in handle.artifacts:
        path = handle.artifacts[index].get(name)
        if path is not None:
            return safe_artifact(path, runs_dir)

    return safe_artifact(_disk_artifact_path(run_id, index, name, runs_dir), runs_dir)


def input_artifact(run_id: str, name: str, runs_dir: Path) -> Response:
    """Serve an input drawing: from memory for a live run, from disk for a past run."""
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=404, detail="unknown artifact")

    handle = RUNS.get(run_id)
    if handle is not None and handle.drawings:
        by_name = dict(zip(handle.drawing_names, handle.drawings, strict=True))
        d = by_name.get(name)
        if d is not None:
            return Response(content=d.data, media_type=d.media_type)

    path = safe_artifact(Path(runs_dir) / run_id / "input" / name, runs_dir)
    return FileResponse(path)


def _disk_artifact_path(run_id: str, index: int, name: str, runs_dir: Path) -> Path:
    """Resolve an artifact for a finished run by reading its iteration.json."""
    iter_dir = Path(runs_dir) / run_id / f"iter_{index:02d}"
    if name == "views.png":
        return iter_dir / "views.png"
    record_file = iter_dir / "iteration.json"
    if not record_file.exists():
        raise HTTPException(status_code=404, detail="run not found")
    record = IterationRecord.model_validate_json(record_file.read_text())
    ex = record.execution
    src = ex.stl_path if (ex and name == "model.stl") else (ex.step_path if ex else None)
    if src is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    return Path(src)
