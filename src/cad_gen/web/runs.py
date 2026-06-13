"""Run lifecycle: launch generation on a worker thread and stream events.

`generate_cad` is async and drives a *blocking* CadQuery subprocess. Running it
on a dedicated worker thread (via ``asyncio.run``) keeps that blocking work off
the server's event loop. The synchronous ``on_iteration`` callback fires on the
worker thread and pushes events into a thread-safe ``queue.Queue``; the SSE
endpoint drains that queue without blocking the loop.
"""

import asyncio
import json
import queue
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from cad_gen.models import IterationRecord, RunConfig, RunResult
from cad_gen.orchestrator import generate_cad

_SENTINEL = object()
# friendly url key -> artifact filename
_ARTIFACTS = {"stl": "model.stl", "step": "model.step", "views": "views.png"}
_ARTIFACT_NAMES = tuple(_ARTIFACTS.values())


@dataclass
class RunHandle:
    run_id: str
    spec: str
    config: RunConfig
    events: "queue.Queue[tuple[Any, Any]]" = field(default_factory=queue.Queue)
    status: str = "running"  # running | done | error
    run_dir: Path | None = None
    result: RunResult | None = None
    # index -> {artifact name -> absolute path}
    artifacts: dict[int, dict[str, Path]] = field(default_factory=dict)


# Process-wide registry of live/finished runs (v1: in-memory, never evicted).
RUNS: dict[str, RunHandle] = {}


def start_run(spec: str, config: RunConfig, runs_dir: Path) -> RunHandle:
    """Register a run and launch its generation worker thread."""
    run_id = uuid.uuid4().hex[:12]
    config = config.model_copy(update={"out_dir": runs_dir})
    handle = RunHandle(run_id=run_id, spec=spec, config=config)
    RUNS[run_id] = handle
    threading.Thread(target=_worker, args=(handle,), daemon=True).start()
    return handle


def _worker(handle: RunHandle) -> None:
    def on_iteration(record: IterationRecord) -> None:
        _capture_artifacts(handle, record)
        handle.events.put(("iteration", record))

    try:
        result = asyncio.run(
            generate_cad(handle.spec, handle.config, on_iteration=on_iteration)
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
    }


def _strip_paths(record_data: dict) -> dict:
    """Remove absolute filesystem paths from a serialized IterationRecord."""
    if record_data.get("execution"):
        record_data["execution"].pop("stl_path", None)
        record_data["execution"].pop("step_path", None)
    record_data.pop("render_path", None)
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
        "iterations": [iteration_payload(run_id, r) for r in result.iterations],
    }


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
