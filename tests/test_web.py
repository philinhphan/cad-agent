"""Tests for the FastAPI web backend.

All tests run offline: generation is monkeypatched, so no LLM or CadQuery
subprocess is ever invoked (conftest sets ALLOW_MODEL_REQUESTS = False).
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cad_gen.models import (
    Critique,
    ExecutionResult,
    GeometryMetrics,
    IterationRecord,
    ReprojectionReport,
    ReprojectionView,
    RunResult,
)
from cad_gen.web.server import create_app


# --------------------------------------------------------------------------- #
# Phase 1: skeleton
# --------------------------------------------------------------------------- #
def test_health_ok():
    client = TestClient(create_app())

    resp = client.get("/api/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_cors_headers_present_for_allowed_origin():
    client = TestClient(create_app(origins=["http://localhost:3000"]))

    resp = client.get("/api/health", headers={"Origin": "http://localhost:3000"})

    assert resp.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_config_defaults_falls_back_to_builtins(monkeypatch):
    monkeypatch.delenv("CAD_GEN_MODEL", raising=False)
    monkeypatch.delenv("CAD_GEN_CRITIC_MODEL", raising=False)
    client = TestClient(create_app())

    body = client.get("/api/config/defaults").json()

    assert body == {
        "model": "openai:gpt-5.5",
        "critic_model": "google:gemini-3.5-flash",
    }


def test_config_defaults_reflect_env(monkeypatch):
    monkeypatch.setenv("CAD_GEN_MODEL", "anthropic:claude-opus-4-8")
    monkeypatch.setenv("CAD_GEN_CRITIC_MODEL", "anthropic:claude-opus-4-8")
    client = TestClient(create_app())

    body = client.get("/api/config/defaults").json()

    assert body == {
        "model": "anthropic:claude-opus-4-8",
        "critic_model": "anthropic:claude-opus-4-8",
    }


# --------------------------------------------------------------------------- #
# Phase 2: run lifecycle + SSE
# --------------------------------------------------------------------------- #
def _make_fake_generate(scores: list[int]):
    """A drop-in for generate_cad that writes tiny real artifacts and fires
    on_iteration per scored iteration. score <= 0 ⇒ a failed (no-geometry)
    iteration."""

    async def fake_generate_cad(spec, config=None, *, on_iteration=None, **kwargs):
        run_dir = Path(config.out_dir) / "run_fake"
        records: list[IterationRecord] = []
        for i, score in enumerate(scores, start=1):
            success = score > 0
            adir = run_dir / f"iter_{i:02d}" / "attempt_01"
            adir.mkdir(parents=True, exist_ok=True)
            stl = adir / "model.stl"
            step = adir / "model.step"
            views = run_dir / f"iter_{i:02d}" / "views.png"
            if success:
                stl.write_bytes(b"solid x\nendsolid x\n")
                step.write_bytes(b"ISO-10303-21;")
                views.write_bytes(b"\x89PNG\r\n\x1a\nfake")
            execution = ExecutionResult(
                success=success,
                code=f"result = cq.Workplane().box({i}, {i}, {i})",
                error=None if success else "NameError: boom",
                metrics=GeometryMetrics(
                    volume_mm3=float(i**3),
                    bbox_mm=(float(i), float(i), float(i)),
                    center_of_mass=(0.0, 0.0, 0.0),
                    n_solids=1,
                    n_faces=6,
                    is_watertight=True,
                )
                if success
                else None,
                stl_path=stl if success else None,
                step_path=step if success else None,
                duration_s=0.01,
            )
            critique = (
                Critique(
                    matches_spec=score >= config.score_threshold,
                    score=score,
                    issues=[] if score >= config.score_threshold else ["too small"],
                    suggestions=[],
                    summary=f"scored {score}",
                )
                if success
                else None
            )
            record = IterationRecord(
                index=i,
                execution=execution,
                render_path=views if success else None,
                critique=critique,
            )
            records.append(record)
            if on_iteration is not None:
                on_iteration(record)
        best = max(records, key=lambda r: (r.effective_score, r.index))
        return RunResult(
            accepted=best.effective_score >= config.score_threshold,
            spec=spec,
            best=best,
            iterations=records,
            run_dir=run_dir,
        )

    return fake_generate_cad


def _client(tmp_path, monkeypatch, scores):
    monkeypatch.setattr(
        "cad_gen.web.runs.generate_cad", _make_fake_generate(scores)
    )
    return TestClient(create_app(runs_dir=tmp_path))


def _read_events(client, run_id):
    events = []
    with client.stream("GET", f"/api/runs/{run_id}/events") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            line = line.strip()
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))
    return events


def test_start_run_streams_started_iterations_result(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, scores=[5, 9])

    run_id = client.post("/api/runs", data={"spec": "a cube"}).json()["run_id"]
    events = _read_events(client, run_id)

    types = [e["type"] for e in events]
    assert types == ["started", "iteration", "iteration", "result"]

    assert events[0]["spec"] == "a cube"
    assert events[1]["record"]["index"] == 1
    assert events[1]["record"]["critique"]["score"] == 5
    assert events[2]["record"]["critique"]["score"] == 9
    # url rewriting: no absolute fs paths leak; browser-facing urls provided
    assert "stl_path" not in events[2]["record"]["execution"]
    assert events[2]["urls"]["stl"] == f"/api/runs/{run_id}/iterations/2/model.stl"
    # terminal result
    assert events[3]["result"]["accepted"] is True
    assert events[3]["result"]["best"]["index"] == 2


def test_failed_iteration_streams_with_no_urls(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, scores=[0, 9])

    run_id = client.post("/api/runs", data={"spec": "x"}).json()["run_id"]
    events = _read_events(client, run_id)

    first = events[1]
    assert first["record"]["execution"]["success"] is False
    assert first["record"]["critique"] is None
    assert first["urls"] == {}


def test_config_overrides_reach_run(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, scores=[9])

    run_id = client.post(
        "/api/runs",
        data={
            "spec": "x",
            "config": json.dumps({"max_iterations": 3, "score_threshold": 9}),
        },
    ).json()["run_id"]
    events = _read_events(client, run_id)

    assert events[0]["config"]["max_iterations"] == 3
    assert events[0]["config"]["score_threshold"] == 9


def test_live_iteration_artifact_is_served(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, scores=[9])

    run_id = client.post("/api/runs", data={"spec": "x"}).json()["run_id"]
    _read_events(client, run_id)  # drain so the run completes

    resp = client.get(f"/api/runs/{run_id}/iterations/1/model.stl")

    assert resp.status_code == 200
    assert b"endsolid" in resp.content


def test_safe_artifact_rejects_paths_outside_runs_root(tmp_path):
    from fastapi import HTTPException

    from cad_gen.web.runs import safe_artifact

    inside = tmp_path / "run" / "model.stl"
    inside.parent.mkdir(parents=True)
    inside.write_bytes(b"ok")
    assert safe_artifact(inside, tmp_path) == inside.resolve()

    with pytest.raises(HTTPException):
        safe_artifact(Path("/etc/passwd"), tmp_path)


# --------------------------------------------------------------------------- #
# Phase 3: history + detail + past-run artifacts
# --------------------------------------------------------------------------- #
def _write_disk_run(runs_dir: Path, name: str, scores: list[int]) -> Path:
    """Materialize a finished run on disk the way the orchestrator would."""
    run_dir = runs_dir / name
    records: list[IterationRecord] = []
    for i, score in enumerate(scores, start=1):
        adir = run_dir / f"iter_{i:02d}" / "attempt_01"
        adir.mkdir(parents=True)
        stl = adir / "model.stl"
        stl.write_bytes(b"solid s\nendsolid s\n")
        step = adir / "model.step"
        step.write_bytes(b"ISO-10303-21;")
        views = run_dir / f"iter_{i:02d}" / "views.png"
        views.write_bytes(b"\x89PNG\r\n\x1a\nfake")
        rec = IterationRecord(
            index=i,
            execution=ExecutionResult(
                success=True,
                code=f"result = cq.Workplane().box({i}, {i}, {i})",
                metrics=GeometryMetrics(
                    volume_mm3=float(i**3),
                    bbox_mm=(float(i), float(i), float(i)),
                    center_of_mass=(0.0, 0.0, 0.0),
                    n_solids=1,
                    n_faces=6,
                    is_watertight=True,
                ),
                stl_path=stl,
                step_path=step,
                duration_s=0.01,
            ),
            render_path=views,
            critique=Critique(
                matches_spec=score >= 8,
                score=score,
                issues=[],
                suggestions=[],
                summary=f"scored {score}",
            ),
        )
        (run_dir / f"iter_{i:02d}" / "iteration.json").write_text(rec.model_dump_json())
        records.append(rec)
    best = max(records, key=lambda r: (r.effective_score, r.index))
    result = RunResult(
        accepted=best.effective_score >= 8,
        spec=f"spec for {name}",
        best=best,
        iterations=records,
        run_dir=run_dir,
    )
    (run_dir / "spec.txt").write_text(result.spec)
    (run_dir / "run_result.json").write_text(result.model_dump_json())
    return run_dir


def test_list_runs_returns_summaries_newest_first(tmp_path):
    _write_disk_run(tmp_path, "20260613_120000", [4, 7])
    _write_disk_run(tmp_path, "20260613_130000", [9])
    client = TestClient(create_app(runs_dir=tmp_path))

    runs = client.get("/api/runs").json()

    assert [r["id"] for r in runs] == ["20260613_130000", "20260613_120000"]
    newest = runs[0]
    assert newest["spec"] == "spec for 20260613_130000"
    assert newest["accepted"] is True
    assert newest["score"] == 9
    assert runs[1]["n_iterations"] == 2


def test_get_run_detail_with_artifact_urls(tmp_path):
    _write_disk_run(tmp_path, "20260613_140000", [5, 9])
    client = TestClient(create_app(runs_dir=tmp_path))

    detail = client.get("/api/runs/20260613_140000").json()

    assert detail["id"] == "20260613_140000"
    assert detail["accepted"] is True
    assert detail["best_index"] == 2
    assert len(detail["iterations"]) == 2
    it2 = detail["iterations"][1]
    assert it2["record"]["critique"]["score"] == 9
    assert "stl_path" not in it2["record"]["execution"]
    assert it2["urls"]["stl"] == "/api/runs/20260613_140000/iterations/2/model.stl"


def test_past_run_artifacts_served_from_disk(tmp_path):
    _write_disk_run(tmp_path, "20260613_150000", [9])
    client = TestClient(create_app(runs_dir=tmp_path))

    stl = client.get("/api/runs/20260613_150000/iterations/1/model.stl")
    views = client.get("/api/runs/20260613_150000/iterations/1/views.png")

    assert stl.status_code == 200 and b"endsolid" in stl.content
    assert views.status_code == 200 and views.content.startswith(b"\x89PNG")


def test_get_unknown_run_404(tmp_path):
    client = TestClient(create_app(runs_dir=tmp_path))

    assert client.get("/api/runs/nope").status_code == 404


# --------------------------------------------------------------------------- #
# Phase 4: drawing input (multipart upload + interpret + input artifacts)
# --------------------------------------------------------------------------- #
PNG = b"\x89PNG\r\n\x1a\nfakepngdata"


def test_interpret_endpoint_returns_digest(tmp_path, monkeypatch):
    async def fake_interpret(agent, *, spec, drawings):
        assert len(drawings) == 1 and drawings[0].media_type == "image/png"
        return "ENVELOPE 85 x 135 x 25, Ø15 THRU"

    monkeypatch.setattr("cad_gen.web.server.interpret_drawing", fake_interpret)
    client = TestClient(create_app(runs_dir=tmp_path))

    resp = client.post(
        "/api/drawings/interpret",
        data={"spec": ""},
        files=[("drawings", ("d.png", PNG, "image/png"))],
    )

    assert resp.status_code == 200
    assert resp.json()["interpretation"] == "ENVELOPE 85 x 135 x 25, Ø15 THRU"


def test_start_run_multipart_forwards_drawing(tmp_path, monkeypatch):
    captured: dict = {}

    async def fake_generate_cad(spec, config=None, *, on_iteration=None, **kwargs):
        captured["spec"] = spec
        captured["drawings"] = kwargs.get("drawings")
        captured["interpretation"] = kwargs.get("interpretation")
        rec = IterationRecord(
            index=1,
            execution=ExecutionResult(success=False, code="x", duration_s=0.0),
        )
        return RunResult(
            accepted=False, spec=spec, best=rec, iterations=[rec], run_dir=tmp_path / "r"
        )

    monkeypatch.setattr("cad_gen.web.runs.generate_cad", fake_generate_cad)
    client = TestClient(create_app(runs_dir=tmp_path))

    resp = client.post(
        "/api/runs",
        data={"spec": "", "interpretation": "USER DIMS"},
        files=[("drawings", ("orig.png", PNG, "image/png"))],
    )
    assert resp.status_code == 200
    _read_events(client, resp.json()["run_id"])  # wait for the worker thread to finish

    assert captured["spec"] == ""
    assert captured["interpretation"] == "USER DIMS"
    assert captured["drawings"] is not None and len(captured["drawings"]) == 1
    assert captured["drawings"][0].media_type == "image/png"


def test_reject_non_image_upload(tmp_path):
    client = TestClient(create_app(runs_dir=tmp_path))

    resp = client.post(
        "/api/runs",
        data={"spec": ""},
        files=[("drawings", ("notes.txt", b"just text", "text/plain"))],
    )

    assert resp.status_code == 422


def test_start_run_requires_spec_or_drawing(tmp_path):
    client = TestClient(create_app(runs_dir=tmp_path))

    assert client.post("/api/runs", data={"spec": ""}).status_code == 422


def test_input_artifact_served_from_disk(tmp_path):
    run_dir = tmp_path / "20260613_160000"
    (run_dir / "input").mkdir(parents=True)
    (run_dir / "input" / "drawing_01.png").write_bytes(PNG)
    client = TestClient(create_app(runs_dir=tmp_path))

    resp = client.get("/api/runs/20260613_160000/input/drawing_01.png")

    assert resp.status_code == 200
    assert resp.content == PNG


def test_input_artifact_served_from_memory_for_live_run(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch, scores=[9])

    run_id = client.post(
        "/api/runs",
        data={"spec": "x"},
        files=[("drawings", ("orig.jpg", PNG, "image/jpeg"))],
    ).json()["run_id"]
    resp = client.get(f"/api/runs/{run_id}/input/drawing_01.jpg")

    assert resp.status_code == 200
    assert resp.content == PNG
    assert resp.headers["content-type"].startswith("image/jpeg")


def test_input_artifact_traversal_rejected(tmp_path):
    from fastapi import HTTPException

    from cad_gen.web.runs import input_artifact

    with pytest.raises(HTTPException):
        input_artifact("somerun", "../../etc/passwd", tmp_path)


def test_strip_paths_removes_reprojection_fs_paths():
    """Absolute overlay/composite paths must not leak to the browser; the digest stays."""
    from cad_gen.web.runs import _strip_paths

    rec = IterationRecord(
        index=1,
        execution=ExecutionResult(
            success=True,
            code="x",
            stl_path=Path("/abs/model.stl"),
            step_path=Path("/abs/model.step"),
            duration_s=0.1,
        ),
        reprojection=ReprojectionReport(
            evaluated=True,
            digest="front view: 90% reproduced",
            views={
                "front": ReprojectionView(
                    coverage=0.9, chamfer_pct=1.0, aspect_ok=True, aspect_rel_err=0.0,
                    aspect_signed=0.0, overlay_path=Path("/abs/overlay_front.png"),
                )
            },
            composite_path=Path("/abs/overlay_composite.png"),
        ),
    )

    data = _strip_paths(rec.model_dump(mode="json"))

    assert "composite_path" not in data["reprojection"]
    assert "overlay_path" not in data["reprojection"]["views"]["front"]
    assert data["reprojection"]["digest"] == "front view: 90% reproduced"
