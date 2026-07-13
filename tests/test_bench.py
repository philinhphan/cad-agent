"""Offline tests for the CADGenBench harness (no API / no network).

Covers the deterministic seams: description.yaml parsing + generation filtering,
the sample -> cad-gen request mapping, run_sample's STEP layout (with
generate_cad stubbed), resume/skip behaviour, and the submission zip contract.
"""
import json
import zipfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from cad_gen.bench import cli as bench_cli
from cad_gen.bench.adapter import output_step_path, run_sample, sample_to_request
from cad_gen.bench.dataset import load_samples
from cad_gen.bench.submission import SubmissionMeta, write_submission_zip
from cad_gen.models import (
    Critique,
    ExecutionResult,
    GeometryMetrics,
    IterationRecord,
    RunConfig,
    RunResult,
)

runner = CliRunner()
FIXTURE_PNG = Path(__file__).parent / "fixtures" / "drawing.png"


def _make_sample(dir: Path, name: str, body: str, *, with_png: bool = False) -> Path:
    sample_dir = dir / name
    sample_dir.mkdir(parents=True)
    (sample_dir / "description.yaml").write_text(body)
    if with_png:
        (sample_dir / "input.png").write_bytes(FIXTURE_PNG.read_bytes())
    return sample_dir


def _fake_run_result(run_dir: Path, *, n_solids: int = 1, watertight: bool = True) -> RunResult:
    (run_dir / "final").mkdir(parents=True, exist_ok=True)
    (run_dir / "final" / "model.step").write_text("ISO-10303-21;\nfake step\n")
    record = IterationRecord(
        index=1,
        execution=ExecutionResult(
            success=True,
            code="result = cq.Workplane().box(1, 1, 1)",
            metrics=GeometryMetrics(
                volume_mm3=1.0, bbox_mm=(1.0, 1.0, 1.0), center_of_mass=(0.0, 0.0, 0.0),
                n_solids=n_solids, n_faces=6, is_watertight=watertight,
            ),
            duration_s=0.1,
        ),
        critique=Critique(matches_spec=True, score=9, issues=[], suggestions=[], summary="ok"),
    )
    return RunResult(accepted=True, spec="x", best=record, iterations=[record], run_dir=run_dir)


# ── dataset parsing ────────────────────────────────────────────────────────

def test_load_samples_parses_and_filters_generation(tmp_path):
    _make_sample(tmp_path, "101", "description: Reproduce the drawing.\ninput_files:\n  - input.png\n")
    _make_sample(tmp_path, "201", "description: Double the rib.\ntask_type: editing\ninput_files:\n  - input.step\n")
    (tmp_path / "README.md").write_text("not a sample")  # ignored

    samples = load_samples(tmp_path, task_type="generation")

    assert [s.name for s in samples] == ["101"]
    assert samples[0].description == "Reproduce the drawing."
    assert samples[0].task_type == "generation"  # defaulted (absent in yaml)
    assert samples[0].input_files == ["input.png"]


def test_load_samples_names_filter_and_all_types(tmp_path):
    _make_sample(tmp_path, "101", "description: a\n")
    _make_sample(tmp_path, "201", "description: b\ntask_type: editing\n")

    assert {s.name for s in load_samples(tmp_path, task_type=None)} == {"101", "201"}
    assert [s.name for s in load_samples(tmp_path, task_type=None, names=["201"])] == ["201"]


# ── sample -> request mapping ──────────────────────────────────────────────

def test_sample_to_request_attaches_drawing(tmp_path):
    sample_dir = _make_sample(
        tmp_path, "101", "description: Reproduce the drawing.\ninput_files:\n  - input.png\n",
        with_png=True,
    )
    sample = load_samples(tmp_path)[0]
    assert sample.image_path == sample_dir / "input.png"

    spec, drawings = sample_to_request(sample)

    assert "Reproduce the drawing." in spec
    assert len(drawings) == 1
    assert drawings[0].media_type == "image/png"
    assert drawings[0].data  # bytes loaded


def test_sample_to_request_text_only_when_no_image(tmp_path):
    _make_sample(tmp_path, "300", "description: A 10mm cube.\n")
    sample = load_samples(tmp_path)[0]

    spec, drawings = sample_to_request(sample)

    assert "10mm cube" in spec
    assert drawings == []


# ── run_sample: STEP layout, resume, errors (generate_cad stubbed) ─────────

async def test_run_sample_writes_output_step(tmp_path, monkeypatch):
    _make_sample(tmp_path / "inputs", "101", "description: a cube.\n")
    sample = load_samples(tmp_path / "inputs")[0]
    out_root = tmp_path / "results"

    async def fake_generate_cad(spec, config, *, drawings=None, **kw):
        return _fake_run_result(config.out_dir / "20260706_000000")

    monkeypatch.setattr("cad_gen.bench.adapter.generate_cad", fake_generate_cad)

    outcome = await run_sample(sample, config=RunConfig(), out_root=out_root)

    out_path = output_step_path(out_root, "101")
    assert out_path.is_file()
    assert outcome.step_written and outcome.valid_signal
    assert outcome.n_solids == 1 and outcome.is_watertight is True
    # cad-gen's trace lands beside the candidate, not at the folder root.
    assert (out_root / "101" / "cadgen").is_dir()


async def test_run_sample_flags_non_watertight(tmp_path, monkeypatch):
    _make_sample(tmp_path / "inputs", "101", "description: a cube.\n")
    sample = load_samples(tmp_path / "inputs")[0]

    async def fake(spec, config, *, drawings=None, **kw):
        return _fake_run_result(config.out_dir / "r", n_solids=2, watertight=False)

    monkeypatch.setattr("cad_gen.bench.adapter.generate_cad", fake)
    outcome = await run_sample(sample, config=RunConfig(), out_root=tmp_path / "results")

    assert outcome.step_written  # a candidate exists...
    assert not outcome.valid_signal  # ...but it fails the cheap validity signal


async def test_run_sample_resumes_existing(tmp_path, monkeypatch):
    _make_sample(tmp_path / "inputs", "101", "description: a cube.\n")
    sample = load_samples(tmp_path / "inputs")[0]
    out_root = tmp_path / "results"
    output_step_path(out_root, "101").parent.mkdir(parents=True)
    output_step_path(out_root, "101").write_text("existing")

    def boom(*a, **k):
        raise AssertionError("generate_cad must not run for an existing candidate")

    monkeypatch.setattr("cad_gen.bench.adapter.generate_cad", boom)
    outcome = await run_sample(sample, config=RunConfig(), out_root=out_root)

    assert outcome.skipped and outcome.step_written


async def test_run_sample_survives_generation_error(tmp_path, monkeypatch):
    _make_sample(tmp_path / "inputs", "101", "description: a cube.\n")
    sample = load_samples(tmp_path / "inputs")[0]

    async def fake(spec, config, *, drawings=None, **kw):
        raise RuntimeError("model exploded")

    monkeypatch.setattr("cad_gen.bench.adapter.generate_cad", fake)
    outcome = await run_sample(sample, config=RunConfig(), out_root=tmp_path / "results")

    assert not outcome.step_written
    assert outcome.error and "model exploded" in outcome.error


# ── submission zip contract ────────────────────────────────────────────────

def test_write_submission_zip_layout(tmp_path):
    results = tmp_path / "results"
    (results / "101").mkdir(parents=True)
    (results / "101" / "output.step").write_text("candidate")
    (results / "101" / "cadgen").mkdir()  # sibling trace — must NOT be packaged
    (results / "101" / "cadgen" / "run.log").write_text("noise")
    (results / "202").mkdir()  # empty folder => missing candidate

    meta = SubmissionMeta(submitter_name="Phi", submission_name="cad-gen v1", agree_to_publish=True)
    out_zip = tmp_path / "submission.zip"
    n_with, n_missing = write_submission_zip(results, meta, out_zip)

    assert (n_with, n_missing) == (1, 1)
    with zipfile.ZipFile(out_zip) as zf:
        names = set(zf.namelist())
        assert "meta.json" in names
        assert "101/output.step" in names
        assert "202/" in names  # preserved so the grader records it "missing"
        assert not any("cadgen" in n for n in names)  # trace excluded
        loaded = json.loads(zf.read("meta.json"))
    assert loaded["agree_to_publish"] is True
    assert set(loaded) == {"submitter_name", "submission_name", "agent_url", "notes", "agree_to_publish"}


def test_write_submission_zip_empty_dir_raises(tmp_path):
    empty = tmp_path / "results"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        write_submission_zip(empty, SubmissionMeta(submitter_name="p", submission_name="n"), tmp_path / "s.zip")


# ── CLI: package command (offline) ─────────────────────────────────────────

def test_package_cli_writes_zip(tmp_path):
    results = tmp_path / "results"
    (results / "101").mkdir(parents=True)
    (results / "101" / "output.step").write_text("candidate")

    result = runner.invoke(bench_cli.app, [
        "package", str(results), "--submitter", "Phi", "--name", "cad-gen v1", "--agree",
    ])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "results.zip").is_file()


def test_package_cli_warns_without_agree(tmp_path):
    results = tmp_path / "results"
    (results / "101").mkdir(parents=True)
    (results / "101" / "output.step").write_text("candidate")

    result = runner.invoke(bench_cli.app, [
        "package", str(results), "--submitter", "Phi", "--name", "cad-gen v1",
    ])

    assert result.exit_code == 0
    assert "agree_to_publish=false" in result.output
