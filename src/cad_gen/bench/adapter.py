"""Drive cad-gen over CADGenBench samples and lay out submission candidates.

For each generation sample we build a cad-gen request (text spec + optional
drawing), run the self-refine loop via :func:`cad_gen.generate_cad`, and copy
the winning STEP to ``<out_root>/<sample>/output.step`` — the layout the
CADGenBench packager and grader expect. cad-gen's full run trace is kept
alongside under ``<out_root>/<sample>/cadgen/`` for debugging; only
``output.step`` is ever packaged.
"""
from __future__ import annotations

import asyncio
import shutil
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel

from cad_gen.bench.dataset import BenchSample
from cad_gen.imaging import media_type_for
from cad_gen.models import DrawingAttachment, RunConfig
from cad_gen.orchestrator import generate_cad

CANDIDATE_NAME = "output.step"


class SampleOutcome(BaseModel):
    """Result of attempting one sample — drives the summary table + validity signal."""

    name: str
    task_type: str
    step_written: bool = False
    skipped: bool = False  # candidate already existed (resume); generation not re-run
    accepted: bool = False  # cad-gen's own critic accepted (score >= threshold)
    score: int = 0  # cad-gen critic score (0-10), best iteration
    n_solids: int | None = None
    is_watertight: bool | None = None
    out_path: Path | None = None
    error: str | None = None

    @property
    def valid_signal(self) -> bool:
        """Cheap proxy for the benchmark validity gate (watertight single solid).

        Uses metrics cad-gen already computed — not the exact CADGenBench gate,
        but a candidate that fails this almost certainly fails there too.
        """
        return self.step_written and self.n_solids == 1 and self.is_watertight is not False


def output_step_path(out_root: Path, sample_name: str) -> Path:
    """Where a sample's submission candidate lives: ``<out_root>/<name>/output.step``."""
    return out_root / sample_name / CANDIDATE_NAME


def sample_to_request(sample: BenchSample) -> tuple[str, list[DrawingAttachment]]:
    """Map a CADGenBench sample onto a cad-gen generation request.

    Returns ``(spec, drawings)`` for ``generate_cad(spec, ..., drawings=drawings)``.
    Loading the drawing bytes into a ``DrawingAttachment`` is plumbing (done
    below); the load-bearing choice is how the benchmark ``description`` becomes
    the cad-gen ``spec`` — see the marked block.
    """
    drawings: list[DrawingAttachment] = []
    image = sample.image_path
    if image is not None:
        data = image.read_bytes()
        media = media_type_for(image.name, data)
        if media is not None:
            drawings.append(
                DrawingAttachment(filename=image.name, media_type=media, data=data)
            )

    # ── DESIGN CHOICE: benchmark framing ───────────────────────────────────
    # CADGenBench generation descriptions are terse and identical across samples
    # ("Reproduce the geometry as accurately as possible from the drawing."), so
    # the drawing carries all the signal and this preamble is cad-gen's only
    # textual guidance. It encodes the benchmark's conventions the generator
    # would otherwise not know: millimetres, drawing-authoritative, reproduce
    # every feature (no simplification), and a single watertight solid (the
    # validity gate scores multi-solid / non-watertight parts 0). Edit freely —
    # this wording shapes all generation samples.
    if drawings:
        spec = (
            f"{sample.description}\n\n"
            "The attached engineering drawing is the authoritative specification: "
            "reproduce every feature and dimension exactly as drawn, and do not add, "
            "omit, or simplify any feature. All dimensions are in millimetres. Produce "
            "a single watertight solid (one closed manifold), correctly filleted and "
            "chamfered where the drawing shows it."
        )
    else:
        spec = sample.description
    # ───────────────────────────────────────────────────────────────────────
    return spec, drawings


async def run_sample(
    sample: BenchSample,
    *,
    config: RunConfig,
    out_root: Path,
    overwrite: bool = False,
) -> SampleOutcome:
    """Generate one candidate and copy its STEP into the submission layout."""
    out_path = output_step_path(out_root, sample.name)
    if out_path.exists() and not overwrite:
        return SampleOutcome(
            name=sample.name, task_type=sample.task_type, skipped=True,
            step_written=True, out_path=out_path,
        )

    spec, drawings = sample_to_request(sample)
    # cad-gen writes its timestamped run dir under out_dir; keep it beside the
    # candidate so the trace (iterations, report.md, views.png) is easy to find.
    sample_config = config.model_copy(update={"out_dir": out_root / sample.name / "cadgen"})

    try:
        result = await generate_cad(spec, sample_config, drawings=drawings)
    except Exception as exc:  # one sample's failure must not abort the batch
        return SampleOutcome(
            name=sample.name, task_type=sample.task_type, error=f"{type(exc).__name__}: {exc}",
        )

    outcome = SampleOutcome(
        name=sample.name,
        task_type=sample.task_type,
        accepted=result.accepted,
        score=result.best.effective_score,
    )
    execution = result.best.execution
    if execution is not None and execution.metrics is not None:
        outcome.n_solids = execution.metrics.n_solids
        outcome.is_watertight = execution.metrics.is_watertight

    step_src = _locate_step(result.run_dir, execution.step_path if execution else None)
    if step_src is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(step_src, out_path)
        outcome.step_written = True
        outcome.out_path = out_path
    return outcome


def _locate_step(run_dir: Path, step_path: Path | None) -> Path | None:
    """Prefer the persisted ``final/model.step``; fall back to the raw export."""
    final = run_dir / "final" / "model.step"
    if final.exists():
        return final
    if step_path is not None and Path(step_path).exists():
        return Path(step_path)
    return None


async def run_all(
    samples: list[BenchSample],
    *,
    config: RunConfig,
    out_root: Path,
    parallel: int = 3,
    overwrite: bool = False,
    on_result: Callable[[SampleOutcome], None] | None = None,
) -> list[SampleOutcome]:
    """Run every sample with bounded concurrency, returning outcomes in input order.

    Resumable: a sample whose ``output.step`` already exists is skipped unless
    `overwrite`. `on_result` is invoked (in completion order) as each finishes,
    for live progress reporting.
    """
    semaphore = asyncio.Semaphore(max(1, parallel))

    async def _guarded(sample: BenchSample) -> SampleOutcome:
        async with semaphore:
            outcome = await run_sample(
                sample, config=config, out_root=out_root, overwrite=overwrite
            )
        if on_result is not None:
            on_result(outcome)
        return outcome

    return await asyncio.gather(*(_guarded(s) for s in samples))
