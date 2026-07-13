"""Assemble a CADGenBench leaderboard submission zip (stdlib only).

The submission contract (``cadgenbench/docs/benchmark/submission.md`` and
``baseline/package.py``): a top-level ``meta.json`` plus one folder per sample,
each holding ``output.step`` (a folder with no candidate is recorded ``missing``
and scored 0). The packager is tool-agnostic — it copies whatever
``<sample>/output.step`` exists — so we replicate its ~20 lines here rather than
depend on the heavy ``cadgenbench`` install.
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from pydantic import BaseModel

# Candidate filenames the grader accepts, in priority order (mirrors package.py).
_CANDIDATE_NAMES = ("output.step", "output.stp")


class SubmissionMeta(BaseModel):
    """Top-level ``meta.json``. The leaderboard rejects the zip until agreed."""

    submitter_name: str
    submission_name: str
    agent_url: str | None = None
    notes: str | None = None
    agree_to_publish: bool = False


def _discover_candidates(results_dir: Path) -> list[tuple[str, Path | None]]:
    """Return ``(sample_name, candidate_path_or_none)`` for each sample folder.

    A sample is any immediate subdirectory; its candidate is the first non-empty
    ``output.*`` at the folder root (sibling debug dirs like ``cadgen/`` are
    ignored). A folder with no candidate is kept so the grader records it missing.
    """
    found: list[tuple[str, Path | None]] = []
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        candidate: Path | None = None
        for name in _CANDIDATE_NAMES:
            path = child / name
            if path.is_file() and path.stat().st_size > 0:
                candidate = path
                break
        found.append((child.name, candidate))
    return found


def write_submission_zip(
    results_dir: Path, meta: SubmissionMeta, out_path: Path
) -> tuple[int, int]:
    """Write `out_path` from the candidates under `results_dir`.

    Returns ``(n_with_candidate, n_missing)``. Raises ``FileNotFoundError`` if
    `results_dir` has no sample folders.
    """
    candidates = _discover_candidates(results_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No sample folders under {results_dir}. Run `cad-gen-bench run` first."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("meta.json", json.dumps(meta.model_dump(), indent=2) + "\n")
        for sample_name, candidate in candidates:
            # Explicit directory entry preserves missing-candidate folders on extract.
            zf.writestr(f"{sample_name}/", "")
            if candidate is not None:
                zf.write(candidate, arcname=f"{sample_name}/{candidate.name}")

    n_with = sum(1 for _, c in candidates if c is not None)
    return n_with, len(candidates) - n_with
