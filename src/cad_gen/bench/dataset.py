"""Load CADGenBench fixture inputs from the HuggingFace Hub dataset.

CADGenBench inputs live in the public dataset repo
``HuggingAI4Engineering/cadgenbench-data``; each top-level entry is a sample
directory holding ``description.yaml`` plus optional attachments
(``input.png`` for generation, ``input.step`` for editing). We resolve the
dataset the same way cadgenbench itself does — ``snapshot_download`` into the
local Hub cache — then parse each ``description.yaml`` into a small typed
``BenchSample``. See ``cadgenbench/src/cadgenbench/common/paths.py``.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

# Public inputs dataset (descriptions + drawings). The private ground-truth repo
# is never needed locally — scoring happens server-side on the leaderboard Space.
DEFAULT_DATA_REPO = "HuggingAI4Engineering/cadgenbench-data"
_ENV_DATA_REPO = "CADGENBENCH_DATA_REPO"

# Image attachments we can feed to cad-gen's drawing mode.
_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
# Base-model attachments for editing samples.
_STEP_SUFFIXES = {".step", ".stp"}


class BenchSample(BaseModel):
    """One CADGenBench fixture: its prompt, task type, and input attachments."""

    name: str  # sample directory name, e.g. "101" — becomes the submission folder
    description: str  # the natural-language prompt
    task_type: str = "generation"  # "generation" | "editing" (default per the spec)
    input_files: list[str] = []  # attachment filenames relative to `dir`
    dir: Path  # the sample directory on disk

    @property
    def image_path(self) -> Path | None:
        """First existing image attachment (the engineering drawing), if any."""
        for name in self.input_files:
            candidate = self.dir / name
            if candidate.suffix.lower() in _IMAGE_SUFFIXES and candidate.is_file():
                return candidate
        return None

    @property
    def step_path(self) -> Path | None:
        """First existing STEP attachment (the base model to edit), if any."""
        for name in self.input_files:
            candidate = self.dir / name
            if candidate.suffix.lower() in _STEP_SUFFIXES and candidate.is_file():
                return candidate
        return None

    @property
    def render_paths(self) -> list[Path]:
        """Preview renders of the base model, from the sample's ``renders/`` dir.

        Editing samples ship ``renders/{front,iso,right,top}.png`` — previews of
        ``input.step`` used as before-state context for the editing generator/critic.
        Returns them sorted by name; empty when there is no ``renders/`` dir.
        """
        renders = self.dir / "renders"
        if not renders.is_dir():
            return []
        return sorted(p for p in renders.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)


def resolve_inputs_dir(repo: str | None = None) -> Path:
    """Snapshot-download the inputs dataset and return its local directory.

    Resolution mirrors cadgenbench: an explicit `repo`, else
    ``$CADGENBENCH_DATA_REPO``, else :data:`DEFAULT_DATA_REPO`. The returned
    directory's immediate children are the sample directories. ``huggingface_hub``
    is imported lazily so the ``[bench]`` extra is only needed at call time.
    """
    import os

    from huggingface_hub import snapshot_download

    repo_id = repo or os.environ.get(_ENV_DATA_REPO) or DEFAULT_DATA_REPO
    return Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))


def load_samples(
    inputs_dir: Path,
    *,
    task_type: str | None = "generation",
    names: list[str] | None = None,
) -> list[BenchSample]:
    """Parse every ``<sample>/description.yaml`` under `inputs_dir`.

    A sample is any immediate subdirectory containing ``description.yaml``
    (other Hub-repo files such as ``README.md`` or ``sanity_check_submission.py``
    are ignored). Pass `task_type` to keep only that family (default
    ``"generation"``; ``None`` keeps all), and `names` to restrict to specific
    sample ids. Returns samples sorted by name.
    """
    wanted = set(names) if names else None
    samples: list[BenchSample] = []
    for child in sorted(inputs_dir.iterdir()):
        desc_file = child / "description.yaml"
        if not child.is_dir() or not desc_file.is_file():
            continue
        if wanted is not None and child.name not in wanted:
            continue
        raw = yaml.safe_load(desc_file.read_text()) or {}
        sample = BenchSample(
            name=child.name,
            description=str(raw.get("description", "")).strip(),
            task_type=str(raw.get("task_type", "generation")),
            input_files=list(raw.get("input_files", []) or []),
            dir=child,
        )
        if task_type is not None and sample.task_type != task_type:
            continue
        samples.append(sample)
    return samples
