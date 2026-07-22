"""CADGenBench harness: run cad-gen over the benchmark and package submissions."""

from cad_gen.bench.adapter import (
    SampleOutcome,
    ensure_all_sample_dirs,
    run_all,
    run_sample,
    sample_to_edit_request,
    sample_to_request,
)
from cad_gen.bench.dataset import BenchSample, load_samples, resolve_inputs_dir
from cad_gen.bench.submission import SubmissionMeta, write_submission_zip

__all__ = [
    "BenchSample",
    "SampleOutcome",
    "SubmissionMeta",
    "ensure_all_sample_dirs",
    "load_samples",
    "resolve_inputs_dir",
    "run_all",
    "run_sample",
    "sample_to_edit_request",
    "sample_to_request",
    "write_submission_zip",
]
