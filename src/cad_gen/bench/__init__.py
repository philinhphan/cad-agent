"""CADGenBench harness: run cad-gen over the benchmark and package submissions."""

from cad_gen.bench.adapter import SampleOutcome, run_all, run_sample, sample_to_request
from cad_gen.bench.dataset import BenchSample, load_samples, resolve_inputs_dir
from cad_gen.bench.submission import SubmissionMeta, write_submission_zip

__all__ = [
    "BenchSample",
    "SampleOutcome",
    "SubmissionMeta",
    "load_samples",
    "resolve_inputs_dir",
    "run_all",
    "run_sample",
    "sample_to_request",
    "write_submission_zip",
]
