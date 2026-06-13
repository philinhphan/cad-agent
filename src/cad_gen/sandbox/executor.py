"""Run generated CadQuery code in an isolated subprocess.

Isolation here means crash/timeout/state isolation, not a security
boundary — see README.
"""

import subprocess
import sys
import time
from pathlib import Path

from cad_gen.models import ExecutionResult, GeometryMetrics

HARNESS = Path(__file__).parent / "harness.py"

_TRACEBACK_TAIL_CHARS = 3000


def run_cad_code(code: str, out_dir: Path, timeout_s: float = 60) -> ExecutionResult:
    """Execute `code` via the harness subprocess; artifacts land in `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_file = out_dir / "model.py"
    code_file.write_text(code)

    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(HARNESS), str(code_file), str(out_dir)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=out_dir,
        )
    except subprocess.TimeoutExpired as exc:
        return ExecutionResult(
            success=False,
            code=code,
            error=(
                f"Execution timed out after {timeout_s:g}s "
                "(infinite loop or excessively complex geometry)."
            ),
            stdout=_coerce(exc.stdout),
            duration_s=time.monotonic() - start,
        )
    duration_s = time.monotonic() - start

    if proc.returncode != 0:
        return ExecutionResult(
            success=False,
            code=code,
            error=proc.stderr[-_TRACEBACK_TAIL_CHARS:] or f"exit code {proc.returncode}",
            stdout=proc.stdout,
            duration_s=duration_s,
        )

    metrics_file = out_dir / "metrics.json"
    stl_path = out_dir / "model.stl"
    step_path = out_dir / "model.step"
    if not (metrics_file.exists() and stl_path.exists()):
        return ExecutionResult(
            success=False,
            code=code,
            error="Harness exited cleanly but produced no artifacts.",
            stdout=proc.stdout,
            duration_s=duration_s,
        )

    metrics = GeometryMetrics.model_validate_json(metrics_file.read_text())
    metrics.is_watertight = _check_watertight(stl_path)
    metrics_file.write_text(metrics.model_dump_json(indent=2))

    return ExecutionResult(
        success=True,
        code=code,
        metrics=metrics,
        stl_path=stl_path,
        step_path=step_path,
        stdout=proc.stdout,
        duration_s=duration_s,
    )


def _check_watertight(stl_path: Path) -> bool | None:
    try:
        import trimesh

        mesh = trimesh.load(stl_path, force="mesh")
        return bool(mesh.is_watertight)
    except Exception:
        return None


def _coerce(stream: str | bytes | None) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode(errors="replace")
    return stream
