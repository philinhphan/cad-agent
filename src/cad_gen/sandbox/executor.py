"""Run generated CAD code in an isolated subprocess.

Isolation here means crash/timeout/state isolation, not a security
boundary — see README.

Each supported CAD library gets its own harness pair rather than one harness that branches
internally: a harness imports exactly one CAD library, so a single file would drag both
kernels into every subprocess.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from cad_gen.models import (
    DEFAULT_LIBRARY,
    CadLibrary,
    ExecutionResult,
    GeometryMetrics,
    IntrospectionResult,
)

_SANDBOX_DIR = Path(__file__).parent

# library -> (execute+export harness, read-only introspection harness). Both scripts in a
# pair honour the same CLI contract, so nothing above this module varies by library.
_HARNESSES: dict[CadLibrary, tuple[Path, Path]] = {
    "cadquery": (_SANDBOX_DIR / "harness.py", _SANDBOX_DIR / "introspect.py"),
    "build123d": (
        _SANDBOX_DIR / "harness_build123d.py",
        _SANDBOX_DIR / "introspect_build123d.py",
    ),
}

# Kept for backwards compatibility with callers that referenced the CadQuery harness paths
# directly before the library became selectable.
HARNESS, INTROSPECT_HARNESS = _HARNESSES[DEFAULT_LIBRARY]

_TRACEBACK_TAIL_CHARS = 3000


def _harnesses(library: CadLibrary) -> tuple[Path, Path]:
    """Return the (execution, introspection) harness pair for `library`."""
    try:
        return _HARNESSES[library]
    except KeyError:
        raise ValueError(
            f"Unknown CAD library {library!r}; expected one of {sorted(_HARNESSES)}"
        ) from None


def run_cad_code(
    code: str,
    out_dir: Path,
    timeout_s: float = 60,
    *,
    seed_files: dict[str, bytes] | None = None,
    library: CadLibrary = DEFAULT_LIBRARY,
) -> ExecutionResult:
    """Execute `code` via the harness subprocess; artifacts land in `out_dir`.

    `library` selects which harness runs the code, and therefore which CAD library `code`
    is expected to be written in.

    `seed_files` (name -> bytes) are written into `out_dir` before the subprocess
    runs — the harness executes with cwd=out_dir, so editing code can load a seeded
    base model with e.g. ``cq.importers.importStep("input.step")`` (CadQuery) or
    ``import_step("input.step")`` (build123d).
    """
    harness, _ = _harnesses(library)
    # Resolve before anything else: the subprocess runs with cwd=out_dir, so
    # relative paths in its argv would resolve against the wrong base.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (seed_files or {}).items():
        (out_dir / name).write_bytes(data)
    code_file = out_dir / "model.py"
    code_file.write_text(code)

    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(harness), str(code_file), str(out_dir)],
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


def introspect_cad_code(
    code: str,
    query: dict,
    out_dir: Path,
    timeout_s: float = 30,
    *,
    seed_files: dict[str, bytes] | None = None,
    library: CadLibrary = DEFAULT_LIBRARY,
) -> IntrospectionResult:
    """Probe the geometry `code` builds WITHOUT exporting (see introspect*.py).

    `query` is {"mode": "describe"} for either library; the selection probe is
    library-specific because the two select geometry differently:
    {"mode": "selector", "target": "edges"|"faces", "selector": "<sel>"} for CadQuery,
    {"mode": "selection", "expression": "<expr>"} for build123d.

    Returns ok=False with the traceback when the *code* fails to build; a bad
    *selector/expression* comes back ok=True with the diagnostic inside `data`.

    `seed_files` mirrors :func:`run_cad_code` — a probe over editing code must see
    the same seeded base model (e.g. ``input.step``) the real execution does.
    """
    _, introspect_harness = _harnesses(library)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (seed_files or {}).items():
        (out_dir / name).write_bytes(data)
    code_file = out_dir / "model.py"
    code_file.write_text(code)
    query_file = out_dir / "query.json"
    query_file.write_text(json.dumps(query))

    try:
        proc = subprocess.run(
            [sys.executable, str(introspect_harness), str(code_file), str(query_file)],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=out_dir,
        )
    except subprocess.TimeoutExpired:
        return IntrospectionResult(
            ok=False, error=f"Introspection timed out after {timeout_s:g}s."
        )

    if proc.returncode != 0:
        return IntrospectionResult(
            ok=False,
            error=proc.stderr[-_TRACEBACK_TAIL_CHARS:] or f"exit code {proc.returncode}",
        )
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return IntrospectionResult(
            ok=False, error=f"Introspection produced no parseable output: {proc.stdout[:500]!r}"
        )
    return IntrospectionResult(ok=True, data=data)


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
