"""cad-gen: agentic text-to-CAD with a visual self-refine loop."""

from cad_gen.models import (
    Critique,
    ExecutionResult,
    GeometryMetrics,
    IterationRecord,
    RunConfig,
    RunResult,
)
from cad_gen.orchestrator import generate_cad

__all__ = [
    "Critique",
    "ExecutionResult",
    "GeometryMetrics",
    "IterationRecord",
    "RunConfig",
    "RunResult",
    "generate_cad",
]
