"""cad-gen: agentic text-to-CAD with a visual self-refine loop."""

from cad_gen.models import (
    Critique,
    DrawingAttachment,
    ExecutionResult,
    GeometryMetrics,
    IterationRecord,
    ReprojectionReport,
    ReprojectionView,
    RunConfig,
    RunResult,
)
from cad_gen.orchestrator import generate_cad

__all__ = [
    "Critique",
    "DrawingAttachment",
    "ExecutionResult",
    "GeometryMetrics",
    "IterationRecord",
    "ReprojectionReport",
    "ReprojectionView",
    "RunConfig",
    "RunResult",
    "generate_cad",
]
