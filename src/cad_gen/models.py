"""Pydantic schemas shared across the sandbox, agents, and orchestrator."""

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

DEFAULT_MODEL = "openai:gpt-5.5"  # generator
DEFAULT_CRITIC_MODEL = "google:gemini-3.5-flash"  # vision critic


class DrawingAttachment(BaseModel):
    """An input engineering drawing supplied alongside (or instead of) a text spec.

    Carried at the call boundary only — the raw bytes are persisted to disk under the
    run directory, never embedded in `run_result.json` (which keeps only filenames).
    """

    filename: str
    media_type: str  # "image/jpeg" | "image/png"
    data: bytes


class GeometryMetrics(BaseModel):
    """Measured properties of an executed CAD model."""

    volume_mm3: float
    bbox_mm: tuple[float, float, float]
    center_of_mass: tuple[float, float, float]
    n_solids: int
    n_faces: int
    is_watertight: bool | None = None


class ExecutionResult(BaseModel):
    """Outcome of running generated CadQuery code in the sandbox subprocess."""

    success: bool
    code: str
    error: str | None = None
    metrics: GeometryMetrics | None = None
    stl_path: Path | None = None
    step_path: Path | None = None
    stdout: str = ""
    duration_s: float


class Critique(BaseModel):
    """Structured visual critique returned by the critic agent."""

    matches_spec: bool
    score: int = Field(ge=0, le=10)
    issues: list[str]
    suggestions: list[str]
    summary: str


class IterationRecord(BaseModel):
    """Everything produced by one outer self-refine iteration."""

    index: int
    execution: ExecutionResult | None = None
    render_path: Path | None = None
    critique: Critique | None = None
    summary: str = ""

    @property
    def effective_score(self) -> int:
        return self.critique.score if self.critique is not None else 0


class RunConfig(BaseModel):
    """Tunable parameters of a generation run."""

    model: str = DEFAULT_MODEL
    critic_model: str | None = None
    max_iterations: int = 5
    score_threshold: int = 8
    exec_timeout_s: float = 60
    max_exec_attempts_per_iteration: int = 4
    out_dir: Path = Path("runs")

    @model_validator(mode="after")
    def _default_critic_model(self) -> "RunConfig":
        if self.critic_model is None:
            self.critic_model = DEFAULT_CRITIC_MODEL
        return self


class RunResult(BaseModel):
    """Final outcome of a self-refine run."""

    accepted: bool
    spec: str
    drawings: list[str] = []  # persisted input-drawing filenames under run_dir/input/
    interpretation: str | None = None  # final (possibly edited) extracted-dimensions digest
    best: IterationRecord
    iterations: list[IterationRecord]
    run_dir: Path
