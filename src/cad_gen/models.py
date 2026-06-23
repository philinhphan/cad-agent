"""Pydantic schemas shared across the sandbox, agents, and orchestrator."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

DEFAULT_MODEL = "google:gemini-3.5-flash"  # generator
DEFAULT_CRITIC_MODEL = "google:gemini-3.5-flash"  # vision critic
# Used by every agent when the global BMW toggle (CAD_GEN_BMW) is set. See _coerce_model.
BMW_DEFAULT_MODEL = "openai/gpt-5-mini"

# Provider-agnostic reasoning/thinking effort levels (pydantic-ai's unified `thinking`
# ModelSettings field). `None` leaves the provider default untouched.
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = ("minimal", "low", "medium", "high", "xhigh")


_LEGACY_PROVIDER = "op" + "enai"
_LEGACY_MODEL_PREFIX = "g" + "pt-"
_LEGACY_PROVIDER_PREFIXES = {
    _LEGACY_PROVIDER,
    f"{_LEGACY_PROVIDER}-responses",
    f"{_LEGACY_PROVIDER}-chat",
}


def _replace_legacy_model(value: object, fallback: str) -> object:
    """Force stale provider/model config onto the Gemini stack."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped.startswith("bmw:"):
        return value  # BMW gateway models are an explicit opt-in — never rewrite them
    provider, _, model_name = stripped.partition(":")
    if provider in _LEGACY_PROVIDER_PREFIXES:
        return fallback
    if provider == stripped and stripped.lower().startswith(_LEGACY_MODEL_PREFIX):
        return fallback
    if model_name.lower().startswith(_LEGACY_MODEL_PREFIX):
        return fallback
    return value


def _bmw_enabled() -> bool:
    """Global toggle: route every agent through the BMW LLM gateway (for BMW PCs)."""
    return os.environ.get("CAD_GEN_BMW", "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_model(value: object, fallback: str) -> object:
    """Resolve a model setting, honoring the global BMW toggle.

    Toggle ON: any non-`bmw:` string becomes `bmw:<BMW_DEFAULT_MODEL>` (so a locked-down BMW
    PC works with one switch), while an explicit `bmw:<other>` is kept as a per-agent override
    (e.g. a vision-capable critic). Toggle OFF: the unchanged legacy-rewrite behavior — and an
    explicit `bmw:` prefix still routes to the gateway via _replace_legacy_model's guard.
    """
    if _bmw_enabled() and isinstance(value, str):
        return value if value.strip().startswith("bmw:") else f"bmw:{BMW_DEFAULT_MODEL}"
    return _replace_legacy_model(value, fallback)


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


class IntrospectionResult(BaseModel):
    """Outcome of a read-only geometry probe (see sandbox/introspect.py).

    ADVISORY tool output for the generator only — never exported, measured, or scored.
    `ok=False` means the *code* failed to build (`error` holds the traceback). A bad
    *selector* still comes back `ok=True`, with the diagnostic (count 0 + selector_error)
    inside `data`.
    """

    ok: bool
    error: str | None = None
    data: dict | None = None


class Critique(BaseModel):
    """Structured visual critique returned by the critic agent."""

    matches_spec: bool
    score: int = Field(ge=0, le=10)
    issues: list[str]
    suggestions: list[str]
    summary: str


class DrawingHoleConstraint(BaseModel):
    """Hole callout parsed from a drawing interpretation."""

    diameter_mm: float
    count: int = 1
    through: bool | None = None
    source: str = ""


class DrawingScalarConstraint(BaseModel):
    """Single scalar drawing constraint such as a radius or angle."""

    value_mm: float
    count: int = 1
    source: str = ""


class DrawingAngleConstraint(BaseModel):
    degrees: float
    source: str = ""


class DrawingCounterboreConstraint(BaseModel):
    diameter_mm: float
    depth_mm: float | None = None
    source: str = ""


class DrawingConstraints(BaseModel):
    """Structured, machine-readable constraints extracted from a drawing digest."""

    envelope_mm: tuple[float, float, float] | None = None
    holes: list[DrawingHoleConstraint] = []
    counterbores: list[DrawingCounterboreConstraint] = []
    radii: list[DrawingScalarConstraint] = []
    angles: list[DrawingAngleConstraint] = []
    source_text: str = ""


class ConstraintCheck(BaseModel):
    """One deterministic or advisory check against generated geometry."""

    name: str
    status: Literal["pass", "fail", "unverified"]
    message: str
    expected: str | None = None
    actual: str | None = None


class ConstraintValidation(BaseModel):
    """Post-generation validation of structured drawing constraints."""

    passed: bool
    checks: list[ConstraintCheck] = []
    digest: str = ""


class ReprojectionMismatch(BaseModel):
    """Connected-component mismatch extracted from a colour reprojection overlay."""

    kind: Literal["missing_drawing_line", "extra_model_line"]
    view: str
    bbox_norm: tuple[float, float, float, float]
    centroid_norm: tuple[float, float]
    area_px: int
    area_frac: float
    location: str


class ReprojectionView(BaseModel):
    """Per-view geometric overlap of the reprojected STEP against one drawing view."""

    coverage: float  # recall: fraction of this view's drawing lines reproduced
    chamfer_pct: float
    aspect_ok: bool
    aspect_rel_err: float
    aspect_signed: float  # + => part too wide for its height in this view, - => too tall
    overlay_path: Path | None = None  # per-view colour overlay PNG (absolute fs path)
    mismatches: list[ReprojectionMismatch] = []


class ReprojectionReport(BaseModel):
    """Adapter-built, LLM-facing summary of a deterministic reproject_check run.

    ADVISORY ONLY: this never affects `effective_score`, accept logic, or champion
    choice. `evaluated=False` means the drawing could not be parsed (no views) or all
    views looked orientation/scale-mismatched — the signal is withheld rather than used
    to penalise the model. The colour overlay is a *locator* (where lines are missing or
    extra), not a ruler: it is dimensionless, so dimensions must be read off the original
    drawing, never estimated from the overlay.
    """

    evaluated: bool
    passed: bool = False  # mirrors report.overall.pass; informational only
    views_found: int = 0
    mean_chamfer_pct: float | None = None
    views: dict[str, ReprojectionView] = {}  # only located views (front/top/side)
    source_drawing: str | None = None
    children: list["ReprojectionReport"] = []  # one child per input drawing for aggregate reports
    digest: str = ""  # compact text block fed to the critic (no images)
    interpretation: str = ""  # one global-vs-local interpretation line
    composite_path: Path | None = None  # labelled multi-view overlay PNG for the generator
    skipped_reason: str | None = None  # why evaluated=False (logging/debug only)


class IterationRecord(BaseModel):
    """Everything produced by one outer self-refine iteration."""

    index: int
    execution: ExecutionResult | None = None
    render_path: Path | None = None
    critique: Critique | None = None
    reprojection: ReprojectionReport | None = None
    constraint_validation: ConstraintValidation | None = None
    summary: str = ""

    @property
    def effective_score(self) -> int:
        return self.critique.score if self.critique is not None else 0


class RunConfig(BaseModel):
    """Tunable parameters of a generation run."""

    # Defaults resolve from the environment at instantiation time (after
    # load_dotenv runs) so CAD_GEN_MODEL / CAD_GEN_CRITIC_MODEL drive both the
    # CLI and the web backend. Precedence: explicit value -> env var -> constant.
    model: str = Field(
        default_factory=lambda: _coerce_model(
            os.environ.get("CAD_GEN_MODEL") or DEFAULT_MODEL, DEFAULT_MODEL
        ),
        validate_default=True,
    )
    # Reasoning/thinking effort for the generator model. `None` (default) leaves the
    # provider's own default untouched; CAD_GEN_REASONING_EFFORT overrides. Applies via
    # pydantic-ai's provider-agnostic `thinking` ModelSettings field, so it works for
    # Gemini and Anthropic generator models alike.
    reasoning_effort: ReasoningEffort | None = Field(
        default_factory=lambda: os.environ.get("CAD_GEN_REASONING_EFFORT") or None,
        validate_default=True,  # run normalization + Literal check on the env-sourced default
    )
    # Same, for the (Gemini) vision critic — lets it think before judging, which closes the
    # false-accepts where a no-thinking flash rubber-stamped a flawed part. Opt-in via
    # CAD_GEN_CRITIC_REASONING_EFFORT; `None` leaves the provider default untouched.
    critic_reasoning_effort: ReasoningEffort | None = Field(
        default_factory=lambda: os.environ.get("CAD_GEN_CRITIC_REASONING_EFFORT") or None,
        validate_default=True,
    )
    critic_model: str | None = None
    # Vision model that locates the drawing's orthographic views for the reprojection
    # check (provider:model). Defaults to the vision-critic default; CAD_GEN_VIEW_MODEL
    # overrides. Only used in drawing mode.
    view_model: str = Field(
        default_factory=lambda: _coerce_model(
            os.environ.get("CAD_GEN_VIEW_MODEL") or DEFAULT_CRITIC_MODEL,
            DEFAULT_CRITIC_MODEL,
        ),
        validate_default=True,
    )
    max_iterations: int = 5
    score_threshold: int = 8
    exec_timeout_s: float = 60
    max_exec_attempts_per_iteration: int = 4
    out_dir: Path = Path("runs")
    # Deterministic reprojection check (drawing mode only). Advisory: it never gates
    # the score. The two coverage knobs only shape the wording / withholding heuristic,
    # they are NOT pass/fail thresholds.
    reproject: bool = True
    reproject_timeout_s: float = 120
    reproject_low_coverage: float = 0.80  # below this a view reads as "geometry missing"
    reproject_orientation_coverage: float = 0.55  # all views below => likely orientation, withhold

    @field_validator("reasoning_effort", "critic_reasoning_effort", mode="before")
    @classmethod
    def _normalize_reasoning_effort(cls, value: object) -> object:
        """Accept env strings case-insensitively; treat empty/blank as unset."""
        if isinstance(value, str):
            stripped = value.strip().lower()
            return stripped or None
        return value

    @field_validator("model", "critic_model", "view_model", mode="before")
    @classmethod
    def _normalize_model_provider(cls, value: object, info) -> object:
        fallback = (
            DEFAULT_CRITIC_MODEL
            if info.field_name in {"critic_model", "view_model"}
            else DEFAULT_MODEL
        )
        return _coerce_model(value, fallback)

    @model_validator(mode="after")
    def _default_critic_model(self) -> "RunConfig":
        if self.critic_model is None:
            self.critic_model = _coerce_model(
                os.environ.get("CAD_GEN_CRITIC_MODEL") or DEFAULT_CRITIC_MODEL,
                DEFAULT_CRITIC_MODEL,
            )
        return self


class RunResult(BaseModel):
    """Final outcome of a self-refine run."""

    accepted: bool
    spec: str
    drawings: list[str] = []  # persisted input-drawing filenames under run_dir/input/
    interpretation: str | None = None  # final (possibly edited) extracted-dimensions digest
    constraints: DrawingConstraints | None = None
    best: IterationRecord
    iterations: list[IterationRecord]
    run_dir: Path
