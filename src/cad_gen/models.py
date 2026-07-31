"""Pydantic schemas shared across the sandbox, agents, and orchestrator."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# NOTE the `openai-responses:` prefix (the Responses API), not plain `openai:` (which still
# resolves to Chat Completions). gpt-5.6-luna rejects function tools on Chat Completions
# with "Function tools with reasoning_effort are not supported ... use /v1/responses", and
# EVERY agent here needs tools — the generator calls execute_cad_code, and the critic's
# structured Critique output is itself tool-backed. Plain `openai:gpt-5.6-luna` 400s.
DEFAULT_MODEL = "openai-responses:gpt-5.6-luna"  # generator
DEFAULT_CRITIC_MODEL = "openai-responses:gpt-5.6-luna"  # vision critic
# Used by every agent when the global BMW toggle (CAD_GEN_BMW) is set. See _coerce_model.
BMW_DEFAULT_MODEL = "openai/gpt-5-mini"

# Provider-agnostic reasoning/thinking effort levels (pydantic-ai's unified `thinking`
# ModelSettings field). `None` leaves the provider default untouched.
ReasoningEffort = Literal["minimal", "low", "medium", "high", "xhigh"]
REASONING_EFFORTS: tuple[ReasoningEffort, ...] = ("minimal", "low", "medium", "high", "xhigh")

# Python CAD library the generator writes code in. Both are OpenCASCADE-backed and both
# export STEP/STL, so everything downstream of the sandbox (rendering, reprojection,
# metrics, the bench adapter) is unaffected by the choice. Only the prompts and the
# sandbox harness differ — see agents/prompts.py and sandbox/executor.py.
# build123d is an optional dependency: install it with `uv sync --extra build123d`.
CadLibrary = Literal["cadquery", "build123d"]
CAD_LIBRARIES: tuple[CadLibrary, ...] = ("cadquery", "build123d")
DEFAULT_LIBRARY: CadLibrary = "cadquery"


def _bmw_enabled() -> bool:
    """Global toggle: route every agent through the BMW LLM gateway (for BMW PCs)."""
    return os.environ.get("CAD_GEN_BMW", "").strip().lower() in {"1", "true", "yes", "on"}


def _coerce_model(value: object) -> object:
    """Resolve a model setting, honoring the global BMW toggle.

    Toggle ON: any non-`bmw:` string becomes `bmw:<BMW_DEFAULT_MODEL>` (so a locked-down BMW
    PC works with one switch), while an explicit `bmw:<other>` is kept as a per-agent override
    (e.g. a vision-capable critic). Toggle OFF: the value is used as-is, so the configured
    `provider:model` string (OpenAI by default) reaches the agent unchanged.
    """
    if _bmw_enabled() and isinstance(value, str):
        return value if value.strip().startswith("bmw:") else f"bmw:{BMW_DEFAULT_MODEL}"
    return value


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
    """Outcome of running generated CAD code in the sandbox subprocess."""

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


class EditDelta(BaseModel):
    """Deterministic comparison of an edited model against the base it was derived from.

    Editing mode only. CADGenBench renormalizes an editing sample's shape score against
    the unmodified input, so returning the base untouched ("no-op") scores 0 on shape and
    caps the sample at 0.4 — it is never worth submitting. The vision critic cannot
    reliably see a small or internal edit in a shaded render, so this measured signal
    backstops it: see `is_noop`, which hard-fails the iteration in the orchestrator.
    """

    is_noop: bool
    base_volume_mm3: float
    candidate_volume_mm3: float
    volume_change_pct: float
    digest: str = ""  # LLM-facing text block (see step_metrics.describe_edit_delta)


class ValidityReport(BaseModel):
    """Verdict of the benchmark's own validity gate on an iteration's exported STEP.

    CADGenBench zeroes `cad_score` for an invalid solid, so this is the one signal that
    overrides everything else: a candidate failing here is worth exactly as much as no
    candidate. Populated from `step_validity.check_step_validity`; `evaluated=False` means
    the gate could not run (timeout / crash) and NOTHING was proven either way, which is
    why `is_valid` alone is never enough to reject on.
    """

    evaluated: bool = True
    is_valid: bool = False
    is_watertight: bool = False
    mesh_checked: bool = False  # False => a mesh-only defect would have gone unnoticed
    errors: list[str] = []  # verbatim OCCT status names, fed to the generator as-is
    unknown_reason: str | None = None
    digest: str = ""  # LLM-facing text block (see step_validity.describe_validity)

    @property
    def known_invalid(self) -> bool:
        """True only when the gate RAN and rejected the shape.

        The distinction matters: an unevaluated gate must never cost an iteration its score.
        """
        return self.evaluated and not self.is_valid


class EditRegion(BaseModel):
    """One connected lump of material the edit added or removed."""

    volume_mm3: float
    bbox_mm: tuple[float, float, float]
    center_mm: tuple[float, float, float]


class EditDiff(BaseModel):
    """Boolean before/after comparison of an edited candidate against its base model.

    Where `EditDelta` asks "did anything change at all?" (bulk volume and bbox), this asks
    "what changed, how much of it, and where?" by cutting the two solids against each other.
    That is the difference between catching a no-op and catching an over-cut: v3 sample 224
    was asked to remove one internal groove and removed 33 058 mm3 spanning the part's whole
    length, which the bulk delta happily reported as "the geometry did change".
    """

    evaluated: bool = True
    removed: list[EditRegion] = []
    added: list[EditRegion] = []
    removed_volume_mm3: float = 0.0
    added_volume_mm3: float = 0.0
    base_volume_mm3: float = 0.0
    changed_fraction: float = 0.0  # (removed + added) / base volume
    locality: float = 0.0  # diagonal(changed bbox) / diagonal(base bbox), 0..~1
    # Added material reaching beyond the base model's bounding box, and how far past it the
    # worst lump reaches (mm, and as a fraction of that axis's extent). Growing the part is
    # not automatically wrong — v3 sample 245 was asked to raise a wall by 5 mm — so the
    # magnitude is what distinguishes it from sample 205's 116 mm spike.
    outside_base_bbox_mm3: float = 0.0
    overshoot_mm: float = 0.0
    overshoot_fraction: float = 0.0
    plausible: bool = True  # verdict of edit_diff.is_plausible_local_edit
    verdict_reason: str = ""
    digest: str = ""
    skipped_reason: str | None = None


class IterationRecord(BaseModel):
    """Everything produced by one outer self-refine iteration."""

    index: int
    execution: ExecutionResult | None = None
    render_path: Path | None = None
    critique: Critique | None = None
    reprojection: ReprojectionReport | None = None
    constraint_validation: ConstraintValidation | None = None
    edit_delta: EditDelta | None = None
    validity: ValidityReport | None = None
    summary: str = ""

    @property
    def effective_score(self) -> int:
        return self.critique.score if self.critique is not None else 0

    @property
    def gate_ok(self) -> bool:
        """False only when the validity gate ran and rejected this iteration.

        Used as the FIRST key when picking a champion, so a valid-but-mediocre candidate
        always beats a great-looking invalid one — the latter scores 0 on the leaderboard.
        An unchecked or unevaluated iteration counts as ok: absence of a verdict is not a
        rejection.
        """
        return self.validity is None or not self.validity.known_invalid


class RunConfig(BaseModel):
    """Tunable parameters of a generation run."""

    # Defaults resolve from the environment at instantiation time (after
    # load_dotenv runs) so CAD_GEN_MODEL / CAD_GEN_CRITIC_MODEL drive both the
    # CLI and the web backend. Precedence: explicit value -> env var -> constant.
    model: str = Field(
        default_factory=lambda: _coerce_model(os.environ.get("CAD_GEN_MODEL") or DEFAULT_MODEL),
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
            os.environ.get("CAD_GEN_VIEW_MODEL") or DEFAULT_CRITIC_MODEL
        ),
        validate_default=True,
    )
    # Python CAD library the generator writes code in, and therefore which sandbox harness
    # executes it. Unlike the model fields this does NOT resolve from the environment: it is
    # a per-run choice made explicitly via `cad-gen --library` or the web form.
    library: CadLibrary = DEFAULT_LIBRARY
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
    # The benchmark's validity gate, run on each iteration's exported STEP. NOT advisory,
    # unlike reprojection: a failing candidate scores 0 on the leaderboard, so the gate
    # zeroes the critique and pushes the loop to retry. Off only for speed on runs that
    # will never be submitted.
    validity_gate: bool = True
    validity_timeout_s: float = 180
    # Boolean before/after diff of an editing candidate against its base model. Run once,
    # on the champion, after the loop — the booleans cost 4-18s per direction on real parts,
    # which is too much per iteration for a check whose feedback arrives too late to use.
    edit_diff: bool = True
    edit_diff_timeout_s: float = 300
    edit_diff_max_candidates: int = 2  # how far down the ranking to look for a plausible edit

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
    def _normalize_model_provider(cls, value: object) -> object:
        return _coerce_model(value)

    @model_validator(mode="after")
    def _default_critic_model(self) -> "RunConfig":
        if self.critic_model is None:
            self.critic_model = _coerce_model(
                os.environ.get("CAD_GEN_CRITIC_MODEL") or DEFAULT_CRITIC_MODEL
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
    # Editing mode only: the boolean before/after measurement of `best` against the base
    # model, run once after the loop. `evaluated=False` means it could not run.
    edit_diff: EditDiff | None = None
