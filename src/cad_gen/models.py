"""Pydantic schemas shared across the sandbox, agents, and orchestrator."""

from enum import Enum
from pathlib import Path
from typing import Literal

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


# --------------------------------------------------------------------------- #
# Typed drawing target — a machine-checkable transcription of a drawing.
# Every field is optional: drawings outside the TooTallToby set rarely state a
# target mass/density, and a plain text spec has none of this. Absent fields make
# the corresponding deterministic check SKIP rather than fail.
# --------------------------------------------------------------------------- #
class HoleType(str, Enum):
    THRU = "thru"
    BLIND = "blind"
    COUNTERBORE = "counterbore"
    COUNTERSINK = "countersink"


class HoleTarget(BaseModel):
    diameter_mm: float
    type: HoleType = HoleType.THRU
    count: int = 1
    depth_mm: float | None = None  # blind depth
    cbore_dia_mm: float | None = None
    cbore_depth_mm: float | None = None
    csk_dia_mm: float | None = None
    csk_angle_deg: float | None = None
    note: str | None = None  # verbatim callout, e.g. "2X Ø5 THRU ALL ⌴Ø10↧5"
    uncertain: bool = False


class FilletTarget(BaseModel):
    radius_mm: float
    count: int = 1
    kind: Literal["fillet", "radius", "chamfer"] = "fillet"
    note: str | None = None
    uncertain: bool = False


class AngleTarget(BaseModel):
    angle_deg: float
    reference: str | None = None
    note: str | None = None
    uncertain: bool = False


class DrawingTarget(BaseModel):
    """Structured transcription of an engineering drawing (all fields optional)."""

    envelope_mm: tuple[float, float, float] | None = None  # overall L × W × H
    material: str | None = None
    density_kg_m3: float | None = None
    target_mass_g: float | None = None  # None when redacted ('XXX g') — never invent
    mass_tol_g: float | None = None
    holes: list[HoleTarget] = []
    fillets: list[FilletTarget] = []
    angles: list[AngleTarget] = []
    symmetry: list[str] = []  # e.g. ["mirror about YZ midplane (CL SYM)"]
    # Derived/positional dimensions worked out ONCE so every iteration shares the same
    # interpretation instead of re-deriving (and disagreeing) — e.g. an upright back-face
    # position implied by a slope + a top-flat width. Free-form, each ideally with its formula.
    key_positions: list[str] = []
    unit_system: str = "MMGS"
    notes: list[str] = []
    raw_digest: str = ""  # the free-form Markdown digest (human-editable surface)


class CylinderFace(BaseModel):
    """One cylindrical B-rep face measured off the OCCT solid (before tessellation).

    Holes, counterbores and round slots all present as cylindrical faces; so do fillets
    and outer round-overs. We record the exact radius + axis so a deterministic check can
    confirm a required hole *diameter* exists, without trusting the vision render.
    """

    radius_mm: float
    axis: tuple[float, float, float] | None = None  # unit axis direction


class GeometryMetrics(BaseModel):
    """Measured properties of an executed CAD model."""

    volume_mm3: float
    bbox_mm: tuple[float, float, float]
    center_of_mass: tuple[float, float, float]
    n_solids: int
    n_faces: int
    is_watertight: bool | None = None
    mass_g: float | None = None  # volume × density, filled when a density is known
    # Cylindrical B-rep faces (radius + axis), measured by the harness. Optional/back-compat:
    # absent in older metrics.json and in scripted test metrics → empty.
    cylinders: list[CylinderFace] = []


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


# --------------------------------------------------------------------------- #
# Deterministic checks — LLM-independent, computed by the CAD kernel.
# ADVISORY: they never override the critic's score for acceptance; they are shown
# to the user, fed to the critic as evidence, and drive the generator's gradient.
# --------------------------------------------------------------------------- #
class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"  # no target data to compare against (general-drawing path)


class Check(BaseModel):
    name: str
    status: CheckStatus
    critical: bool = False
    target: float | str | None = None
    observed: float | str | None = None
    delta: float | None = None
    tolerance: float | None = None
    message: str = ""


class CheckReport(BaseModel):
    checks: list[Check] = []

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if c.status == CheckStatus.FAIL]

    @property
    def critical_failures(self) -> list[Check]:
        return [c for c in self.checks if c.critical and c.status == CheckStatus.FAIL]

    @property
    def all_critical_pass(self) -> bool:
        return not self.critical_failures


class ChecklistItem(BaseModel):
    """One enumerated requirement the critic checked against the drawing/target."""

    requirement: str
    target: str | None = None
    observed: str | None = None
    status: Literal["pass", "fail", "uncertain"] = "uncertain"
    severity: Literal["critical", "major", "minor"] = "major"


class Critique(BaseModel):
    """Structured visual critique returned by the critic agent."""

    matches_spec: bool
    score: int = Field(ge=0, le=10)
    issues: list[str]
    suggestions: list[str]
    summary: str
    # Free-form, callout-by-callout reasoning the critic writes BEFORE scoring (a cheap
    # single-call chain-of-thought; optional → backward compatible).
    analysis: str = ""
    # Richer, enumerated evaluation (optional → backward compatible).
    checklist: list[ChecklistItem] = []
    dimensional_score: int | None = Field(default=None, ge=0, le=10)
    feature_completeness_score: int | None = Field(default=None, ge=0, le=10)
    proportion_score: int | None = Field(default=None, ge=0, le=10)

    @model_validator(mode="after")
    def _clamp_inconsistent_perfect(self) -> "Critique":
        # Internal-consistency guard (NOT a deterministic override): a self-reported
        # perfect 10 cannot coexist with the critic's OWN failing/uncertain checklist.
        if self.score == 10 and any(
            i.status in ("fail", "uncertain") for i in self.checklist
        ):
            self.score = 9
        return self


class Refutation(BaseModel):
    """Verdict from the adversarial refuter: skeptic looking for any discrepancy."""

    found_discrepancy: bool
    discrepancies: list[str] = []
    most_severe: str | None = None
    # Severity of the most-severe discrepancy — graded so a sub-mm cosmetic nit no longer
    # blocks acceptance the way a wrong/missing feature does. Default "none" keeps a
    # refutation that omits it (older scripts) from capping the score.
    severity: Literal["critical", "major", "minor", "none"] = "none"


class IterationRecord(BaseModel):
    """Everything produced by one outer self-refine iteration."""

    index: int
    execution: ExecutionResult | None = None
    render_path: Path | None = None
    section_path: Path | None = None
    critique: Critique | None = None
    check_report: CheckReport | None = None
    panel_critiques: list[Critique] = []
    refutation: Refutation | None = None
    summary: str = ""

    @property
    def effective_score(self) -> int:
        # Advisory gating: the score is the critic's verdict; deterministic checks
        # never cap it. They surface as a UI warning + critic evidence + feedback.
        return self.critique.score if self.critique is not None else 0

    @property
    def passes_checks(self) -> bool:
        return self.check_report is None or self.check_report.all_critical_pass


class RunConfig(BaseModel):
    """Tunable parameters of a generation run."""

    model: str = DEFAULT_MODEL
    critic_model: str | None = None
    max_iterations: int = 5
    score_threshold: int = 8
    exec_timeout_s: float = 60
    max_exec_attempts_per_iteration: int = 4
    out_dir: Path = Path("runs")

    # Critic robustness (defaults are the cheap single-critic path; the CLI/web
    # surfaces turn the refuter on by default for users).
    critic_samples: int = 1
    critic_models: list[str] | None = None
    enable_adversarial: bool = False
    critic_aggregation: Literal["min", "median"] = "min"

    # Rendering.
    enable_sections: bool = True

    # Optional known-target overrides (used when the answer is known or the drawing
    # redacts it). All optional → general drawings with no known mass still work.
    target_mass_g: float | None = None
    mass_tol_g: float | None = None
    envelope_mm: tuple[float, float, float] | None = None
    density_kg_m3: float | None = None

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
    target: DrawingTarget | None = None  # typed transcription used for the checks
    best: IterationRecord
    iterations: list[IterationRecord]
    run_dir: Path
