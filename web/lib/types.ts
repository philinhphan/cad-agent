// TypeScript mirror of the cad_gen Pydantic models + web API event shapes.
// Keep in sync with src/cad_gen/models.py and src/cad_gen/web/runs.py.

export interface GeometryMetrics {
  volume_mm3: number;
  bbox_mm: [number, number, number];
  center_of_mass: [number, number, number];
  n_solids: number;
  n_faces: number;
  is_watertight: boolean | null;
}

// Absolute *_path fields are stripped server-side; the browser uses ArtifactUrls.
export interface ExecutionResult {
  success: boolean;
  code: string;
  error: string | null;
  metrics: GeometryMetrics | null;
  stdout: string;
  duration_s: number;
}

export interface Critique {
  matches_spec: boolean;
  score: number; // 0-10
  issues: string[];
  suggestions: string[];
  summary: string;
}

export interface IterationRecord {
  index: number;
  execution: ExecutionResult | null;
  critique: Critique | null;
  summary: string;
}

export interface ArtifactUrls {
  stl?: string;
  step?: string;
  views?: string;
}

export interface IterationPayload {
  record: IterationRecord;
  urls: ArtifactUrls;
}

export interface RunConfig {
  model: string;
  critic_model: string | null;
  max_iterations: number;
  score_threshold: number;
  exec_timeout_s: number;
  max_exec_attempts_per_iteration: number;
}

// SSE events ---------------------------------------------------------------
export interface StartedEvent {
  type: "started";
  run_id: string;
  spec: string;
  config: RunConfig;
}
export interface IterationEvent extends IterationPayload {
  type: "iteration";
}
export interface ResultEvent {
  type: "result";
  run_dir: string;
  result: {
    accepted: boolean;
    spec: string;
    best: IterationRecord;
    iterations: IterationRecord[];
  };
}
export interface ErrorEvent {
  type: "error";
  message: string;
}
export type RunEvent = StartedEvent | IterationEvent | ResultEvent | ErrorEvent;

// REST shapes --------------------------------------------------------------
export interface RunSummary {
  id: string;
  spec: string;
  accepted: boolean;
  score: number;
  n_iterations: number;
  created_at: number; // epoch seconds
}

export interface RunDetail {
  id: string;
  spec: string;
  accepted: boolean;
  best_index: number;
  iterations: IterationPayload[];
}

// Request config knobs the UI exposes (partial RunConfig).
export interface RunConfigInput {
  model?: string;
  critic_model?: string | null;
  max_iterations?: number;
  score_threshold?: number;
  exec_timeout_s?: number;
}
