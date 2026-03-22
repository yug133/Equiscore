/**
 * EquiScore TypeScript Type Definitions
 *
 * Interfaces matching the FastAPI Pydantic schemas for type-safe
 * frontend-backend communication.
 */

// ─── Request Types ──────────────────────────────────────────────────

/** Request body for POST /predict */
export interface PredictRequest {
  age: number;
  gender: string;
  income: number;
  employment_type: string;
  occupation_type: string;
  education_type: string;
  family_status: string;
  housing_type: string;
  region_rating: number;
  own_car: boolean;
  own_realty: boolean;
  children_count: number;
  family_members: number;
  credit_amount: number;
  annuity_amount: number;
  goods_price: number;
  ext_source_1?: number;
  ext_source_2?: number;
}

/** Request body for POST /improve */
export interface ImproveRequest {
  application_id: string;
  num_tips?: number;
}

// ─── Response Types ─────────────────────────────────────────────────

/** Response from POST /predict */
export interface PredictResponse {
  application_id: string;
  credit_score: number;
  default_probability: number;
  risk_level: "LOW" | "MEDIUM" | "HIGH";
  shap_explanation: Record<string, number>;
  top_factors: string[];
}

/** Response from GET /audit */
export interface AuditResponse {
  model_name: string;
  overall_metrics: Record<string, number>;
  fairness_metrics: Record<string, Record<string, number>>;
  fairness_flags: string[];
}

/** Response from POST /improve */
export interface ImproveResponse {
  application_id: string;
  current_score: number;
  tips: CounterfactualTip[];
  potential_score: number;
}

// ─── Component Data Types ───────────────────────────────────────────

/** Individual SHAP feature contribution for waterfall chart */
export interface ShapFeature {
  feature: string;
  value: number;
  contribution: number;
}

/** Fairness metrics structure */
export interface FairnessMetrics {
  dpg: Record<string, number>;
  eod: Record<string, number>;
  dir: Record<string, number>;
}

/** Single counterfactual improvement suggestion */
export interface CounterfactualTip {
  feature: string;
  current_value: number | string;
  suggested_value: number | string;
  impact: string;
}
