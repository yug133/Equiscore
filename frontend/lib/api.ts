/**
 * EquiScore API Client
 *
 * Typed API calls to the FastAPI backend.
 * All functions return typed responses matching the Pydantic schemas.
 */

import axios from "axios";
import {
  PredictRequest,
  PredictResponse,
  AuditResponse,
  ImproveRequest,
  ImproveResponse,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * Submit applicant data for credit scoring.
 *
 * @param data - PredictRequest with 18 applicant features
 * @returns PredictResponse with score, risk level, and SHAP explanation
 */
export async function predictCreditScore(
  data: PredictRequest
): Promise<PredictResponse> {
  throw new Error("To be implemented");
}

/**
 * Fetch the latest fairness audit report.
 *
 * @returns AuditResponse with DPG, EOD, DIR metrics and flags
 */
export async function getAuditReport(): Promise<AuditResponse> {
  throw new Error("To be implemented");
}

/**
 * Request improvement tips for a specific application.
 *
 * @param data - ImproveRequest with application_id and num_tips
 * @returns ImproveResponse with DiCE counterfactual suggestions
 */
export async function getImprovementTips(
  data: ImproveRequest
): Promise<ImproveResponse> {
  throw new Error("To be implemented");
}
