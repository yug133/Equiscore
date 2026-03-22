"use client";

import React from "react";

/**
 * Props for the ApplicantForm component.
 */
interface ApplicantFormProps {
  /** Callback function when form is submitted */
  onSubmit: (data: Record<string, any>) => void;
  /** Whether the form is currently submitting */
  isLoading: boolean;
}

/**
 * ApplicantForm Component
 *
 * 18-field input form for loan applicant data. Collects all features
 * required by the EquiScore model: age, gender, income, employment,
 * occupation, education, family status, housing, region, assets,
 * credit amount, annuity, goods price, and external source scores.
 */
export default function ApplicantForm({
  onSubmit,
  isLoading,
}: ApplicantFormProps) {
  // TODO: Implement 18-field form with proper input types and validation
  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit({});
      }}
    >
      <h3>Applicant Information</h3>
      {/* Placeholder: Add 18 input fields */}
      <button type="submit" disabled={isLoading}>
        {isLoading ? "Scoring..." : "Get Credit Score"}
      </button>
    </form>
  );
}
