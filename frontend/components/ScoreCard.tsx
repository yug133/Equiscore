import React from "react";

/**
 * Props for the ScoreCard component.
 */
interface ScoreCardProps {
  /** Credit score on 300-900 scale */
  score: number;
  /** Risk level: LOW | MEDIUM | HIGH */
  riskLevel: string;
  /** Probability of default (0-1) */
  defaultProbability: number;
}

/**
 * ScoreCard Component
 *
 * Displays the applicant's credit score (300-900 scale) with a
 * visual gauge, risk level indicator, and default probability.
 * Color-coded based on risk level.
 */
export default function ScoreCard({
  score,
  riskLevel,
  defaultProbability,
}: ScoreCardProps) {
  // TODO: Implement credit score display with visual gauge
  return (
    <div>
      <h2>Credit Score</h2>
      <p>{score}</p>
      <p>Risk: {riskLevel}</p>
      <p>Default Probability: {(defaultProbability * 100).toFixed(1)}%</p>
    </div>
  );
}
