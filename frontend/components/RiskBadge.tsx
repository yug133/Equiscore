import React from "react";

/**
 * Props for the RiskBadge component.
 */
interface RiskBadgeProps {
  /** Risk level: LOW | MEDIUM | HIGH */
  level: "LOW" | "MEDIUM" | "HIGH";
}

/**
 * RiskBadge Component
 *
 * Displays a color-coded badge indicating the applicant's risk level.
 * - LOW: Green badge
 * - MEDIUM: Yellow/amber badge
 * - HIGH: Red badge
 */
export default function RiskBadge({ level }: RiskBadgeProps) {
  // TODO: Implement color-coded risk badge
  return <span>{level}</span>;
}
