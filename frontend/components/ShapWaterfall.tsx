"use client";

import React from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from "recharts";
import { ShapFeature } from "@/lib/types";

/**
 * Props for the ShapWaterfall component.
 */
interface ShapWaterfallProps {
  /** Array of SHAP feature contributions */
  features: ShapFeature[];
  /** Base value (expected model output) */
  baseValue: number;
  /** Final prediction value */
  outputValue: number;
}

/**
 * ShapWaterfall Component
 *
 * Renders a SHAP waterfall bar chart using Recharts, showing how each
 * feature contributes positively or negatively to the final prediction.
 * Features pushing toward default are red; those reducing risk are green.
 */
export default function ShapWaterfall({
  features,
  baseValue,
  outputValue,
}: ShapWaterfallProps) {
  // TODO: Implement SHAP waterfall visualization using Recharts BarChart
  return (
    <div>
      <h3>Feature Contributions (SHAP)</h3>
      <p>Base Value: {baseValue.toFixed(4)}</p>
      <p>Output Value: {outputValue.toFixed(4)}</p>
      {/* Placeholder: Add Recharts BarChart waterfall */}
    </div>
  );
}
