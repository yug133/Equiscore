import type { Metadata } from "next";
import React from "react";

export const metadata: Metadata = {
  title: "EquiScore — Fair Credit Scoring",
  description:
    "Fair and Explainable Credit Scoring for Thin-File Applicants in India",
};

/**
 * Root layout component for the EquiScore application.
 *
 * Wraps all pages with shared HTML structure, metadata, and global styles.
 */
export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
