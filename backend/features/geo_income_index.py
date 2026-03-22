"""
Geo-Income Index (GII) Feature
Regional income-adjusted risk indicator.
"""

import pandas as pd
import numpy as np


def compute_geo_income_index(
    application_df: pd.DataFrame,
) -> pd.Series:
    """
    Compute Geo-Income Index for each applicant.

    Compares an applicant's income to the median income of their
    geographic region (REGION_POPULATION_RELATIVE, REGION_RATING_CLIENT).
    Accounts for regional economic disparity in creditworthiness assessment.

    Higher score = income above regional median = stronger financial position.

    Args:
        application_df: Main application table with income and region columns.

    Returns:
        Series indexed by SK_ID_CURR with geo-income index 0 to 1.
    """
    raise NotImplementedError("To be implemented")
