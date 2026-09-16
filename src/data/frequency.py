"""Frequency conversion helpers that preserve point-in-time missingness."""

import pandas as pd


def last_observation_by_month(
    data: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Return each month's final observed row without skipping cell-level NaNs.

    ``resample(...).last()`` takes the last *non-null value per column*, which
    can silently substitute a stale price when an asset is missing on the last
    session. Selecting the final row preserves that missingness for validation.
    """
    if data.empty:
        return data.copy()
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Data must have a DatetimeIndex.")
    if data.index.has_duplicates:
        raise ValueError("Dates must be unique.")

    sorted_data = data.sort_index()
    return sorted_data.groupby(sorted_data.index.to_period("M"), sort=True).tail(1)
