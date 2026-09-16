"""Portfolio concentration analytics derived from recorded target weights."""

import numpy as np
import pandas as pd


def concentration_timeseries(backtest: pd.DataFrame) -> pd.DataFrame:
    """Calculate maximum weight, HHI, and effective holdings by period."""
    if "weights" not in backtest or "date" not in backtest:
        return pd.DataFrame(
            columns=["date", "maximum_weight", "hhi", "effective_holdings"]
        )
    rows = []
    for date, raw_weights in zip(backtest["date"], backtest["weights"]):
        if not isinstance(raw_weights, dict) or not raw_weights:
            continue
        weights = np.asarray(list(raw_weights.values()), dtype=float)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            continue
        weights = weights / weights.sum()
        hhi = float(np.square(weights).sum())
        rows.append(
            {
                "date": pd.Timestamp(date),
                "maximum_weight": float(weights.max()),
                "hhi": hhi,
                "effective_holdings": 1.0 / hhi,
            }
        )
    return pd.DataFrame(rows)


def concentration_summary(backtest: pd.DataFrame) -> pd.Series:
    """Summarize concentration through time, returning N/A if unavailable."""
    series = concentration_timeseries(backtest)
    if series.empty:
        return pd.Series(
            {
                "Average Maximum Weight": "N/A",
                "Maximum Single-Name Weight": "N/A",
                "Average HHI": "N/A",
                "Maximum HHI": "N/A",
                "Average Effective Holdings": "N/A",
                "Minimum Effective Holdings": "N/A",
            },
            dtype=object,
        )
    return pd.Series(
        {
            "Average Maximum Weight": series["maximum_weight"].mean(),
            "Maximum Single-Name Weight": series["maximum_weight"].max(),
            "Average HHI": series["hhi"].mean(),
            "Maximum HHI": series["hhi"].max(),
            "Average Effective Holdings": series["effective_holdings"].mean(),
            "Minimum Effective Holdings": series["effective_holdings"].min(),
        }
    )
