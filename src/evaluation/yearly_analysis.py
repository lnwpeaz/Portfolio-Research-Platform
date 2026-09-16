"""Calendar-year portfolio analytics."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import max_drawdown


def yearly_performance(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Compute non-annualized partial-year aware calendar-year statistics."""
    if not isinstance(returns.index, pd.DatetimeIndex):
        raise TypeError("Returns must have a DatetimeIndex.")

    rows = []
    clean_returns = returns.dropna().sort_index()
    for year, values in clean_returns.groupby(clean_returns.index.year):
        volatility = values.std(ddof=1) * np.sqrt(periods_per_year)
        rows.append(
            {
                "year": int(year),
                "return": (1 + values).prod() - 1,
                "volatility": volatility,
                "sharpe": (
                    values.mean() / values.std(ddof=1) * np.sqrt(periods_per_year)
                    if len(values) > 1 and not np.isclose(values.std(ddof=1), 0)
                    else np.nan
                ),
                "max_drawdown": max_drawdown(values),
                "observations": len(values),
            }
        )
    return pd.DataFrame(rows).set_index("year") if rows else pd.DataFrame()


def common_period_yearly_analysis(
    common_returns: pd.DataFrame,
    benchmark_name: str = "benchmark",
    expected_periods_per_year: int = 12,
) -> pd.DataFrame:
    """Calendar-year returns, leaders, laggards, and benchmark excess returns."""
    if benchmark_name not in common_returns:
        raise ValueError(f"Missing benchmark column: {benchmark_name!r}.")
    if common_returns.empty or common_returns.isna().any().any():
        raise ValueError("common_returns must be non-empty and complete.")
    annual = (
        (1 + common_returns)
        .groupby(common_returns.index.year)
        .prod()
        .sub(1)
        .rename_axis(index="year")
    )
    strategy_columns = [
        column for column in annual.columns if column != benchmark_name
    ]
    observations = common_returns.groupby(common_returns.index.year).size()
    analysis = annual.copy()
    analysis["observations"] = observations
    analysis["partial_year"] = observations < expected_periods_per_year
    analysis["best_strategy"] = annual[strategy_columns].idxmax(axis=1)
    analysis["worst_strategy"] = annual[strategy_columns].idxmin(axis=1)
    for strategy in strategy_columns:
        analysis[f"{strategy}_excess_vs_{benchmark_name}"] = (
            annual[strategy] - annual[benchmark_name]
        )
    return analysis
