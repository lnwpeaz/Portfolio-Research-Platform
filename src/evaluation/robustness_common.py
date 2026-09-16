"""Shared helpers for robustness experiments, separate from official reports."""

import pandas as pd

from src.evaluation.metrics import active_performance_summary, performance_summary


def evaluate_return_series(
    returns: pd.Series,
    benchmark_returns: pd.Series,
) -> dict[str, float | int | pd.Timestamp]:
    """Evaluate one strategy and benchmark on their identical available dates."""
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna().sort_index()
    if aligned.empty:
        raise ValueError("Strategy and benchmark have no common observations.")
    absolute = performance_summary(aligned["strategy"])
    active = active_performance_summary(
        aligned["strategy"], aligned["benchmark"]
    )
    return {
        "start_date": aligned.index.min(),
        "end_date": aligned.index.max(),
        "observations": len(aligned),
        "annual_return": absolute["Annual Return"],
        "annual_volatility": absolute["Annual Volatility"],
        "sharpe": absolute["Sharpe Ratio"],
        "sortino": absolute["Sortino Ratio"],
        "max_drawdown": absolute["Max Drawdown"],
        "calmar": absolute["Calmar Ratio"],
        "annual_active_return": active["Annual Active Return"],
        "tracking_error": active["Tracking Error"],
        "information_ratio": active["Information Ratio"],
        "beta": active["Beta"],
        "annual_alpha": active["Annual Alpha"],
    }


def aligned_pair(
    first: pd.Series,
    second: pd.Series,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Align two experimental return paths and a benchmark without filling."""
    result = pd.concat(
        [
            first.rename("first"),
            second.rename("second"),
            benchmark.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna().sort_index()
    if result.empty:
        raise ValueError("Experimental paths have no complete common period.")
    return result
