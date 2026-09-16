# src/evaluation/report.py

from pathlib import Path
import pandas as pd

from src.evaluation.metrics import (
    active_performance_summary,
    performance_summary,
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

REPORT_DIR = (
    BASE_DIR
    / "reports"
)


def save_performance_report(
    comparison: pd.DataFrame,
    filename: str = "performance.csv",
    index: bool = True,
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        REPORT_DIR / filename,
        index=index,
    )


def save_backtest_results(
    backtest: pd.DataFrame,
    filename: str,
) -> None:

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    backtest.to_parquet(
        REPORT_DIR / filename
    )


def common_period_returns(
    strategies: dict[str, pd.Series],
    benchmark_returns: pd.Series,
    benchmark_name: str = "benchmark",
) -> pd.DataFrame:
    """Inner-join every strategy and benchmark on identical realization dates."""
    if not strategies:
        raise ValueError("At least one strategy return series is required.")
    if benchmark_name in strategies:
        raise ValueError("benchmark_name must not duplicate a strategy name.")

    series = {}
    for name, returns in strategies.items():
        if returns.index.has_duplicates:
            raise ValueError(f"Duplicate return dates for strategy {name!r}.")
        series[name] = returns.rename(name)
    if benchmark_returns.index.has_duplicates:
        raise ValueError("Duplicate benchmark return dates.")
    series[benchmark_name] = benchmark_returns.rename(benchmark_name)

    common = pd.concat(series.values(), axis=1, join="inner").dropna().sort_index()
    if common.empty:
        raise ValueError("Strategies and benchmark have no complete common period.")
    return common


def common_period_performance_report(
    common_returns: pd.DataFrame,
    benchmark_name: str = "benchmark",
) -> pd.DataFrame:
    """Calculate absolute and benchmark-relative metrics on one common sample."""
    if benchmark_name not in common_returns.columns:
        raise ValueError(f"Missing benchmark column: {benchmark_name!r}.")
    if common_returns.isna().any().any():
        raise ValueError("common_returns must be a complete aligned return panel.")

    report = {}
    benchmark = common_returns[benchmark_name]
    for name in common_returns.columns:
        metrics = performance_summary(common_returns[name])
        if name != benchmark_name:
            metrics = pd.concat(
                [metrics, active_performance_summary(common_returns[name], benchmark)]
            )
        report[name] = metrics
    metric_order = [
        "Annual Return",
        "Annual Volatility",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Max Drawdown",
        "Calmar Ratio",
        "Annual Active Return",
        "Tracking Error",
        "Information Ratio",
        "Beta",
        "Annual Alpha",
    ]
    return pd.DataFrame(report).reindex(metric_order)


def common_period_yearly_returns(
    common_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Calendar-year compound returns from an already aligned common panel."""
    if not isinstance(common_returns.index, pd.DatetimeIndex):
        raise TypeError("common_returns must have a DatetimeIndex.")
    if common_returns.empty or common_returns.isna().any().any():
        raise ValueError("common_returns must be non-empty and complete.")
    return (
        (1 + common_returns)
        .groupby(common_returns.index.year)
        .prod()
        .sub(1)
        .rename_axis(index="year")
    )


def format_backtest_diagnostics(
    diagnostics: pd.Series,
    uses_optimizer: bool,
) -> pd.Series:
    """Format diagnostics without representing non-applicable values as NaN."""
    formatted = diagnostics.round(4).astype(object).copy()
    if not uses_optimizer:
        formatted.loc["Optimization Fallback Rate"] = "N/A"
    return formatted
