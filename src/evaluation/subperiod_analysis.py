"""Fixed, calendar-driven subperiod analysis (not economic regime analysis)."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import performance_summary
from src.evaluation.robustness_common import evaluate_return_series


FIXED_SUBPERIODS = (
    ("period_1", pd.Timestamp("2018-09-28"), pd.Timestamp("2019-12-31")),
    ("period_2", pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31")),
    ("period_3", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    ("period_4", pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31")),
)


def validate_fixed_subperiods() -> None:
    """Raise if configured subperiod boundaries overlap or are invalid."""
    previous_end = None
    for _, start, end in FIXED_SUBPERIODS:
        if start > end:
            raise ValueError("Subperiod starts after it ends.")
        if previous_end is not None and start <= previous_end:
            raise ValueError("Fixed subperiods overlap.")
        previous_end = end


def subperiod_performance(
    common_returns: pd.DataFrame,
    benchmark_name: str = "benchmark",
) -> pd.DataFrame:
    """Evaluate every series within four predetermined calendar intervals."""
    validate_fixed_subperiods()
    if benchmark_name not in common_returns:
        raise ValueError(f"Missing benchmark column: {benchmark_name!r}.")
    rows = []
    for period, requested_start, requested_end in FIXED_SUBPERIODS:
        subset = common_returns.loc[requested_start:requested_end].dropna()
        if subset.empty:
            continue
        benchmark = subset[benchmark_name]
        for strategy in common_returns.columns:
            if strategy == benchmark_name:
                metrics = performance_summary(benchmark)
                row = {
                    "annual_return": metrics["Annual Return"],
                    "annual_volatility": metrics["Annual Volatility"],
                    "sharpe": metrics["Sharpe Ratio"],
                    "sortino": metrics["Sortino Ratio"],
                    "max_drawdown": metrics["Max Drawdown"],
                    "calmar": metrics["Calmar Ratio"],
                    "annual_active_return": np.nan,
                    "tracking_error": np.nan,
                    "information_ratio": np.nan,
                    "beta": np.nan,
                    "annual_alpha": np.nan,
                }
            else:
                row = evaluate_return_series(subset[strategy], benchmark)
            rows.append(
                {
                    "subperiod": period,
                    "requested_start": requested_start,
                    "requested_end": requested_end,
                    "strategy": strategy,
                    "start_date": subset.index.min(),
                    "end_date": subset.index.max(),
                    "observations": len(subset),
                    **{
                        key: value
                        for key, value in row.items()
                        if key not in {"start_date", "end_date", "observations"}
                    },
                }
            )
    return pd.DataFrame(rows)


def subperiod_stability_summary(
    performance: pd.DataFrame,
    benchmark_name: str = "benchmark",
) -> pd.DataFrame:
    """Summarize cross-subperiod consistency for non-benchmark strategies."""
    benchmark = performance[performance["strategy"].eq(benchmark_name)].set_index(
        "subperiod"
    )
    rows = []
    for strategy, group in performance[
        ~performance["strategy"].eq(benchmark_name)
    ].groupby("strategy", sort=False):
        group = group.set_index("subperiod")
        matched_benchmark = benchmark.reindex(group.index)
        rows.append(
            {
                "strategy": strategy,
                "total_subperiods": len(group),
                "positive_active_return_subperiods": int(
                    (group["annual_active_return"] > 0).sum()
                ),
                "positive_active_return_ratio": float(
                    (group["annual_active_return"] > 0).mean()
                ),
                "benchmark_beating_return_subperiods": int(
                    (group["annual_return"] > matched_benchmark["annual_return"]).sum()
                ),
                "benchmark_beating_sharpe_subperiods": int(
                    (group["sharpe"] > matched_benchmark["sharpe"]).sum()
                ),
                "best_subperiod_sharpe": group["sharpe"].max(),
                "worst_subperiod_sharpe": group["sharpe"].min(),
                "mean_subperiod_sharpe": group["sharpe"].mean(),
                "std_subperiod_sharpe": group["sharpe"].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)
