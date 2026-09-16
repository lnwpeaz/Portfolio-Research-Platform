"""Rolling complete-window strategy robustness diagnostics."""

import pandas as pd

from src.evaluation.metrics import performance_summary, active_performance_summary


def rolling_robustness(
    common_returns: pd.DataFrame, window: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if window < 2 or len(common_returns) < window or common_returns.isna().any().any():
        raise ValueError("A complete return panel with enough observations is required.")
    rows = []
    benchmark = common_returns["benchmark"]
    for strategy in common_returns:
        for end in range(window - 1, len(common_returns)):
            values = common_returns[strategy].iloc[end - window + 1:end + 1]
            bench = benchmark.reindex(values.index)
            absolute = performance_summary(values)
            active = active_performance_summary(values, bench) if strategy != "benchmark" else None
            rows.append({
                "strategy": strategy, "start_date": values.index[0], "date": values.index[-1],
                "observations": len(values), "annual_return": absolute["Annual Return"],
                "annual_volatility": absolute["Annual Volatility"], "sharpe": absolute["Sharpe Ratio"],
                "max_drawdown": absolute["Max Drawdown"],
                "annual_active_return": active["Annual Active Return"] if active is not None else pd.NA,
                "tracking_error": active["Tracking Error"] if active is not None else pd.NA,
                "information_ratio": active["Information Ratio"] if active is not None else pd.NA,
            })
    details = pd.DataFrame(rows)
    benchmark_sharpe = details[details.strategy.eq("benchmark")].set_index("date")["sharpe"]
    summaries = []
    for strategy, group in details[~details.strategy.eq("benchmark")].groupby("strategy", sort=False):
        matched_benchmark = benchmark_sharpe.reindex(group.date).to_numpy()
        summaries.append({
            "strategy": strategy, "rolling_windows": len(group),
            "positive_sharpe_ratio": (group.sharpe > 0).mean(),
            "sharpe_above_benchmark_ratio": (group.sharpe.to_numpy() > matched_benchmark).mean(),
            "positive_active_return_ratio": (pd.to_numeric(group.annual_active_return) > 0).mean(),
            "minimum_rolling_sharpe": group.sharpe.min(),
            "median_rolling_sharpe": group.sharpe.median(),
            "maximum_rolling_sharpe": group.sharpe.max(),
            "minimum_rolling_information_ratio": pd.to_numeric(group.information_ratio).min(),
        })
    return details, pd.DataFrame(summaries)
