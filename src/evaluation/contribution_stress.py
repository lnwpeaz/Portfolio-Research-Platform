"""Ex-post contribution concentration and explicitly typed counterfactuals."""

import numpy as np
import pandas as pd

from src.evaluation.attribution import period_contribution_attribution
from src.evaluation.research_pipeline import strategy_return_series
from src.evaluation.robustness_common import evaluate_return_series


def contribution_concentration(
    backtests: dict[str, pd.DataFrame], dates: pd.Index
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate gross arithmetic contribution; costs remain excluded."""
    ticker_rows, summary_rows = [], []
    for strategy, backtest in backtests.items():
        period = period_contribution_attribution(backtest, dates=dates)
        values = period.groupby("ticker")["contribution"].sum().sort_values(ascending=False)
        positive_total = values.clip(lower=0).sum()
        net_total = values.sum()
        for rank, (ticker, contribution) in enumerate(values.items(), 1):
            ticker_rows.append({
                "strategy": strategy, "ticker": ticker, "total_contribution": contribution,
                "rank": rank,
                "share_of_positive_contribution": contribution / positive_total if contribution > 0 and positive_total > 0 else 0.0,
                "share_of_net_arithmetic_contribution": contribution / net_total if not np.isclose(net_total, 0) else np.nan,
            })
        positives = values[values > 0]
        top = {n: positives.head(n).sum() for n in (1, 2, 3, 5)}
        shares = {n: top[n] / positive_total if positive_total > 0 else np.nan for n in (1, 2, 3, 5)}
        positive_shares = positives / positive_total if positive_total > 0 else pd.Series(dtype=float)
        summary_rows.append({
            "strategy": strategy, "net_arithmetic_contribution": net_total,
            "positive_contribution_total": positive_total,
            "largest_contributor": values.index[0], "largest_contributor_contribution": values.iloc[0],
            "top_2_contribution": top[2], "top_3_contribution": top[3], "top_5_contribution": top[5],
            "bottom_contributor": values.index[-1], "bottom_3_contribution": values.tail(3).sum(),
            "top_1_share": shares[1], "top_2_share": shares[2],
            "top_3_share": shares[3], "top_5_share": shares[5],
            "positive_contribution_hhi": float((positive_shares ** 2).sum()) if len(positive_shares) else np.nan,
            "share_denominator": "sum_of_positive_security_contributions",
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(ticker_rows)


def contribution_counterfactuals(
    backtests: dict[str, pd.DataFrame],
    ticker_summary: pd.DataFrame,
    benchmark_returns: pd.Series,
    dates: pd.Index,
    rerun_results: dict[tuple[str, int], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Label attribution subtraction separately from ex-post strategy reruns."""
    rows = []
    rerun_results = rerun_results or {}
    for strategy, backtest in backtests.items():
        baseline_returns = strategy_return_series(backtest).reindex(dates).dropna()
        baseline = evaluate_return_series(baseline_returns, benchmark_returns)
        ranked = ticker_summary[ticker_summary.strategy.eq(strategy)].sort_values("rank")
        for n in (1, 2, 3):
            removed = ranked.head(n)
            tickers = "|".join(removed.ticker)
            rows.append({
                "strategy": strategy, "counterfactual_type": "attribution_counterfactual",
                "removed_tickers": tickers, "annual_return": np.nan, "sharpe": np.nan,
                "max_drawdown": np.nan, "information_ratio": np.nan,
                "delta_vs_baseline": -removed.total_contribution.sum(),
                "delta_definition": "change_in_net_arithmetic_contribution_only",
            })
            rerun = rerun_results.get((strategy, n))
            if rerun is not None:
                metrics = evaluate_return_series(
                    strategy_return_series(rerun).reindex(dates).dropna(), benchmark_returns
                )
                rows.append({
                    "strategy": strategy, "counterfactual_type": "rerun_counterfactual",
                    "removed_tickers": tickers, "annual_return": metrics["annual_return"],
                    "sharpe": metrics["sharpe"], "max_drawdown": metrics["max_drawdown"],
                    "information_ratio": metrics["information_ratio"],
                    "delta_vs_baseline": metrics["sharpe"] - baseline["sharpe"],
                    "delta_definition": "sharpe_delta_vs_baseline",
                })
    return pd.DataFrame(rows)
