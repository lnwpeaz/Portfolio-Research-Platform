"""Diagnostics derived from recorded optimizer target weights."""

import numpy as np
import pandas as pd


OPTIMIZER_STRATEGIES = ("momentum_max_sharpe", "momentum_min_vol")


def optimizer_weight_stability(
    backtests: dict[str, pd.DataFrame], tolerance: float = 1e-10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, ticker_rows = [], []
    for strategy in OPTIMIZER_STRATEGIES:
        backtest = backtests[strategy].sort_values("date")
        previous = pd.Series(dtype=float)
        all_tickers = sorted({t for weights in backtest.weights for t in weights})
        weight_history = []
        for record in backtest.itertuples(index=False):
            weights = pd.Series(record.weights, dtype=float)
            if not np.isclose(weights.sum(), 1, atol=tolerance):
                raise ValueError("Recorded optimizer weights are not fully invested.")
            positive = weights[weights > tolerance].sort_values(ascending=False)
            hhi = float((weights ** 2).sum())
            union = previous.index.union(weights.index)
            l1 = float((weights.reindex(union, fill_value=0) - previous.reindex(union, fill_value=0)).abs().sum()) if len(previous) else 1.0
            rows.append({
                "strategy": strategy, "date": pd.Timestamp(record.date),
                "number_nonzero_holdings": len(positive), "maximum_weight": positive.iloc[0],
                "top_2_weight": positive.head(2).sum(), "top_3_weight": positive.head(3).sum(),
                "hhi": hhi, "effective_holdings": 1 / hhi,
                "entropy": float(-(positive * np.log(positive)).sum()),
                "turnover": record.turnover, "l1_target_weight_change": l1,
                "optimization_status": record.optimization_status,
                "fallback_indicator": str(record.optimization_status).startswith("fallback_equal:"),
            })
            weight_history.append(weights.reindex(all_tickers, fill_value=0))
            previous = weights
        matrix = pd.DataFrame(weight_history, index=backtest.date).fillna(0)
        for ticker in matrix:
            held = matrix[ticker] > tolerance
            transitions = held.astype(int).diff().fillna(int(held.iloc[0]))
            streak = held.groupby((held != held.shift()).cumsum()).cumsum().max()
            ticker_rows.append({
                "strategy": strategy, "ticker": ticker,
                "held_period_fraction": held.mean(),
                "average_weight_when_held": matrix.loc[held, ticker].mean() if held.any() else 0,
                "unconditional_average_weight": matrix[ticker].mean(),
                "median_weight": matrix[ticker].median(), "maximum_weight": matrix[ticker].max(),
                "largest_position_fraction": (matrix[ticker].eq(matrix.max(axis=1)) & held).mean(),
                "longest_consecutive_holding_streak": int(streak),
                "entries": int((transitions == 1).sum()), "exits": int((transitions == -1).sum()),
            })
    timeseries = pd.DataFrame(rows)
    summaries = []
    for strategy, group in timeseries.groupby("strategy", sort=False):
        summaries.append({
            "strategy": strategy, "rebalance_periods": len(group),
            "average_nonzero_holdings": group.number_nonzero_holdings.mean(),
            "average_max_weight": group.maximum_weight.mean(), "maximum_max_weight": group.maximum_weight.max(),
            "average_top3_weight": group.top_3_weight.mean(), "average_hhi": group.hhi.mean(),
            "average_effective_holdings": group.effective_holdings.mean(),
            "minimum_effective_holdings": group.effective_holdings.min(),
            "average_turnover": group.turnover.mean(), "median_turnover": group.turnover.median(),
            "maximum_turnover": group.turnover.max(),
            "average_l1_weight_change": group.l1_target_weight_change.mean(),
            "fallback_rate": group.fallback_indicator.mean(),
        })
    return timeseries, pd.DataFrame(summaries), pd.DataFrame(ticker_rows)
