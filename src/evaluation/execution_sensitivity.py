"""Monthly timing sensitivity without claiming next-open execution realism."""

import numpy as np
import pandas as pd

from src.backtest.costs import (
    calculate_one_way_transaction_cost,
    calculate_turnover,
)
from src.config import ONE_WAY_TURNOVER_COST_BPS
from src.data.frequency import last_observation_by_month
from src.evaluation.research_pipeline import strategy_return_series
from src.evaluation.robustness_common import aligned_pair, evaluate_return_series


def _drift_delayed_weights(
    weights: pd.Series,
    asset_returns: pd.Series,
) -> pd.Series:
    ending_values = weights * (1 + asset_returns.reindex(weights.index))
    total = ending_values.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Invalid delayed portfolio ending value.")
    return ending_values / total


def one_period_delayed_backtest(
    official_backtest: pd.DataFrame,
    prices: pd.DataFrame,
    one_way_turnover_cost_bps: float = ONE_WAY_TURNOVER_COST_BPS,
) -> pd.DataFrame:
    """Apply weights formed at *t* to returns from *t+1* through *t+2*.

    Official rows contain weights formed with information at ``signal_date=t``
    and a same-close return realized at ``date=t+1``. In this experiment those
    exact weights become effective at ``execution_date=t+1`` and earn the next
    monthly return, realized at ``date=t+2``. No *t+1* or *t+2* data are used to
    construct the weights. This is a monthly lag sensitivity test, not a
    next-open market-microstructure simulation.
    """
    monthly_prices = last_observation_by_month(prices)
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    monthly_dates = monthly_prices.index.tolist()
    date_positions = {date: position for position, date in enumerate(monthly_dates)}

    rows = []
    pretrade_weights = None
    for record in official_backtest.sort_values("date").itertuples(index=False):
        execution_date = pd.Timestamp(record.date)
        position = date_positions.get(execution_date)
        if position is None or position + 1 >= len(monthly_dates):
            continue
        realization_date = monthly_dates[position + 1]
        weights = pd.Series(record.weights, dtype=float)
        realized_returns = monthly_returns.loc[realization_date, weights.index]
        if realized_returns.isna().any():
            raise ValueError("Missing delayed realized asset return.")

        turnover = calculate_turnover(pretrade_weights, weights)
        transaction_cost = calculate_one_way_transaction_cost(
            turnover,
            one_way_turnover_cost_bps=one_way_turnover_cost_bps,
        )
        gross_return = float((weights * realized_returns).sum())
        rows.append(
            {
                "signal_date": pd.Timestamp(record.signal_date),
                "execution_date": execution_date,
                "date": realization_date,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "turnover": turnover,
                "portfolio_return": gross_return - transaction_cost,
                "weights": weights.to_dict(),
                "asset_returns": realized_returns.to_dict(),
                "execution_scenario": "one_period_delay",
            }
        )
        pretrade_weights = _drift_delayed_weights(weights, realized_returns)
    return pd.DataFrame(rows)


def execution_timing_sensitivity(
    backtests: dict[str, pd.DataFrame],
    prices: pd.DataFrame,
    benchmark_returns: pd.Series,
    evaluation_dates: pd.Index,
) -> pd.DataFrame:
    """Compare official same-close and one-period-delay paths on paired dates."""
    rows = []
    for strategy, official in backtests.items():
        delayed = one_period_delayed_backtest(official, prices)
        same_returns = strategy_return_series(official).reindex(evaluation_dates)
        delayed_returns = strategy_return_series(delayed).reindex(evaluation_dates)
        paired = aligned_pair(same_returns, delayed_returns, benchmark_returns)
        same = evaluate_return_series(paired["first"], paired["benchmark"])
        lagged = evaluate_return_series(paired["second"], paired["benchmark"])
        deltas = {
            "annual_return_delta": lagged["annual_return"] - same["annual_return"],
            "sharpe_delta": lagged["sharpe"] - same["sharpe"],
            "information_ratio_delta": (
                lagged["information_ratio"] - same["information_ratio"]
            ),
        }
        rows.append(
            {
                "strategy": strategy,
                "execution_scenario": "same_close",
                **same,
                "annual_return_delta": 0.0,
                "sharpe_delta": 0.0,
                "information_ratio_delta": 0.0,
            }
        )
        rows.append(
            {
                "strategy": strategy,
                "execution_scenario": "one_period_delay",
                **lagged,
                **deltas,
            }
        )
    return pd.DataFrame(rows)
