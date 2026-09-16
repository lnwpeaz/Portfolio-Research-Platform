# src/backtest/costs.py

import numpy as np
import pandas as pd


def calculate_turnover(
    old_weights: pd.Series | None,
    new_weights: pd.Series,
) -> float:
    """
    One-way portfolio turnover.

    Example:
    Previous portfolio:
        AAPL 0.5
        MSFT 0.5

    New portfolio:
        AAPL 0.2
        NVDA 0.8

    Turnover measures how much portfolio allocation changed.
    """

    if new_weights.empty:
        raise ValueError("New portfolio weights are empty.")
    if not np.isfinite(new_weights.to_numpy(dtype=float)).all():
        raise ValueError("New portfolio weights must be finite.")

    if old_weights is None:
        return 1.0

    all_assets = old_weights.index.union(
        new_weights.index
    )

    old = old_weights.reindex(
        all_assets,
        fill_value=0.0,
    )

    new = new_weights.reindex(
        all_assets,
        fill_value=0.0,
    )

    turnover = (
        (new - old).abs().sum() / 2
    )

    return float(turnover)


def calculate_one_way_transaction_cost(
    one_way_turnover: float,
    one_way_turnover_cost_bps: float = 10,
) -> float:
    """Convert one-way turnover into a portfolio-return cost.

    ``net_return = gross_return - one_way_turnover
    * one_way_turnover_cost_bps / 10_000``.
    """

    if not np.isfinite(one_way_turnover) or one_way_turnover < 0:
        raise ValueError("one_way_turnover must be finite and non-negative.")
    if (
        not np.isfinite(one_way_turnover_cost_bps)
        or one_way_turnover_cost_bps < 0
    ):
        raise ValueError(
            "one_way_turnover_cost_bps must be finite and non-negative."
        )

    cost_rate = one_way_turnover_cost_bps / 10_000

    return one_way_turnover * cost_rate
