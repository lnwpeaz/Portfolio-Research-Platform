"""Shared orchestration for the repository's implemented strategy suite."""

import pandas as pd

from src.backtest.engine import run_momentum_backtest, run_score_backtest
from src.config import (
    EXECUTION_TIMING_MODE,
    MAX_POSITION_WEIGHT,
    ML_MINIMUM_TRAIN_MONTHS,
    MOMENTUM_12_1_LOOKBACK_MONTHS,
    MOMENTUM_12_1_SKIP_MONTHS,
    MOMENTUM_LOOKBACK_MONTHS,
    ONE_WAY_TURNOVER_COST_BPS,
    OPTIMIZATION_LOOKBACK_DAYS,
    RISK_FREE_RATE,
    TOP_N,
)
from src.data.frequency import last_observation_by_month
from src.features.ml_dataset import build_ml_dataset
from src.models.walk_forward import generate_walk_forward_scores


def strategy_return_series(backtest: pd.DataFrame) -> pd.Series:
    """Extract realization-date net returns from a backtest result."""
    return backtest.set_index("date")["portfolio_return"]


def run_strategy_suite(
    prices: pd.DataFrame,
    benchmark: pd.Series,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.Series]:
    """Run all fixed research strategies without changing their parameters."""
    backtests = {}
    momentum_definitions = (
        ("momentum_3m_equal", "equal", MOMENTUM_LOOKBACK_MONTHS, 0),
        ("momentum_12_1_equal", "equal", MOMENTUM_12_1_LOOKBACK_MONTHS,
         MOMENTUM_12_1_SKIP_MONTHS),
        ("momentum_max_sharpe", "max_sharpe", MOMENTUM_LOOKBACK_MONTHS, 0),
        ("momentum_min_vol", "min_vol", MOMENTUM_LOOKBACK_MONTHS, 0),
    )
    for name, weighting, lookback, skip in momentum_definitions:
        backtests[name] = run_momentum_backtest(
            prices=prices,
            lookback_months=lookback,
            skip_recent_months=skip,
            top_n=TOP_N,
            weighting=weighting,
            optimization_lookback_days=OPTIMIZATION_LOOKBACK_DAYS,
            one_way_turnover_cost_bps=ONE_WAY_TURNOVER_COST_BPS,
            max_position_weight=MAX_POSITION_WEIGHT,
            risk_free_rate=RISK_FREE_RATE,
            execution_timing=EXECUTION_TIMING_MODE,
        )

    dataset = build_ml_dataset(prices, benchmark)
    scores = generate_walk_forward_scores(
        dataset,
        minimum_train_months=ML_MINIMUM_TRAIN_MONTHS,
    )
    backtests["ml"] = run_score_backtest(
        scores,
        top_n=TOP_N,
        one_way_turnover_cost_bps=ONE_WAY_TURNOVER_COST_BPS,
        execution_timing=EXECUTION_TIMING_MODE,
    )
    benchmark_returns = last_observation_by_month(benchmark).pct_change(
        fill_method=None
    )
    return backtests, scores, benchmark_returns
