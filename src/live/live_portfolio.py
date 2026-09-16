"""Current target portfolios using unchanged selection and weight construction."""

import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models

from src.backtest.costs import calculate_one_way_transaction_cost, calculate_turnover
from src.backtest.engine import _construct_weights
from src.config import MAX_POSITION_WEIGHT, OPTIMIZATION_LOOKBACK_DAYS, RISK_FREE_RATE
from src.portfolio.optimizer import equal_weight_weights


STRATEGY_NAMES = ("momentum_3m_equal", "momentum_12_1_equal", "momentum_max_sharpe", "momentum_min_vol", "ml")


def _summary_risk(history: pd.DataFrame, weights: pd.Series, include_return: bool):
    if len(history) < OPTIMIZATION_LOOKBACK_DAYS or history.isna().any().any():
        return np.nan, np.nan, np.nan
    covariance = risk_models.sample_cov(history, frequency=252)
    volatility = float(np.sqrt(weights @ covariance.loc[weights.index, weights.index] @ weights))
    condition = float(np.linalg.cond(covariance))
    expected = np.nan
    if include_return:
        mu = expected_returns.mean_historical_return(history, frequency=252)
        expected = float(weights @ mu.reindex(weights.index))
    return expected, volatility, condition


def construct_current_portfolios(
    prices: pd.DataFrame, signals: pd.DataFrame, predictions: pd.DataFrame,
    top_n: int, transaction_cost_bps: float,
    previous_weights: dict[str, dict] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_date = pd.Timestamp(signals.signal_date.max())
    previous_weights = previous_weights or {}
    rank_specs = {
        "momentum_3m_equal": (signals, "rank_3m", "equal"),
        "momentum_12_1_equal": (signals, "rank_12_1", "equal"),
        "momentum_max_sharpe": (signals, "rank_3m", "max_sharpe"),
        "momentum_min_vol": (signals, "rank_3m", "min_vol"),
        "ml": (predictions.rename(columns={"prediction_date": "signal_date"}), "ml_rank", "equal"),
    }
    portfolio_rows, summaries = [], []
    for strategy, (ranking, rank_column, weighting) in rank_specs.items():
        eligible_ranking = ranking
        if "signal_valid" in eligible_ranking:
            eligible_ranking = eligible_ranking[eligible_ranking.signal_valid]
        selected_frame = eligible_ranking.dropna(subset=[rank_column]).sort_values([rank_column, "ticker"]).head(top_n)
        selected = selected_frame.ticker.tolist()
        history = prices.loc[:signal_date, selected].tail(OPTIMIZATION_LOOKBACK_DAYS)
        if weighting == "equal":
            weights, status = equal_weight_weights(selected), "equal"
        else:
            weights, status = _construct_weights(
                weighting, selected, history, MAX_POSITION_WEIGHT,
                OPTIMIZATION_LOOKBACK_DAYS, RISK_FREE_RATE,
            )
        positive_weights = weights[weights > 1e-10]
        for ticker, weight in positive_weights.items():
            rank = float(selected_frame.set_index("ticker").loc[ticker, rank_column])
            portfolio_rows.append({"strategy": strategy, "ticker": ticker, "signal_date": signal_date, "target_weight": weight, "rank_or_reason": rank, "optimization_status": status})
        old = pd.Series(previous_weights.get(strategy, {}), dtype=float) if strategy in previous_weights else None
        turnover = calculate_turnover(old, weights)
        cost = calculate_one_way_transaction_cost(turnover, transaction_cost_bps)
        hhi = float((weights ** 2).sum())
        exp_return, exp_vol, condition = _summary_risk(history, weights, weighting == "max_sharpe") if weighting != "equal" else (np.nan, np.nan, np.nan)
        summaries.append({
            "strategy": strategy, "number_holdings": len(positive_weights), "effective_holdings": 1 / hhi,
            "max_weight": weights.max(), "top3_weight": weights.nlargest(3).sum(),
            "estimated_turnover_vs_previous": turnover, "estimated_transaction_cost": cost,
            "deployment_status": "rebalance" if old is not None else "initial_deployment",
            "optimization_status": status, "expected_return": exp_return,
            "expected_volatility": exp_vol, "covariance_condition_number": condition,
        })
    return pd.DataFrame(portfolio_rows), pd.DataFrame(summaries)
