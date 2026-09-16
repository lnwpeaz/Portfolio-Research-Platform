import numpy as np
import pandas as pd

from src.evaluation.attribution import (
    contribution_by_ticker,
    period_contribution_attribution,
)
from src.evaluation.concentration import (
    concentration_summary,
    concentration_timeseries,
)
from src.evaluation.drawdown import drawdown_summary
from src.evaluation.ranking_metrics import ranking_diagnostics_timeseries
from src.evaluation.report import common_period_returns
from src.evaluation.risk import rolling_risk_report
from src.evaluation.yearly_analysis import common_period_yearly_analysis
from src.features.momentum import calculate_monthly_momentum_signal


def test_12_1_momentum_excludes_recent_and_realization_months():
    dates = pd.date_range("2020-01-31", periods=14, freq="ME")
    prices = pd.DataFrame({"A": np.arange(100.0, 114.0)}, index=dates)
    signal = calculate_monthly_momentum_signal(
        prices, lookback_months=12, skip_recent_months=1
    )
    expected = prices.loc[dates[11], "A"] / prices.loc[dates[0], "A"] - 1
    assert np.isclose(signal.loc[dates[12], "A"], expected)

    changed = prices.copy()
    changed.loc[dates[12]:, "A"] *= 100
    changed_signal = calculate_monthly_momentum_signal(
        changed, lookback_months=12, skip_recent_months=1
    )
    assert np.isclose(changed_signal.loc[dates[12], "A"], expected)


def test_all_research_strategies_share_common_dates():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    names = (
        "momentum_3m_equal",
        "momentum_12_1_equal",
        "momentum_max_sharpe",
        "momentum_min_vol",
        "ml",
    )
    strategies = {
        name: pd.Series(0.01, index=dates[position:])
        for position, name in enumerate(names)
    }
    common = common_period_returns(
        strategies,
        pd.Series(0.0, index=dates),
    )
    assert common.index.tolist() == [dates[-1]]
    assert not common.isna().any().any()


def test_rolling_metrics_handle_short_samples_safely():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    common = pd.DataFrame(
        {"strategy": [0.01] * 5, "benchmark": [0.005] * 5},
        index=dates,
    )
    result = rolling_risk_report(common, window=12)
    metric_columns = result.columns.difference(["date", "strategy"])
    assert len(result) == 10
    assert result[metric_columns].isna().all().all()


def test_hhi_effective_holdings_and_unavailable_summary():
    backtest = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31")],
            "weights": [{"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}],
        }
    )
    result = concentration_timeseries(backtest)
    assert np.isclose(result.loc[0, "hhi"], 0.25)
    assert np.isclose(result.loc[0, "effective_holdings"], 4.0)
    assert concentration_summary(pd.DataFrame())["Average HHI"] == "N/A"


def test_drawdown_episode_start_trough_recovery_and_duration():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    returns = pd.Series([0.10, -0.20, 0.10, 0.20], index=dates)
    summary = drawdown_summary(returns)
    assert summary["Drawdown Start"] == dates[0]
    assert summary["Trough Date"] == dates[1]
    assert summary["Recovery Date"] == dates[3]
    assert summary["Drawdown Duration"] == 3
    assert summary["Longest Drawdown Duration"] == 3


def test_security_contributions_reconcile_and_cost_is_separate():
    backtest = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31")],
            "weights": [{"A": 0.6, "B": 0.4}],
            "asset_returns": [{"A": 0.10, "B": -0.05}],
            "gross_return": [0.04],
            "transaction_cost": [0.001],
        }
    )
    period = period_contribution_attribution(backtest)
    attribution = contribution_by_ticker(backtest, "strategy")
    assert np.isclose(period["contribution"].sum(), backtest.loc[0, "gross_return"])
    securities = attribution[attribution["attribution_type"].eq("security")]
    costs = attribution[attribution["attribution_type"].eq("transaction_cost")]
    assert np.isclose(securities["contribution"].sum(), 0.04)
    assert costs["ticker"].tolist() == ["TRANSACTION_COST"]
    assert np.isclose(costs["contribution"].iloc[0], -0.001)


def test_rank_ic_timeseries_and_rolling_average():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    scores = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "ticker": ["A", "B"] * 3,
            "ml_score": [0.9, 0.1, 0.8, 0.2, 0.7, 0.3],
            "future_return": [0.1, 0.0, 0.2, -0.1, 0.05, -0.02],
            "target": [1, 0, 1, 0, 1, 0],
        }
    )
    result = ranking_diagnostics_timeseries(scores, top_n=1, rolling_window=2)
    assert np.allclose(result["rank_ic"], 1.0)
    assert np.isnan(result.loc[0, "rolling_mean_rank_ic"])
    assert np.allclose(result.loc[1:, "rolling_mean_rank_ic"], 1.0)
    assert np.allclose(result["top_n_hit_rate"], 1.0)


def test_yearly_analysis_marks_partial_year_and_calculates_excess():
    dates = pd.to_datetime(["2020-11-30", "2020-12-31", "2021-01-31"])
    common = pd.DataFrame(
        {
            "alpha": [0.10, 0.00, -0.02],
            "beta": [0.00, 0.05, 0.03],
            "benchmark": [0.01, 0.01, 0.01],
        },
        index=dates,
    )
    result = common_period_yearly_analysis(common)
    assert result.loc[2020, "partial_year"]
    assert result.loc[2020, "best_strategy"] == "alpha"
    assert result.loc[2020, "worst_strategy"] == "beta"
    assert np.isclose(
        result.loc[2020, "alpha_excess_vs_benchmark"],
        result.loc[2020, "alpha"] - result.loc[2020, "benchmark"],
    )
