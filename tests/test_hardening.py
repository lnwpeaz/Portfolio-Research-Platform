import numpy as np
import pandas as pd
import pytest
from pypfopt.exceptions import OptimizationError

from src.backtest.costs import calculate_one_way_transaction_cost
from src.backtest.engine import run_momentum_backtest, run_score_backtest
from src.evaluation.diagnostics import backtest_diagnostics
from src.evaluation.report import (
    common_period_performance_report,
    common_period_returns,
    common_period_yearly_returns,
    format_backtest_diagnostics,
)


def _daily_prices() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    return pd.DataFrame(
        {
            "A": np.linspace(100, 130, len(dates)),
            "B": np.linspace(100, 120, len(dates)),
        },
        index=dates,
    )


def _score_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "target_date": pd.to_datetime(["2020-02-28", "2020-02-28"]),
            "ticker": ["A", "B"],
            "ml_score": [0.9, 0.8],
            "future_return": [0.1, 0.0],
        }
    )


def test_common_period_reports_use_identical_dates_and_years():
    dates = pd.to_datetime(
        ["2019-12-31", "2020-01-31", "2020-02-28", "2021-01-29"]
    )
    strategies = {
        "early": pd.Series([0.01, 0.02, 0.03, 0.04], index=dates),
        "late": pd.Series([0.05, 0.06, 0.07], index=dates[1:]),
    }
    benchmark = pd.Series([0.0, 0.01, 0.02], index=dates[1:])

    common = common_period_returns(strategies, benchmark)
    performance = common_period_performance_report(common)
    yearly = common_period_yearly_returns(common)

    assert common.index.equals(dates[1:])
    assert set(performance.columns) == {"early", "late", "benchmark"}
    assert yearly.index.tolist() == [2020, 2021]
    assert np.isclose(yearly.loc[2020, "late"], 1.05 * 1.06 - 1)


def test_execution_timing_is_labeled_and_unsupported_modes_fail():
    result = run_score_backtest(
        _score_rows(), top_n=1, one_way_turnover_cost_bps=0,
        execution_timing="same_close",
    )
    assert result["execution_timing"].eq("same_close").all()

    with pytest.raises(ValueError, match="Unsupported execution_timing"):
        run_score_backtest(_score_rows(), execution_timing="next_close")


def test_one_way_cost_convention_is_explicit():
    assert np.isclose(
        calculate_one_way_transaction_cost(
            one_way_turnover=0.5,
            one_way_turnover_cost_bps=10,
        ),
        0.0005,
    )


def test_optimizer_fallback_statuses_are_specific(monkeypatch):
    missing_prices = _daily_prices()
    missing_prices.iloc[20, 0] = np.nan
    missing = run_momentum_backtest(
        missing_prices,
        lookback_months=1,
        top_n=2,
        weighting="min_vol",
        optimization_lookback_days=80,
        minimum_optimization_observations=20,
        max_position_weight=1.0,
    )
    assert "fallback_equal:missing_data" in set(missing["optimization_status"])

    def fail_optimizer(*args, **kwargs):
        raise OptimizationError("expected solver failure")

    monkeypatch.setattr(
        "src.backtest.engine.min_volatility_weights", fail_optimizer
    )
    failed = run_momentum_backtest(
        _daily_prices(),
        lookback_months=1,
        top_n=2,
        weighting="min_vol",
        optimization_lookback_days=80,
        minimum_optimization_observations=20,
        max_position_weight=1.0,
    )
    assert set(failed["optimization_status"]) == {
        "fallback_equal:optimizer_failure"
    }
    assert failed["optimization_fallback_reason"].eq(
        "optimizer_failure"
    ).all()


def test_non_optimizer_diagnostics_display_na():
    backtest = run_score_backtest(_score_rows(), top_n=1)
    numerical = backtest_diagnostics(backtest)
    displayed = format_backtest_diagnostics(numerical, uses_optimizer=False)

    assert np.isnan(numerical["Optimization Fallback Rate"])
    assert displayed["Optimization Fallback Rate"] == "N/A"
