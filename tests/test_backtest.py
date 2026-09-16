import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import run_momentum_backtest, run_score_backtest


def test_momentum_dates_are_realization_dates_and_turnover_uses_drifted_weights():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31", "2020-04-30"])
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, 121.0, 133.1],
            "B": [100.0, 105.0, 105.0, 105.0],
            "C": [100.0, 90.0, 80.0, 70.0],
        },
        index=dates,
    )

    result = run_momentum_backtest(
        prices, lookback_months=1, top_n=2, max_position_weight=1.0,
        one_way_turnover_cost_bps=0,
    )

    assert result.loc[0, "signal_date"] == pd.Timestamp("2020-02-29")
    assert result.loc[0, "date"] == pd.Timestamp("2020-03-31")
    expected_second_turnover = abs(0.5 - (0.5 * 1.1 / 1.05))
    assert np.isclose(result.loc[1, "turnover"], expected_second_turnover)


def test_optimizer_falls_back_until_requested_history_is_available():
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    prices = pd.DataFrame(
        {
            "A": np.linspace(100, 130, len(dates)),
            "B": np.linspace(100, 120, len(dates)),
        },
        index=dates,
    )
    result = run_momentum_backtest(
        prices,
        lookback_months=1,
        top_n=2,
        weighting="min_vol",
        optimization_lookback_days=80,
        minimum_optimization_observations=80,
        max_position_weight=1.0,
    )
    assert result.loc[0, "optimization_status"] == (
        "fallback_equal:insufficient_history"
    )


def test_momentum_does_not_replace_selected_asset_using_future_availability():
    dates = pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"])
    prices = pd.DataFrame(
        {"A": [100.0, 120.0, np.nan], "B": [100.0, 110.0, 121.0]},
        index=dates,
    )
    with pytest.raises(ValueError, match="Missing next-period returns"):
        run_momentum_backtest(
            prices, lookback_months=1, top_n=1,
            max_position_weight=1.0,
        )


def test_score_backtest_ranks_before_checking_realized_returns():
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 2),
            "target_date": pd.to_datetime(["2020-02-29"] * 2),
            "ticker": ["A", "B"],
            "ml_score": [0.9, 0.8],
            "future_return": [np.nan, 0.1],
        }
    )
    with pytest.raises(ValueError, match="Missing next-period returns"):
        run_score_backtest(scores, top_n=1)


def test_score_backtest_indexes_return_by_target_date():
    scores = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "target_date": pd.to_datetime(["2020-02-29", "2020-02-29"]),
            "ticker": ["A", "B"],
            "ml_score": [0.9, 0.8],
            "future_return": [0.1, 0.0],
        }
    )
    result = run_score_backtest(scores, top_n=1, one_way_turnover_cost_bps=0)
    assert result.loc[0, "signal_date"] == pd.Timestamp("2020-01-31")
    assert result.loc[0, "date"] == pd.Timestamp("2020-02-29")
