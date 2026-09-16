import numpy as np
import pandas as pd

from src.config import ONE_WAY_TURNOVER_COST_BPS
from src.data.loader import load_benchmark, load_prices
from src.evaluation.cost_sensitivity import (
    apply_cost_scenario,
    transaction_cost_sensitivity,
)
from src.evaluation.execution_sensitivity import one_period_delayed_backtest
from src.evaluation.report import common_period_returns
from src.evaluation.research_pipeline import (
    run_strategy_suite,
    strategy_return_series,
)
from src.evaluation.robustness_common import evaluate_return_series
from src.evaluation.subperiod_analysis import (
    FIXED_SUBPERIODS,
    subperiod_performance,
    validate_fixed_subperiods,
)
from src.evaluation.universe_robustness import (
    MINIMUM_LOO_STRATEGIES,
    leave_one_out_analysis,
    leave_one_out_universes,
)


def _simple_backtest(dates: pd.DatetimeIndex, scale: float = 1.0) -> pd.DataFrame:
    gross = np.array([0.02, -0.01, 0.03, 0.01])[: len(dates)] * scale
    turnover = np.array([1.0, 0.5, 0.25, 0.5])[: len(dates)]
    cost = turnover * ONE_WAY_TURNOVER_COST_BPS / 10_000
    return pd.DataFrame(
        {
            "signal_date": dates - pd.offsets.MonthEnd(1),
            "date": dates,
            "gross_return": gross,
            "turnover": turnover,
            "transaction_cost": cost,
            "portfolio_return": gross - cost,
            "weights": [{"A": 0.5, "B": 0.5}] * len(dates),
        }
    )


def test_leave_one_out_universes_remove_exactly_one_without_mutation():
    original = ["A", "B", "C", "D"]
    snapshot = original.copy()
    scenarios = leave_one_out_universes(original)

    assert original == snapshot
    assert len(scenarios) == len(original)
    for removed, universe in scenarios.items():
        assert removed not in universe
        assert set(universe).issubset(original)
        assert len(universe) == len(original) - 1


def test_leave_one_out_analysis_has_every_scenario_and_aligned_dates():
    tickers = ["A", "B", "C"]
    price_dates = pd.date_range("2019-01-31", periods=6, freq="ME")
    prices = pd.DataFrame(
        {ticker: np.linspace(100, 110, len(price_dates)) for ticker in tickers},
        index=price_dates,
    )
    benchmark_prices = pd.Series(np.linspace(100, 105, 6), index=price_dates)
    realization_dates = price_dates[2:6]
    benchmark_returns = pd.Series([0.01, 0.0, 0.02, -0.01], index=realization_dates)
    baseline = {
        strategy: _simple_backtest(realization_dates)
        for strategy in MINIMUM_LOO_STRATEGIES
    }
    seen_universes = []

    def fake_runner(experimental_prices, _benchmark):
        seen_universes.append(experimental_prices.columns.tolist())
        return {
            strategy: _simple_backtest(realization_dates, scale=0.9)
            for strategy in MINIMUM_LOO_STRATEGIES
        }

    details, _ = leave_one_out_analysis(
        prices,
        benchmark_returns,
        benchmark_prices,
        baseline,
        strategy_runner=fake_runner,
    )
    assert prices.columns.tolist() == tickers
    assert len(seen_universes) == len(tickers)
    assert len(details) == len(tickers) * len(MINIMUM_LOO_STRATEGIES)
    assert set(details["strategy"]) == set(MINIMUM_LOO_STRATEGIES)
    assert details["observations"].eq(len(realization_dates)).all()
    assert (details["start_date"] <= details["end_date"]).all()


def test_fixed_subperiods_do_not_overlap_and_observations_match_boundaries():
    validate_fixed_subperiods()
    dates = pd.date_range("2018-09-30", "2025-12-31", freq="ME")
    common = pd.DataFrame(
        {
            "strategy": np.sin(np.arange(len(dates))) / 100,
            "benchmark": np.cos(np.arange(len(dates))) / 200,
        },
        index=dates,
    )
    result = subperiod_performance(common)
    observed_dates = []
    for period, start, end in FIXED_SUBPERIODS:
        rows = result[result["subperiod"].eq(period)]
        assert rows["start_date"].ge(start).all()
        assert rows["end_date"].le(end).all()
        period_dates = common.loc[start:end].index
        observed_dates.extend(period_dates)
        strategy = rows[rows["strategy"].eq("strategy")].iloc[0]
        benchmark = rows[rows["strategy"].eq("benchmark")].iloc[0]
        assert strategy["observations"] == benchmark["observations"]
    assert len(observed_dates) == len(set(observed_dates))


def test_cost_sensitivity_preserves_config_and_is_mechanically_monotone():
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    backtest = _simple_backtest(dates)
    original = backtest.copy(deep=True)
    configured_cost = ONE_WAY_TURNOVER_COST_BPS
    zero = apply_cost_scenario(backtest, 0)
    high = apply_cost_scenario(backtest, 100)

    assert zero["transaction_cost"].eq(0).all()
    assert (high["transaction_cost"] >= zero["transaction_cost"]).all()
    assert (high["portfolio_return"] <= zero["portfolio_return"]).all()
    pd.testing.assert_frame_equal(backtest, original)
    assert ONE_WAY_TURNOVER_COST_BPS == configured_cost

    benchmark = pd.Series([0.005, -0.005, 0.01, 0.0], index=dates)
    details, _ = transaction_cost_sensitivity(
        {"strategy": backtest}, benchmark, dates, cost_levels=(0, 50, 100)
    )
    assert details.sort_values("cost_bps")["total_transaction_cost"].is_monotonic_increasing


def test_delayed_execution_is_shifted_deterministic_and_uses_formed_weights():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    prices = pd.DataFrame(
        {
            "A": [100, 110, 121, 133.1, 146.41],
            "B": [100, 100, 105, 105, 110],
        },
        index=dates,
    )
    official = pd.DataFrame(
        {
            "signal_date": dates[:3],
            "date": dates[1:4],
            "weights": [
                {"A": 0.5, "B": 0.5},
                {"A": 0.6, "B": 0.4},
                {"A": 0.4, "B": 0.6},
            ],
        }
    )
    delayed = one_period_delayed_backtest(official, prices)

    assert delayed["date"].tolist() == dates[2:5].tolist()
    assert delayed["execution_date"].tolist() == dates[1:4].tolist()
    assert delayed["signal_date"].tolist() == dates[:3].tolist()
    assert delayed["weights"].tolist() == official["weights"].tolist()
    assert len(delayed) <= len(official)
    assert (delayed["signal_date"] < delayed["execution_date"]).all()
    assert (delayed["execution_date"] < delayed["date"]).all()


def test_official_ml_baseline_regression_remains_unchanged():
    prices = load_prices()
    benchmark = load_benchmark()
    backtests, _, benchmark_returns = run_strategy_suite(prices, benchmark)
    strategies = {
        name: strategy_return_series(backtest)
        for name, backtest in backtests.items()
    }
    common = common_period_returns(strategies, benchmark_returns)
    metrics = evaluate_return_series(common["ml"], common["benchmark"])

    assert np.isclose(metrics["annual_return"], 0.3083, atol=5e-4)
    assert np.isclose(metrics["annual_volatility"], 0.2299, atol=5e-4)
    assert np.isclose(metrics["sharpe"], 1.2909, atol=5e-4)
    assert np.isclose(metrics["max_drawdown"], -0.1986, atol=5e-4)
    assert np.isclose(metrics["information_ratio"], 1.2458, atol=5e-4)
