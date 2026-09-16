import numpy as np
import pandas as pd

from src.data.loader import load_benchmark, load_prices
from src.evaluation.bootstrap import moving_block_indices, paired_block_bootstrap
from src.evaluation.contribution_stress import contribution_concentration, contribution_counterfactuals
from src.evaluation.ml_stability import ml_signal_decay
from src.evaluation.optimizer_diagnostics import optimizer_estimator_diagnostics
from src.evaluation.optimizer_stability import optimizer_weight_stability
from src.evaluation.report import common_period_returns
from src.evaluation.research_pipeline import run_strategy_suite, strategy_return_series
from src.evaluation.return_outliers import return_outlier_analysis
from src.evaluation.robustness_common import evaluate_return_series
from src.evaluation.rolling_robustness import rolling_robustness


def test_moving_block_bootstrap_is_deterministic_paired_and_valid():
    first = moving_block_indices(17, 20, 6, 42)
    second = moving_block_indices(17, 20, 6, 42)
    different = moving_block_indices(17, 20, 6, 43)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)
    assert first.shape == (20, 17)
    assert first.min() >= 0 and first.max() < 17
    dates = pd.date_range("2020-01-31", periods=17, freq="ME")
    strategy = pd.Series(np.arange(17) / 100, index=dates)
    benchmark = strategy - .01
    samples = paired_block_bootstrap(strategy, benchmark, 20, 6, 42)
    assert len(samples) == 20
    assert np.allclose(samples.annual_active_return, .12)
    assert samples.sharpe.quantile(.05) <= samples.sharpe.median() <= samples.sharpe.quantile(.95)


def _backtest(dates):
    return pd.DataFrame({
        "signal_date": dates - pd.offsets.MonthEnd(1), "date": dates,
        "weights": [{"A": .6, "B": .4}, {"A": .4, "B": .6}],
        "asset_returns": [{"A": .1, "B": -.05}, {"A": .05, "B": .1}],
        "gross_return": [.04, .08], "transaction_cost": [.001, .001],
        "portfolio_return": [.039, .079], "turnover": [1, .2],
        "optimization_status": ["optimized", "optimized"],
    })


def test_contribution_concentration_reconciles_gross_and_counterfactual_is_typed():
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    backtest = _backtest(dates)
    summary, tickers = contribution_concentration({"strategy": backtest}, dates)
    assert np.isclose(tickers.total_contribution.sum(), backtest.gross_return.sum())
    assert tickers.sort_values("rank").ticker.tolist() == ["A", "B"]
    benchmark = pd.Series(0.0, index=dates)
    counter = contribution_counterfactuals({"strategy": backtest}, tickers, benchmark, dates)
    assert counter.counterfactual_type.eq("attribution_counterfactual").all()
    assert counter.annual_return.isna().all()


def test_outlier_removal_is_exact_and_does_not_mutate_input():
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    common = pd.DataFrame({"strategy": np.arange(12) / 100, "benchmark": 0.0}, index=dates)
    original = common.copy()
    diagnostics, counter = return_outlier_analysis(common)
    pd.testing.assert_frame_equal(common, original)
    row = counter[(counter.strategy == "strategy") & (counter.counterfactual == "remove_best_3_months")].iloc[0]
    assert row.observations == 9
    assert diagnostics.loc[diagnostics.strategy.eq("strategy"), "best_month"].iloc[0] == dates[-1]


def test_optimizer_weight_and_ticker_stability_identities():
    dates = pd.date_range("2020-01-31", periods=2, freq="ME")
    backtests = {name: _backtest(dates) for name in ("momentum_max_sharpe", "momentum_min_vol")}
    timeseries, _, tickers = optimizer_weight_stability(backtests)
    assert timeseries.hhi.between(0, 1).all()
    assert np.allclose(timeseries.effective_holdings, 1 / timeseries.hhi)
    assert np.allclose(timeseries.maximum_weight, .6)
    assert timeseries.top_3_weight.between(timeseries.maximum_weight, 1).all()
    assert tickers.held_period_fraction.between(0, 1).all()


def test_estimator_diagnostics_are_finite_and_do_not_mutate_prices():
    price_dates = pd.date_range("2019-01-01", periods=260, freq="B")
    prices = pd.DataFrame({"A": 100 * np.cumprod(np.repeat(1.001, 260)), "B": 100 * np.cumprod(np.resize([1.002, .999], 260))}, index=price_dates)
    original = prices.copy()
    date = price_dates[-1]
    row = pd.DataFrame({"signal_date": [date], "date": [date], "weights": [{"A": .5, "B": .5}], "optimization_status": ["optimized"]})
    details, _ = optimizer_estimator_diagnostics({"momentum_max_sharpe": row, "momentum_min_vol": row}, prices)
    pd.testing.assert_frame_equal(prices, original)
    valid = details.dropna(subset=["covariance_condition_number"])
    assert np.isfinite(valid.minimum_eigenvalue).all()
    assert (valid.covariance_rank <= valid.assets).all()
    assert (valid.covariance_condition_number >= 0).all()


def test_ml_decay_uses_original_scores_at_every_future_horizon():
    dates = pd.date_range("2020-01-31", periods=5, freq="ME")
    prices = pd.DataFrame({"A": [100, 110, 121, 133.1, 146.41], "B": [100, 100, 100, 100, 100]}, index=dates)
    benchmark = pd.Series([100] * 5, index=dates)
    scores = pd.DataFrame({"date": [dates[0], dates[0]], "ticker": ["A", "B"], "ml_score": [.9, .1]})
    decay = ml_signal_decay(scores, prices, benchmark, top_n=1)
    assert decay.horizon_months.tolist() == [1, 2, 3]
    assert np.allclose(decay.mean_rank_ic, 1)
    assert (decay.average_top_n_spread > 0).all()


def test_rolling_robustness_uses_exact_complete_windows_and_matching_dates():
    dates = pd.date_range("2018-01-31", periods=30, freq="ME")
    common = pd.DataFrame({"strategy": .01, "benchmark": .005}, index=dates)
    details, summary = rolling_robustness(common, window=24)
    assert details.observations.eq(24).all()
    assert details.groupby("strategy").date.apply(lambda x: x.is_monotonic_increasing).all()
    assert summary.loc[0, "positive_active_return_ratio"] == 1


def test_phase2_official_ml_and_phase1_common_period_regression():
    prices, benchmark = load_prices(), load_benchmark()
    backtests, _, benchmark_returns = run_strategy_suite(prices, benchmark)
    common = common_period_returns({k: strategy_return_series(v) for k, v in backtests.items()}, benchmark_returns)
    metrics = evaluate_return_series(common.ml, common.benchmark)
    assert len(common) == 88
    assert np.isclose(metrics["annual_return"], .3083, atol=5e-4)
    assert np.isclose(metrics["annual_volatility"], .2299, atol=5e-4)
    assert np.isclose(metrics["sharpe"], 1.2909, atol=5e-4)
    assert np.isclose(metrics["max_drawdown"], -.1986, atol=5e-4)
    assert np.isclose(metrics["information_ratio"], 1.2458, atol=5e-4)
