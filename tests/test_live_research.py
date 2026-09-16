from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.config import MAX_POSITION_WEIGHT
from src.data.loader import load_benchmark, load_prices
from src.features.ml_dataset import FEATURE_COLUMNS
from src.live.data_snapshot import create_data_snapshot
from src.live.live_features import FORBIDDEN_LIVE_COLUMNS, generate_current_features
from src.live.paper_evaluation import evaluate_previous_paper_decision
from src.live.paper_state import previous_target_weights, record_paper_decisions
from src.live.run_config import LiveRunConfig
from src.live.runner import run_live_research


def test_snapshot_is_asof_safe_deterministic_and_reports_stale_missing():
    dates = pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-20"])
    prices = pd.DataFrame({"A": [1, 2, 3], "B": [1, np.nan, np.nan]}, index=dates)
    benchmark = pd.Series([1, 2, 3], index=dates)
    original = prices.copy()
    config = LiveRunConfig(pd.Timestamp("2024-01-20"), ("A", "B", "MISSING"))
    first = create_data_snapshot(prices, benchmark, config, "created")
    second = create_data_snapshot(prices, benchmark, config, "different-created")
    pd.testing.assert_frame_equal(prices, original)
    assert first.prices.index.max() <= config.as_of_date
    assert first.benchmark.index.max() <= config.as_of_date
    assert first.metadata["input_hash"] == second.metadata["input_hash"]
    assert first.status.set_index("ticker").loc["B", "reason"] == "stale_price"
    assert first.status.set_index("ticker").loc["MISSING", "reason"] == "missing_ticker"


@pytest.fixture(scope="module")
def live_result(tmp_path_factory):
    prices = load_prices()
    config = LiveRunConfig(pd.Timestamp("2024-12-31"), tuple(prices.columns), output_directory=tmp_path_factory.mktemp("live_runs"))
    return run_live_research(config, paper_update=False)


def test_live_features_and_ml_training_have_no_future_information(live_result):
    features = live_result["features"]
    predictions = live_result["predictions"]
    metadata = live_result["model_metadata"]
    assert not FORBIDDEN_LIVE_COLUMNS.intersection(features.columns)
    assert set(FEATURE_COLUMNS).issubset(features.columns)
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert pd.Timestamp(metadata["latest_training_label_date"]) < pd.Timestamp(metadata["prediction_date"])
    assert predictions.ticker.tolist() == predictions.sort_values(["ml_score", "ticker"], ascending=[False, True]).ticker.tolist()
    assert metadata["hyperparameters"] == {"max_iter": 200, "learning_rate": .05, "max_depth": 3, "random_state": 42}


def test_live_portfolios_preserve_constraints_and_optimizer_status(live_result):
    portfolios = live_result["portfolios"]
    summaries = live_result["summaries"]
    assert np.allclose(portfolios.groupby("strategy").target_weight.sum(), 1)
    assert portfolios.target_weight.max() <= MAX_POSITION_WEIGHT + 1e-10
    assert portfolios.target_weight.gt(0).all()
    assert summaries.optimization_status.notna().all()


def test_repeated_identical_asof_produces_identical_signals(live_result, tmp_path):
    prices = load_prices()
    config = LiveRunConfig(pd.Timestamp("2024-12-31"), tuple(prices.columns), output_directory=tmp_path / "runs")
    repeated = run_live_research(config, paper_update=False)
    pd.testing.assert_frame_equal(live_result["signals"].reset_index(drop=True), repeated["signals"].reset_index(drop=True))
    pd.testing.assert_frame_equal(live_result["predictions"].reset_index(drop=True), repeated["predictions"].reset_index(drop=True))
    assert repeated["snapshot"].prices.index.max() <= pd.Timestamp("2024-12-31")


def _paper_inputs(signal_date):
    portfolios = pd.DataFrame({"strategy": ["s", "s"], "ticker": ["A", "B"], "signal_date": signal_date, "target_weight": [.75, .25]})
    summaries = pd.DataFrame({"strategy": ["s"], "estimated_transaction_cost": [.01]})
    return portfolios, summaries


def test_paper_decisions_are_immutable_and_evaluation_uses_frozen_weights(tmp_path):
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    prices = pd.DataFrame({"A": [100, 110], "B": [100, 80]}, index=dates)
    benchmark = pd.Series([100, 105], index=dates)
    portfolios, summaries = _paper_inputs(dates[0])
    first = record_paper_decisions(tmp_path, "run1", "2024-01-31T00:00:00Z", portfolios, summaries, prices)
    frozen = first.copy()
    transactions = pd.read_csv(tmp_path / "transactions.csv")
    assert np.allclose(transactions.pretrade_weight, 0)
    assert np.allclose(transactions.trade_weight, transactions.target_weight)
    assert np.isclose(transactions.estimated_transaction_cost.sum(), .01)
    with pytest.raises(FileExistsError):
        record_paper_decisions(tmp_path, "run1", "2024-02-01T00:00:00Z", portfolios.assign(target_weight=[.5, .5]), summaries, prices)
    stored = pd.read_csv(tmp_path / "runs/run1/recommendations.csv", parse_dates=["signal_date"])
    pd.testing.assert_frame_equal(stored, frozen, check_dtype=False)
    pretrade = previous_target_weights(tmp_path, prices, dates[1])["s"]
    assert np.isclose(pretrade["A"], (.75 * 1.10) / (.75 * 1.10 + .25 * .80))
    result = evaluate_previous_paper_decision(tmp_path, prices, benchmark, dates[1])
    assert np.isclose(result.loc[0, "gross_return"], .75 * .10 + .25 * -.20)
    assert np.isclose(result.loc[0, "transaction_cost"], .01)
    assert result.loc[0, "realization_date"] == dates[1]


def test_advancing_snapshot_does_not_modify_prior_saved_decision(tmp_path):
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    prices = pd.DataFrame({"A": [100, 110], "B": [100, 90]}, index=dates)
    first_portfolio, summary = _paper_inputs(dates[0])
    record_paper_decisions(tmp_path, "first", "created", first_portfolio, summary, prices)
    original = (tmp_path / "runs/first/recommendations.csv").read_bytes()
    second_portfolio, _ = _paper_inputs(dates[1])
    record_paper_decisions(tmp_path, "second", "later", second_portfolio.assign(target_weight=[.5, .5]), summary, prices)
    assert (tmp_path / "runs/first/recommendations.csv").read_bytes() == original


def test_paper_transactions_include_entries_exits_and_signed_rebalance(tmp_path):
    date = pd.Timestamp("2024-02-29")
    prices = pd.DataFrame({"A": [100], "B": [100], "C": [100]}, index=[date])
    portfolios = pd.DataFrame({
        "strategy": ["s", "s"], "ticker": ["A", "C"],
        "signal_date": [date, date], "target_weight": [.5, .5],
    })
    summaries = pd.DataFrame({"strategy": ["s"], "estimated_transaction_cost": [.002]})
    record_paper_decisions(
        tmp_path, "rebalance", "2024-02-29T00:00:00Z", portfolios,
        summaries, prices, previous_weights={"s": {"A": .8, "B": .2}},
    )
    trades = pd.read_csv(tmp_path / "transactions.csv").set_index("ticker")
    assert np.isclose(trades.loc["A", "trade_weight"], -.3)
    assert np.isclose(trades.loc["B", "trade_weight"], -.2)
    assert np.isclose(trades.loc["C", "trade_weight"], .5)
    assert np.isclose(trades.estimated_transaction_cost.sum(), .002)


def test_live_manifest_and_snapshot_artifacts_exist(live_result):
    required = {"manifest.json", "data_snapshot_metadata.json", "universe_status.csv", "current_features.csv", "momentum_signals.csv", "ml_predictions.csv", "target_portfolios.csv", "portfolio_summary.csv", "cross_strategy_consensus.csv", "risk_flags.csv", "research_report.md"}
    assert required.issubset({p.name for p in Path(live_result["run_dir"]).iterdir()})
