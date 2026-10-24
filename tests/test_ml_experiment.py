"""Behavioral guards for frozen ML selection and chronological evaluation."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from src.features.ml_dataset import build_ml_dataset
from src.models.walk_forward import generate_walk_forward_scores
from src.evaluation import ml_experiment


def _dataset():
    rng = np.random.default_rng(17)
    dates = pd.date_range("2017-01-31", periods=64, freq="ME")
    returns = rng.normal(0.012, 0.07, (64, 6))
    prices = pd.DataFrame(
        100 * np.cumprod(1 + returns, axis=0),
        index=dates, columns=list("ABCDEF"),
    )
    benchmark = pd.Series(
        100 * np.cumprod(1 + returns.mean(axis=1)), index=dates,
    )
    return build_ml_dataset(prices, benchmark)


def _validation_returns():
    dates = pd.date_range("2021-01-31", "2023-12-31", freq="ME")
    values = pd.DataFrame(index=dates)
    for name in ml_experiment.CANDIDATES:
        values[name] = np.tile([0.01, -0.01], 18)
    values.loc[:"2022-12-31", "depth2_l2_1"] = np.tile([0.03, -0.01], 12)
    # A spectacular evaluation winner must never affect validation selection.
    values.loc["2023-01-01":, "depth3_l2_10"] = np.tile([0.90, 0.10], 6)
    values["benchmark"] = np.tile([0.005, -0.005], 18)
    return values


def test_selection_ignores_evaluation_returns_and_maximizes_net_validation_sharpe():
    returns = _validation_returns()
    assert ml_experiment.select_candidate(returns) == "depth2_l2_1"
    returns.loc["2023-01-01":, "baseline"] = -0.99
    returns.loc["2023-01-01":, "depth2_l2_1"] = -0.95
    assert ml_experiment.select_candidate(returns) == "depth2_l2_1"


def test_exact_validation_ties_prefer_baseline():
    returns = _validation_returns()
    for name in ml_experiment.CANDIDATES:
        returns[name] = np.tile([0.02, -0.01], 18)
    assert ml_experiment.select_candidate(returns) == "baseline"


def test_selection_refuses_missing_validation_observations():
    returns = _validation_returns()
    returns.loc["2021-06-30", "baseline"] = np.nan
    with pytest.raises(ValueError, match="complete|missing"):
        ml_experiment.select_candidate(returns)


def test_selection_refuses_incomplete_validation_calendar():
    returns = _validation_returns().drop(pd.Timestamp("2021-06-30"))
    with pytest.raises(ValueError, match="24|month|calendar"):
        ml_experiment.select_candidate(returns)


def test_selection_rejects_all_undefined_sharpe_ratios():
    returns = _validation_returns()
    returns.loc[:, list(ml_experiment.CANDIDATES)] = 0.0
    with pytest.raises(ValueError, match="finite|undefined"):
        ml_experiment.select_candidate(returns)


def test_explicit_zero_regularization_matches_default_predictions():
    dataset = _dataset()
    cutoff = dataset.date.max()
    baseline = generate_walk_forward_scores(dataset, prediction_start=cutoff)
    explicit = generate_walk_forward_scores(
        dataset, prediction_start=cutoff,
        model_parameters={"max_depth": 3, "l2_regularization": 0.0},
    )
    pd.testing.assert_frame_equal(baseline, explicit)


def test_regularization_changes_scores_and_preserves_requested_prediction_dates():
    dataset = _dataset()
    cutoff = sorted(dataset.date.unique())[-2]
    arguments = {"prediction_start": cutoff, "prediction_end": cutoff}
    baseline = generate_walk_forward_scores(dataset, **arguments)
    regularized = generate_walk_forward_scores(
        dataset, **arguments,
        model_parameters={"max_depth": 2, "l2_regularization": 10.0},
    )
    assert regularized.date.unique().tolist() == [cutoff]
    assert set(regularized.ticker) == set("ABCDEF")
    assert regularized.ml_score.between(0, 1).all()
    left = baseline.set_index("ticker").ml_score.sort_index()
    right = regularized.set_index("ticker").ml_score.sort_index()
    assert not np.allclose(left, right)


def test_unavailable_labels_and_future_features_cannot_change_regularized_scores():
    dataset = _dataset()
    cutoff = sorted(dataset.date.unique())[-2]
    arguments = {
        "prediction_start": cutoff, "prediction_end": cutoff,
        "model_parameters": {"max_depth": 2, "l2_regularization": 10.0},
    }
    before = generate_walk_forward_scores(dataset, **arguments)
    poisoned = dataset.copy()
    unavailable = poisoned.target_date.isna() | (poisoned.target_date >= cutoff)
    poisoned.loc[unavailable, "target"] = (
        1 - poisoned.loc[unavailable, "target"].fillna(0)
    )
    poisoned.loc[unavailable, "future_return"] = 1000.0
    poisoned.loc[poisoned.date > cutoff, "momentum_3m"] = 1000.0
    after = generate_walk_forward_scores(poisoned, **arguments)
    pd.testing.assert_series_equal(
        before.set_index("ticker").ml_score.sort_index(),
        after.set_index("ticker").ml_score.sort_index(),
    )


def test_compare_segment_rejects_different_candidate_months():
    dates = pd.date_range("2021-01-31", periods=3, freq="ME")
    backtest = pd.DataFrame({
        "date": dates, "portfolio_return": [0.02, -0.01, 0.03],
        "turnover": [1.0, 0.2, 0.3], "transaction_cost": [0.001, 0.0002, 0.0003],
    })
    benchmark = pd.Series([0.01, -0.02, 0.02], index=dates)
    with pytest.raises(ValueError, match="identical|align|month"):
        ml_experiment.compare_segment(
            {"baseline": backtest, "candidate": backtest.iloc[:-1]}, benchmark,
        )


def test_compare_segment_uses_net_returns_and_initial_capital_drawdown():
    dates = pd.date_range("2021-01-31", periods=3, freq="ME")
    backtest = pd.DataFrame({
        "date": dates, "portfolio_return": [-0.10, 0.10, 0.0],
        "turnover": [1.0, 0.2, 0.3], "transaction_cost": [0.001, 0.0002, 0.0003],
    })
    benchmark = pd.Series([0.01, -0.02, 0.02], index=dates)
    summary, returns = ml_experiment.compare_segment({"baseline": backtest}, benchmark)
    row = summary.set_index("candidate").loc["baseline"]
    assert row.max_drawdown == pytest.approx(-0.10)
    assert row.annual_volatility == pytest.approx(0.10 * np.sqrt(12))
    assert row.mean_monthly_turnover == pytest.approx(0.5)
    assert row.total_transaction_cost == pytest.approx(0.0015)
    assert returns.baseline.tolist() == [-0.10, 0.10, 0.0]


def test_invalid_prediction_range_fails_explicitly():
    with pytest.raises(ValueError, match="prediction_start|range"):
        generate_walk_forward_scores(
            _dataset(), prediction_start="2023-01-01", prediction_end="2022-01-01",
        )


def _experiment_prices():
    rng = np.random.default_rng(33)
    dates = pd.date_range("2015-01-31", "2025-12-31", freq="ME")
    asset_returns = rng.normal(0.008, 0.065, (len(dates), 6))
    prices = pd.DataFrame(
        100 * np.cumprod(1 + asset_returns, axis=0),
        index=dates, columns=list("ABCDEF"),
    )
    benchmark = pd.Series(
        100 * np.cumprod(1 + asset_returns.mean(axis=1)), index=dates,
    )
    return prices, benchmark


def test_runner_freezes_selection_before_evaluation_and_saves_auditable_results(tmp_path):
    prices, benchmark = _experiment_prices()
    output = tmp_path / "run"
    frozen_bytes = []

    def observe_progress(stage, name):
        if stage == "evaluation":
            frozen_bytes.append((output / "selection.json").read_bytes())

    ml_experiment.run_ml_experiment(prices, benchmark, output, observe_progress)
    assert len(frozen_bytes) == 5
    assert len(set(frozen_bytes)) == 1
    assert (output / "selection.json").read_bytes() == frozen_bytes[0]
    selection = json.loads(frozen_bytes[0])
    assert selection["selected_candidate"] in ml_experiment.CANDIDATES
    assert selection["validation_end"] == "2022-12-31"
    assert selection["unseen_holdout"] is False
    validation = pd.read_csv(output / "validation_summary.csv")
    evaluation = pd.read_csv(output / "evaluation_summary.csv")
    assert len(validation) == len(evaluation) == 5
    assert validation.observations.eq(24).all()
    assert evaluation.observations.eq(36).all()
    assert evaluation.is_selected.sum() == 1
    assert evaluation.annual_alpha.notna().all()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["outputs_sha256"]
    for relative, expected_hash in manifest["outputs_sha256"].items():
        assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == expected_hash
    history = pd.read_csv(output / "evaluation_backtests.csv")
    first = history.groupby("candidate").head(1)
    assert first.turnover.eq(1.0).all()
    assert first.transaction_cost.eq(0.001).all()
    assert first.signal_date.str.startswith("2022-12").all()
    assert first.date.str.startswith("2023-01").all()
    assert (output / "comparison.png").is_file()
    assert (output / "report.md").is_file()


def test_runner_does_not_overwrite_existing_run(tmp_path):
    prices, benchmark = _experiment_prices()
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "selection.json"
    marker.write_text("original selection")
    with pytest.raises(FileExistsError):
        ml_experiment.run_ml_experiment(prices, benchmark, output)
    assert marker.read_text() == "original selection"


def test_runner_records_failure_instead_of_silently_shortening_evaluation(tmp_path):
    prices, benchmark = _experiment_prices()
    prices = prices.loc[:"2025-11-30"]
    benchmark = benchmark.loc[:"2025-11-30"]
    output = tmp_path / "incomplete"
    with pytest.raises(ValueError, match="2025|calendar|month"):
        ml_experiment.run_ml_experiment(prices, benchmark, output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert not (output / "selection.json").exists()
