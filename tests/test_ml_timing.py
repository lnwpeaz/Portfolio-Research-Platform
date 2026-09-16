import numpy as np
import pandas as pd

from src.features.ml_dataset import FEATURE_COLUMNS, build_ml_dataset
from src.models import walk_forward


def _monthly_data(periods=10):
    dates = pd.date_range("2020-01-31", periods=periods, freq="ME")
    prices = pd.DataFrame(
        {"A": 100 * np.cumprod(np.repeat(1.02, periods)),
         "B": 100 * np.cumprod(np.repeat(1.01, periods))},
        index=dates,
    )
    benchmark = pd.Series(
        100 * np.cumprod(np.repeat(1.015, periods)), index=dates
    )
    return prices, benchmark


def test_dataset_retains_terminal_feature_cross_section_without_labels():
    prices, benchmark = _monthly_data()
    dataset = build_ml_dataset(prices, benchmark)
    terminal = dataset[dataset["date"] == dataset["date"].max()]

    assert len(terminal) == prices.shape[1]
    assert terminal["target"].isna().all()
    assert terminal["future_return"].isna().all()
    assert terminal["target_date"].isna().all()


def test_walk_forward_embargoes_labels_ending_on_prediction_date(monkeypatch):
    prices, benchmark = _monthly_data(14)
    dataset = build_ml_dataset(prices, benchmark)
    fit_target_dates = []

    class RecordingRanker:
        def fit(self, data):
            fit_target_dates.append(data["target_date"].max())

        def predict_scores(self, data):
            prediction_date = data["date"].iloc[0]
            assert fit_target_dates[-1] < prediction_date
            return pd.Series(0.5, index=data.index)

    monkeypatch.setattr(walk_forward, "MLStockRanker", RecordingRanker)
    scores = walk_forward.generate_walk_forward_scores(
        dataset, minimum_train_months=4
    )
    assert not scores.empty
    assert set(FEATURE_COLUMNS).isdisjoint(scores.columns)
