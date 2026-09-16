"""Prospective scoring with the unchanged model and strict label embargo."""

import pandas as pd

from src.config import ML_MINIMUM_TRAIN_MONTHS
from src.features.ml_dataset import FEATURE_COLUMNS, build_ml_dataset
from src.live.data_snapshot import DataSnapshot
from src.models.ml_ranker import MLStockRanker


def assert_prediction_cutoff(training: pd.DataFrame, prediction_date) -> None:
    if training.empty or training.target_date.max() >= pd.Timestamp(prediction_date):
        raise ValueError("Live training includes unavailable or future label.")


def generate_live_ml_predictions(
    snapshot: DataSnapshot, current_features: pd.DataFrame, top_n: int,
) -> tuple[pd.DataFrame, dict]:
    prediction_date = pd.Timestamp(current_features.signal_date.max())
    eligible = set(snapshot.metadata["eligible_universe"])
    dataset = build_ml_dataset(snapshot.prices, snapshot.benchmark)
    training = dataset[
        dataset.ticker.isin(eligible) & dataset.target_date.notna()
        & (dataset.target_date < prediction_date) & dataset.target.notna()
    ].copy()
    assert_prediction_cutoff(training, prediction_date)
    if training.date.nunique() < ML_MINIMUM_TRAIN_MONTHS or training.target.nunique() < 2:
        raise ValueError("Insufficient valid live ML training history.")
    scoring = current_features[
        current_features.feature_valid & current_features.ticker.isin(eligible)
    ][["ticker", *FEATURE_COLUMNS]].sort_values("ticker").reset_index(drop=True)
    if scoring.empty:
        raise ValueError("No eligible complete live feature rows.")
    model = MLStockRanker()
    model.fit(training)
    scoring["ml_score"] = model.predict_scores(scoring)
    scoring = scoring.sort_values(["ml_score", "ticker"], ascending=[False, True]).reset_index(drop=True)
    scoring["ml_rank"] = range(1, len(scoring) + 1)
    scoring["selected_top_n"] = scoring.ml_rank <= top_n
    scoring["prediction_date"] = prediction_date
    scoring["training_start_date"] = training.date.min()
    scoring["latest_training_label_date"] = training.target_date.max()
    scoring["training_rows"] = len(training)
    scoring["model_status"] = "fitted"
    output = scoring[["ticker", "prediction_date", "ml_score", "ml_rank", "selected_top_n", "training_start_date", "latest_training_label_date", "training_rows", "model_status"]]
    params = model.model.get_params()
    metadata = {
        "model_class": type(model.model).__name__,
        "hyperparameters": {k: params[k] for k in ("max_iter", "learning_rate", "max_depth", "random_state")},
        "feature_columns": FEATURE_COLUMNS, "training_rows": len(training),
        "training_start": training.date.min().isoformat(), "training_end": training.date.max().isoformat(),
        "latest_training_label_date": training.target_date.max().isoformat(),
        "prediction_date": prediction_date.isoformat(), "minimum_train_months": ML_MINIMUM_TRAIN_MONTHS,
    }
    return output, metadata
