import pandas as pd

from src.features.ml_dataset import FEATURE_COLUMNS, build_ml_dataset
from src.live.data_snapshot import DataSnapshot, assert_no_future_market_data


FORBIDDEN_LIVE_COLUMNS = {"target", "target_date", "future_return", "benchmark_return"}


def assert_no_future_features(features: pd.DataFrame, as_of_date):
    if not features.empty and pd.to_datetime(features.signal_date).max() > pd.Timestamp(as_of_date):
        raise ValueError("Future feature date detected.")
    if FORBIDDEN_LIVE_COLUMNS.intersection(features.columns):
        raise ValueError("Evaluation-only columns detected in live features.")


def generate_current_features(snapshot: DataSnapshot) -> pd.DataFrame:
    cutoff = pd.Timestamp(snapshot.metadata["as_of_date"])
    assert_no_future_market_data(snapshot.prices, snapshot.benchmark, cutoff)
    dataset = build_ml_dataset(snapshot.prices, snapshot.benchmark)
    signal_date = dataset.date.max()
    current = dataset[dataset.date.eq(signal_date)].set_index("ticker")
    eligibility = snapshot.status.set_index("ticker")
    rows = []
    for ticker in snapshot.metadata["requested_universe"]:
        eligible = ticker in eligibility.index and bool(eligibility.loc[ticker, "eligible"])
        valid = eligible and ticker in current.index and current.loc[ticker, FEATURE_COLUMNS].notna().all()
        row = {"ticker": ticker, "signal_date": signal_date}
        for feature in FEATURE_COLUMNS:
            row[feature] = current.loc[ticker, feature] if ticker in current.index else pd.NA
        reason = None if valid else (eligibility.loc[ticker, "reason"] if ticker in eligibility.index and pd.notna(eligibility.loc[ticker, "reason"]) else "insufficient_or_missing_feature_history")
        row.update({"feature_valid": bool(valid), "invalid_reason": reason})
        rows.append(row)
    result = pd.DataFrame(rows)
    assert_no_future_features(result, cutoff)
    return result
