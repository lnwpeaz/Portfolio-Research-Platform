import numpy as np
import pandas as pd

from src.evaluation.cross_sectional import (
    assign_score_quantiles, quantile_returns, sector_exposure,
)
from src.evaluation.institutional_pipeline import apply_point_in_time_membership


def _scores():
    return pd.DataFrame({
        "date": [pd.Timestamp("2024-01-31")] * 10,
        "ticker": [f"T{i}" for i in range(10)],
        "ml_score": np.arange(10, dtype=float),
        "future_return": np.arange(10, dtype=float) / 100,
    })


def test_every_scored_security_has_one_ordered_quantile():
    assigned = assign_score_quantiles(_scores())
    assert len(assigned) == 10
    assert not assigned.duplicated(["date", "ticker"]).any()
    assert set(assigned["quantile"]) == {"Q1", "Q2", "Q3", "Q4", "Q5"}
    assert assigned[assigned["quantile"].eq("Q1")].ml_score.max() < assigned[assigned["quantile"].eq("Q5")].ml_score.min()


def test_quantile_assignment_does_not_use_future_returns():
    original = assign_score_quantiles(_scores())[["ticker", "quantile"]]
    changed = _scores()
    changed["future_return"] = changed.future_return.iloc[::-1].to_numpy()
    revised = assign_score_quantiles(changed)[["ticker", "quantile"]]
    pd.testing.assert_frame_equal(original.reset_index(drop=True), revised.reset_index(drop=True))
    returns = quantile_returns(_scores())
    assert returns.loc[0, "Q5"] > returns.loc[0, "Q1"]


def test_sector_weights_reconcile_and_unknown_is_not_dropped():
    membership = pd.DataFrame({
        "ticker": [f"T{i}" for i in range(9)],
        "sector": ["A"] * 5 + ["B"] * 4,
    })
    exposure = sector_exposure(_scores(), membership, top_n=4)
    assert "Unknown" in set(exposure.sector)
    assert np.isclose(exposure.groupby("date").universe_weight.sum().iloc[0], 1)
    assert np.isclose(exposure.groupby("date").topn_weight.sum().iloc[0], 1)


def test_pit_filter_uses_signal_date_and_half_open_boundaries():
    dataset = pd.DataFrame({
        "date": pd.to_datetime(["2019-12-31", "2020-01-01", "2020-01-01"]),
        "ticker": ["OLD", "OLD", "NEW"], "target": [1, 1, 1],
    })
    membership = pd.DataFrame({
        "ticker": ["OLD", "NEW"], "effective_from": pd.to_datetime(["2019-01-01", "2020-01-01"]),
        "effective_to": pd.to_datetime(["2020-01-01", None]),
    })
    result = apply_point_in_time_membership(dataset, membership)
    assert list(zip(result.date.dt.strftime("%Y-%m-%d"), result.ticker)) == [
        ("2019-12-31", "OLD"), ("2020-01-01", "NEW")
    ]
