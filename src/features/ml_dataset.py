"""Construction of the monthly point-in-time feature and label panel."""

import numpy as np
import pandas as pd

from src.data.frequency import last_observation_by_month


FEATURE_COLUMNS = [
    "momentum_1m",
    "momentum_3m",
    "momentum_6m",
    "volatility_3m",
    "drawdown_3m",
]


def _feature_panels(monthly_prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    return {
        "momentum_1m": monthly_prices.pct_change(1, fill_method=None),
        "momentum_3m": monthly_prices.pct_change(3, fill_method=None),
        "momentum_6m": monthly_prices.pct_change(6, fill_method=None),
        "volatility_3m": monthly_returns.rolling(3, min_periods=3).std() * np.sqrt(12),
        "drawdown_3m": monthly_prices / monthly_prices.rolling(3, min_periods=3).max() - 1,
    }


def build_ml_feature_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Return unchanged ML features, retaining partial rows for coverage audits."""
    if prices.empty or not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices must be a non-empty DatetimeIndex panel.")
    monthly_prices = last_observation_by_month(prices.sort_index().astype(float))
    features = _feature_panels(monthly_prices)
    return (
        pd.concat([_stack_panel(panel, name) for name, panel in features.items()], axis=1)
        .reset_index().sort_values(["date", "ticker"]).reset_index(drop=True)
    )


def _stack_panel(panel: pd.DataFrame, name: str) -> pd.Series:
    return panel.rename_axis(index="date", columns="ticker").stack(
        future_stack=True
    ).rename(name)


def build_ml_dataset(
    prices: pd.DataFrame,
    benchmark: pd.Series,
) -> pd.DataFrame:
    """Build monthly features at *t* and labels realized at *t+1*.

    Rows are retained based on feature availability only. In particular, the
    prediction universe is not filtered using the availability of a future
    stock return. ``target`` is nullable for the terminal/live cross-section
    and for assets whose realized outcome is unavailable.
    """
    if prices.empty or benchmark.empty:
        raise ValueError("Prices and benchmark must be non-empty.")
    if not isinstance(prices.index, pd.DatetimeIndex) or not isinstance(
        benchmark.index, pd.DatetimeIndex
    ):
        raise TypeError("Prices and benchmark must use DatetimeIndex objects.")
    if prices.index.has_duplicates or benchmark.index.has_duplicates:
        raise ValueError("Prices and benchmark dates must be unique.")

    prices = prices.sort_index().astype(float)
    benchmark = benchmark.sort_index().astype(float)
    monthly_prices = last_observation_by_month(prices)
    benchmark_monthly = last_observation_by_month(benchmark)
    if not monthly_prices.index.equals(benchmark_monthly.index):
        raise ValueError(
            "Stock and benchmark month-end observation dates are not aligned."
        )
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    benchmark_returns = benchmark_monthly.pct_change(fill_method=None)

    features = _feature_panels(monthly_prices)

    dataset = pd.concat(
        [_stack_panel(panel, name) for name, panel in features.items()], axis=1
    )
    future_return = _stack_panel(monthly_returns.shift(-1), "future_return")
    dataset = dataset.join(future_return)

    target_dates = pd.Series(
        monthly_prices.index.to_series().shift(-1).to_numpy(),
        index=monthly_prices.index,
    )
    date_values = dataset.index.get_level_values("date")
    dataset["target_date"] = date_values.map(target_dates)
    dataset["benchmark_return"] = date_values.map(benchmark_returns.shift(-1))

    valid_outcome = dataset["future_return"].notna() & dataset[
        "benchmark_return"
    ].notna()
    target = pd.Series(pd.NA, index=dataset.index, dtype="Int8")
    target.loc[valid_outcome] = (
        dataset.loc[valid_outcome, "future_return"]
        > dataset.loc[valid_outcome, "benchmark_return"]
    ).astype("int8")
    dataset["target"] = target

    return (
        dataset.dropna(subset=FEATURE_COLUMNS)
        .reset_index()
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
