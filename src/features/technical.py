# src/features/technical.py

import pandas as pd


def rolling_return(
    prices: pd.DataFrame,
    window: int,
) -> pd.DataFrame:

    return prices.pct_change(
        window,
        fill_method=None,
    )


def moving_average_ratio(
    prices: pd.DataFrame,
    window: int,
) -> pd.DataFrame:

    ma = (
        prices
        .rolling(window)
        .mean()
    )

    return (
        prices / ma
        - 1
    )


def rolling_drawdown(
    prices: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:

    rolling_high = (
        prices
        .rolling(window)
        .max()
    )

    return (
        prices
        / rolling_high
        - 1
    )