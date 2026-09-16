import numpy as np
import pandas as pd


def calculate_rolling_volatility(
    returns: pd.DataFrame,
    window: int = 60,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    """
    Annualized rolling volatility.
    """

    return (
        returns
        .rolling(window)
        .std()
        * np.sqrt(periods_per_year)
    )


def calculate_annualized_volatility(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> pd.Series:
    return (
        returns.std()
        * np.sqrt(periods_per_year)
    )