import pandas as pd

from pypfopt import EfficientFrontier
from pypfopt import expected_returns
from pypfopt import risk_models

from src.portfolio.constraints import cap_weights, validate_weights


def equal_weight_weights(
    tickers: list[str],
) -> pd.Series:
    if not tickers:
        raise ValueError("Ticker list is empty.")

    return pd.Series(
        1 / len(tickers),
        index=tickers,
        name="weight",
    )


def max_sharpe_weights(
    prices: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> pd.Series:
    prices = prices.dropna(axis=1, how="any")

    if prices.shape[1] < 2:
        return equal_weight_weights(
            prices.columns.tolist()
        )

    mu = expected_returns.mean_historical_return(
        prices,
        frequency=252,
    )

    covariance = risk_models.sample_cov(
        prices,
        frequency=252,
    )

    ef = EfficientFrontier(
        mu,
        covariance,
        weight_bounds=(0, max_weight),
    )

    ef.max_sharpe(
        risk_free_rate=risk_free_rate
    )

    weights = pd.Series(
        ef.clean_weights(),
        dtype=float,
    )

    weights = cap_weights(weights, max_weight=max_weight)
    validate_weights(weights, max_weight=max_weight)
    return weights


def min_volatility_weights(
    prices: pd.DataFrame,
    max_weight: float = 1.0,
) -> pd.Series:
    prices = prices.dropna(axis=1, how="any")

    if prices.shape[1] < 2:
        return equal_weight_weights(
            prices.columns.tolist()
        )

    covariance = risk_models.sample_cov(
        prices,
        frequency=252,
    )

    ef = EfficientFrontier(
        expected_returns=None,
        cov_matrix=covariance,
        weight_bounds=(0, max_weight),
    )

    ef.min_volatility()

    weights = pd.Series(
        ef.clean_weights(),
        dtype=float,
    )

    weights = cap_weights(weights, max_weight=max_weight)
    validate_weights(weights, max_weight=max_weight)
    return weights
