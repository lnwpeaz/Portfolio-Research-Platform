import pandas as pd


def calculate_momentum(
    prices: pd.DataFrame,
    window: int = 60,
) -> pd.DataFrame:
    """
    Historical price momentum.

    Example:
    window=60 -> approximately 3 trading months.
    """

    return prices.pct_change(
        periods=window,
        fill_method=None
    )


def calculate_monthly_momentum_signal(
    monthly_prices: pd.DataFrame,
    lookback_months: int,
    skip_recent_months: int = 0,
) -> pd.DataFrame:
    """Calculate a monthly momentum signal with an optional recent-month skip.

    With ``lookback_months=12`` and ``skip_recent_months=1``, the signal at
    month *t* is ``price[t-1] / price[t-12] - 1``. Thus the latest month used
    by the signal ends before the signal date, and no realization-month data
    enter the factor.
    """
    if lookback_months < 1:
        raise ValueError("lookback_months must be positive.")
    if skip_recent_months < 0 or skip_recent_months >= lookback_months:
        raise ValueError(
            "skip_recent_months must be non-negative and less than lookback_months."
        )
    return (
        monthly_prices.shift(skip_recent_months)
        / monthly_prices.shift(lookback_months)
        - 1
    )
