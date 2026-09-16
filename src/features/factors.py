import pandas as pd

from src.features.momentum import (
    calculate_momentum,
)
from src.features.volatility import (
    calculate_rolling_volatility,
)


def build_factor_table(
    prices: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    returns = prices.pct_change(
        fill_method=None
    )

    factors = {
        "momentum_20d": calculate_momentum(
            prices,
            20,
        ),
        "momentum_60d": calculate_momentum(
            prices,
            60,
        ),
        "momentum_120d": calculate_momentum(
            prices,
            120,
        ),
        "volatility_60d":
            calculate_rolling_volatility(
                returns,
                window=60,
            ),
    }

    return factors


def latest_factor_snapshot(
    factors: dict[str, pd.DataFrame],
) -> pd.DataFrame:

    snapshot = pd.DataFrame({
        factor_name:
        factor_data.iloc[-1]

        for factor_name, factor_data
        in factors.items()
    })

    return snapshot.dropna()