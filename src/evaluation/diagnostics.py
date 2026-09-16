import numpy as np
import pandas as pd


def average_turnover(
    backtest: pd.DataFrame,
) -> float:

    if (
        "turnover"
        not in backtest.columns
    ):
        return np.nan

    return float(
        backtest[
            "turnover"
        ].mean()
    )


def total_transaction_cost(
    backtest: pd.DataFrame,
) -> float:

    if (
        "transaction_cost"
        not in backtest.columns
    ):
        return np.nan

    return float(
        backtest[
            "transaction_cost"
        ].sum()
    )


def average_number_holdings(
    backtest: pd.DataFrame,
) -> float:

    if (
        "holdings"
        not in backtest.columns
    ):
        return np.nan

    return float(
        backtest[
            "holdings"
        ]
        .apply(len)
        .mean()
    )


def gross_net_difference(
    backtest: pd.DataFrame,
) -> float:

    required = {
        "gross_return",
        "portfolio_return",
    }

    if not required.issubset(
        backtest.columns
    ):
        return np.nan

    gross_growth = (
        1
        + backtest[
            "gross_return"
        ]
    ).prod()

    net_growth = (
        1
        + backtest[
            "portfolio_return"
        ]
    ).prod()

    return float(
        gross_growth
        - net_growth
    )


def backtest_diagnostics(
    backtest: pd.DataFrame,
) -> pd.Series:

    return pd.Series({
        "Average Turnover":
            average_turnover(
                backtest
            ),

        "Maximum Turnover": (
            float(backtest["turnover"].max())
            if "turnover" in backtest.columns and not backtest.empty
            else np.nan
        ),

        "Total Transaction Cost":
            total_transaction_cost(
                backtest
            ),

        "Average Holdings":
            average_number_holdings(
                backtest
            ),

        "Gross-Net Growth Difference":
            gross_net_difference(
                backtest
            ),

        "Optimization Fallback Rate": (
            float(
                backtest["optimization_status"]
                .astype(str)
                .str.startswith("fallback")
                .mean()
            )
            if "optimization_status" in backtest.columns and not backtest.empty
            else np.nan
        ),
    })
