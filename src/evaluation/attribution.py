"""Realized security-level return attribution for portfolio backtests."""

import numpy as np
import pandas as pd


def period_contribution_attribution(
    backtest: pd.DataFrame,
    dates: pd.Index | None = None,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Calculate ``weight * realized return`` and reconcile every period."""
    required = {"date", "weights", "asset_returns", "gross_return"}
    missing = required.difference(backtest.columns)
    if missing:
        raise ValueError(f"Backtest lacks attribution fields: {sorted(missing)}")
    selected = backtest
    if dates is not None:
        selected = selected[selected["date"].isin(dates)]

    rows = []
    for record in selected.itertuples(index=False):
        weights = pd.Series(record.weights, dtype=float)
        asset_returns = pd.Series(record.asset_returns, dtype=float).reindex(
            weights.index
        )
        if asset_returns.isna().any():
            raise ValueError("Attribution asset returns do not match portfolio weights.")
        contributions = weights * asset_returns
        if not np.isclose(contributions.sum(), record.gross_return, atol=tolerance):
            raise ValueError(
                f"Security contributions do not reconcile on {record.date}."
            )
        for ticker in weights.index:
            rows.append(
                {
                    "date": pd.Timestamp(record.date),
                    "ticker": ticker,
                    "weight": weights[ticker],
                    "asset_return": asset_returns[ticker],
                    "contribution": contributions[ticker],
                }
            )
    return pd.DataFrame(rows)


def contribution_by_ticker(
    backtest: pd.DataFrame,
    strategy: str,
    dates: pd.Index | None = None,
) -> pd.DataFrame:
    """Aggregate security contribution and keep transaction costs separate."""
    period = period_contribution_attribution(backtest, dates=dates)
    securities = (
        period.groupby("ticker", as_index=False)["contribution"].sum()
        if not period.empty
        else pd.DataFrame(columns=["ticker", "contribution"])
    )
    securities.insert(0, "strategy", strategy)
    securities["attribution_type"] = "security"

    selected = backtest if dates is None else backtest[backtest["date"].isin(dates)]
    costs = pd.DataFrame(
        {
            "strategy": [strategy],
            "ticker": ["TRANSACTION_COST"],
            "contribution": [-float(selected["transaction_cost"].sum())],
            "attribution_type": ["transaction_cost"],
        }
    )
    return pd.concat([securities, costs], ignore_index=True)
