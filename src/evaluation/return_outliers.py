"""Non-investable diagnostics for dependence on extreme realized months."""

import pandas as pd

from src.evaluation.metrics import performance_summary


def return_outlier_analysis(
    common_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostics, counterfactuals = [], []
    for strategy in common_returns:
        returns = common_returns[strategy].dropna().copy()
        total = returns.sum()
        ordered = returns.sort_values()
        diagnostics.append({
            "strategy": strategy,
            "best_month": returns.idxmax(), "best_monthly_return": returns.max(),
            "worst_month": returns.idxmin(), "worst_monthly_return": returns.min(),
            "median_monthly_return": returns.median(),
            "positive_month_fraction": (returns > 0).mean(),
            "negative_month_fraction": (returns < 0).mean(),
            "skewness": returns.skew(), "excess_kurtosis": returns.kurt(),
            **{f"best_{n}_month_arithmetic_share": ordered.tail(n).sum() / total if total != 0 else float("nan") for n in (1, 3, 5)},
            **{f"worst_{n}_month_arithmetic_share": ordered.head(n).sum() / total if total != 0 else float("nan") for n in (1, 3, 5)},
        })
        baseline = performance_summary(returns)
        counterfactuals.append({
            "strategy": strategy, "counterfactual": "baseline", "removed_months": "",
            "observations": len(returns), "annual_return": baseline["Annual Return"],
            "annual_volatility": baseline["Annual Volatility"],
            "sharpe": baseline["Sharpe Ratio"], "max_drawdown": baseline["Max Drawdown"],
        })
        for side in ("best", "worst"):
            for n in (1, 3, 5):
                removed = ordered.tail(n).index if side == "best" else ordered.head(n).index
                remaining = returns.drop(removed)
                metrics = performance_summary(remaining)
                counterfactuals.append({
                    "strategy": strategy, "counterfactual": f"remove_{side}_{n}_months",
                    "removed_months": "|".join(d.strftime("%Y-%m-%d") for d in removed),
                    "observations": len(remaining), "annual_return": metrics["Annual Return"],
                    "annual_volatility": metrics["Annual Volatility"],
                    "sharpe": metrics["Sharpe Ratio"], "max_drawdown": metrics["Max Drawdown"],
                })
    return pd.DataFrame(diagnostics), pd.DataFrame(counterfactuals)
