"""Transaction-cost sensitivity using official turnover mechanics."""

import pandas as pd

from src.backtest.costs import calculate_one_way_transaction_cost
from src.evaluation.robustness_common import evaluate_return_series
from src.evaluation.research_pipeline import strategy_return_series


TESTED_COST_BPS = (0, 5, 10, 25, 50, 100)


def apply_cost_scenario(
    backtest: pd.DataFrame,
    cost_bps: float,
) -> pd.DataFrame:
    """Reapply per-period costs to fixed gross returns and fixed turnover."""
    scenario = backtest.copy(deep=True)
    scenario["transaction_cost"] = scenario["turnover"].map(
        lambda turnover: calculate_one_way_transaction_cost(
            turnover,
            one_way_turnover_cost_bps=cost_bps,
        )
    )
    scenario["portfolio_return"] = (
        scenario["gross_return"] - scenario["transaction_cost"]
    )
    scenario["cost_bps"] = cost_bps
    return scenario


def transaction_cost_sensitivity(
    backtests: dict[str, pd.DataFrame],
    benchmark_returns: pd.Series,
    evaluation_dates: pd.Index,
    cost_levels: tuple[int, ...] = TESTED_COST_BPS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate fixed strategy paths under independently tested cost levels."""
    rows = []
    benchmark = benchmark_returns.reindex(evaluation_dates).dropna()
    benchmark_metrics = evaluate_return_series(benchmark, benchmark)
    for strategy, backtest in backtests.items():
        for cost_bps in cost_levels:
            scenario = apply_cost_scenario(backtest, cost_bps)
            scenario = scenario[scenario["date"].isin(evaluation_dates)]
            metrics = evaluate_return_series(
                strategy_return_series(scenario), benchmark
            )
            rows.append(
                {
                    "strategy": strategy,
                    "cost_bps": cost_bps,
                    **metrics,
                    "average_turnover": scenario["turnover"].mean(),
                    "total_transaction_cost": scenario[
                        "transaction_cost"
                    ].sum(),
                    "benchmark_sharpe": benchmark_metrics["sharpe"],
                }
            )
    details = pd.DataFrame(rows)

    breakpoints = []
    for strategy, group in details.groupby("strategy", sort=False):
        group = group.sort_values("cost_bps")
        active_break = group[group["annual_active_return"] <= 0]
        sharpe_break = group[group["sharpe"] <= group["benchmark_sharpe"]]
        breakpoints.append(
            {
                "strategy": strategy,
                "first_cost_bps_active_return_nonpositive": (
                    int(active_break["cost_bps"].iloc[0])
                    if not active_break.empty
                    else "not_reached"
                ),
                "first_cost_bps_sharpe_at_or_below_benchmark": (
                    int(sharpe_break["cost_bps"].iloc[0])
                    if not sharpe_break.empty
                    else "not_reached"
                ),
            }
        )
    return details, pd.DataFrame(breakpoints)
