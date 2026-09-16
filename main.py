import pandas as pd

from src.data.loader import (
    load_prices,
    load_benchmark,
)
from src.data.frequency import last_observation_by_month

from src.backtest.engine import (
    run_momentum_backtest,
)

from src.evaluation.metrics import (
    performance_summary,
)

from src.evaluation.visualization import (
    plot_strategy_comparison,
)
from src.config import (
    EXECUTION_TIMING_MODE,
    MAX_POSITION_WEIGHT,
    MOMENTUM_LOOKBACK_MONTHS,
    ONE_WAY_TURNOVER_COST_BPS,
    OPTIMIZATION_LOOKBACK_DAYS,
    RISK_FREE_RATE,
    TOP_N,
)


def prepare_returns(
    backtest: pd.DataFrame,
) -> pd.Series:

    return (
        backtest
        .set_index("date")["portfolio_return"]
    )


from src.evaluation.provenance import record_research_run


@record_research_run
def main():

    prices = load_prices()
    benchmark = load_benchmark()

    print(
        f"Stocks loaded: {prices.shape[1]}"
    )

    strategies = {}

    for method in [
        "equal",
        "max_sharpe",
        "min_vol",
    ]:

        print(
            f"\nRunning {method}..."
        )

        result = run_momentum_backtest(
            prices=prices,
            lookback_months=MOMENTUM_LOOKBACK_MONTHS,
            top_n=TOP_N,
            weighting=method,
            optimization_lookback_days=OPTIMIZATION_LOOKBACK_DAYS,
            one_way_turnover_cost_bps=ONE_WAY_TURNOVER_COST_BPS,
            max_position_weight=MAX_POSITION_WEIGHT,
            risk_free_rate=RISK_FREE_RATE,
            execution_timing=EXECUTION_TIMING_MODE,
        )

        strategies[method] = prepare_returns(
            result
        )

    benchmark_monthly = (
        last_observation_by_month(benchmark)
        .pct_change(fill_method=None)
    )

    comparison = {}

    for name, returns in strategies.items():

        comparison[name] = (
            performance_summary(
                returns
            )
        )

    comparison["S&P 500"] = (
        performance_summary(
            benchmark_monthly
            .reindex(
                strategies["equal"].index
            )
            .dropna()
        )
    )

    comparison = pd.DataFrame(
        comparison
    )

    print(
        "\nPerformance Comparison\n"
    )

    print(
        comparison.round(4)
    )
    plot_strategy_comparison(
        strategies=strategies,
        benchmark_returns=benchmark_monthly.reindex(
            strategies["equal"].index
        ),
    )


if __name__ == "__main__":
    main()
