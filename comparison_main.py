"""Run and save a common-period comparison of every implemented strategy."""

from src.config import EXECUTION_TIMING_MODE, ONE_WAY_TURNOVER_COST_BPS
from src.data.loader import load_benchmark, load_prices
from src.evaluation.diagnostics import backtest_diagnostics
from src.evaluation.report import (
    common_period_performance_report,
    common_period_returns,
    common_period_yearly_returns,
    save_performance_report,
)
from src.evaluation.research_pipeline import (
    run_strategy_suite,
    strategy_return_series,
)


from src.evaluation.provenance import record_research_run


@record_research_run
def main() -> None:
    prices = load_prices()
    benchmark = load_benchmark()
    backtests, _, benchmark_returns = run_strategy_suite(prices, benchmark)
    strategies = {
        name: strategy_return_series(result)
        for name, result in backtests.items()
    }

    common = common_period_returns(strategies, benchmark_returns)
    performance = common_period_performance_report(common)
    yearly = common_period_yearly_returns(common)

    print(
        f"\nCommon realization period: {common.index.min().date()} to "
        f"{common.index.max().date()} ({len(common)} months)"
    )
    print(f"Execution timing: {EXECUTION_TIMING_MODE}")
    print(f"One-way turnover cost: {ONE_WAY_TURNOVER_COST_BPS} bps\n")
    print("Common-Period Performance\n")
    print(performance.round(4))
    print("\nCommon-Period Calendar-Year Returns\n")
    print(yearly.round(4))

    print("\nOptimizer Fallback Rates\n")
    for name in ("momentum_max_sharpe", "momentum_min_vol"):
        rate = backtest_diagnostics(backtests[name])[
            "Optimization Fallback Rate"
        ]
        print(f"{name}: {rate:.2%}")

    save_performance_report(performance, "common_period_performance.csv")
    save_performance_report(yearly, "common_period_yearly_returns.csv")


if __name__ == "__main__":
    main()
