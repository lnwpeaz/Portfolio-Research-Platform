"""Run Phase 1 robustness and bias-validation experiments."""

import numpy as np

from src.data.loader import load_benchmark, load_prices
from src.evaluation.cost_sensitivity import transaction_cost_sensitivity
from src.evaluation.execution_sensitivity import execution_timing_sensitivity
from src.evaluation.report import (
    REPORT_DIR,
    common_period_returns,
    save_performance_report,
)
from src.evaluation.research_pipeline import (
    run_strategy_suite,
    strategy_return_series,
)
from src.evaluation.robustness_common import evaluate_return_series
from src.evaluation.subperiod_analysis import (
    subperiod_performance,
    subperiod_stability_summary,
)
from src.evaluation.universe_robustness import leave_one_out_analysis
from src.evaluation.visualization import (
    plot_cost_sensitivity_sharpe,
    plot_execution_sensitivity_sharpe,
    plot_leave_one_out_sharpe,
    plot_subperiod_sharpe,
)


def _print_leave_one_out_summary(summary) -> None:
    print("Leave-One-Out Universe Robustness")
    print("---------------------------------")
    for row in summary.itertuples(index=False):
        print(f"{row.strategy}:")
        print(f"  Baseline Sharpe: {row.baseline_sharpe:.4f}")
        print(
            f"  Minimum LOO Sharpe: {row.minimum_leave_one_out_sharpe:.4f} "
            f"({row.worst_removed_ticker_by_sharpe} removed)"
        )
        print(
            f"  Maximum LOO Sharpe: {row.maximum_leave_one_out_sharpe:.4f} "
            f"({row.best_removed_ticker_by_sharpe} removed)"
        )


from src.evaluation.provenance import record_research_run


@record_research_run
def main() -> None:
    print("=" * 60)
    print("ROBUSTNESS & BIAS VALIDATION — PHASE 1")
    print("=" * 60)

    prices = load_prices()
    benchmark_prices = load_benchmark()
    backtests, _, benchmark_returns = run_strategy_suite(prices, benchmark_prices)
    strategies = {
        name: strategy_return_series(backtest)
        for name, backtest in backtests.items()
    }
    common = common_period_returns(strategies, benchmark_returns)
    print("Baseline common period:")
    print(f"{common.index.min().date()} to {common.index.max().date()}")
    print(f"Observations: {len(common)}\n")

    ml_baseline = evaluate_return_series(common["ml"], common["benchmark"])
    expected_ml = {
        "annual_return": 0.3083,
        "annual_volatility": 0.2299,
        "sharpe": 1.2909,
        "max_drawdown": -0.1986,
        "information_ratio": 1.2458,
    }
    for metric, expected in expected_ml.items():
        if not np.isclose(ml_baseline[metric], expected, atol=5e-4):
            raise RuntimeError(
                f"Official ML baseline regression for {metric}: "
                f"{ml_baseline[metric]:.6f} vs {expected:.6f}."
            )

    loo_details, loo_summary = leave_one_out_analysis(
        prices=prices,
        benchmark_returns=benchmark_returns,
        benchmark_prices=benchmark_prices,
        baseline_backtests=backtests,
        evaluation_dates=common.index,
    )
    save_performance_report(
        loo_details, "leave_one_out_robustness.csv", index=False
    )
    save_performance_report(
        loo_summary, "leave_one_out_summary.csv", index=False
    )
    _print_leave_one_out_summary(loo_summary)

    subperiod = subperiod_performance(common)
    stability = subperiod_stability_summary(subperiod)
    save_performance_report(subperiod, "subperiod_performance.csv", index=False)
    save_performance_report(
        stability, "subperiod_stability_summary.csv", index=False
    )
    print("\nSubperiod Stability")
    print("-------------------")
    for row in stability.itertuples(index=False):
        print(
            f"{row.strategy}: mean Sharpe {row.mean_subperiod_sharpe:.4f}, "
            f"positive active-return periods "
            f"{row.positive_active_return_subperiods}/{row.total_subperiods}"
        )

    cost_details, cost_breakpoints = transaction_cost_sensitivity(
        backtests, benchmark_returns, common.index
    )
    save_performance_report(
        cost_details, "transaction_cost_sensitivity.csv", index=False
    )
    save_performance_report(
        cost_breakpoints, "transaction_cost_breakpoints.csv", index=False
    )
    print("\nTransaction Cost Sensitivity")
    print("----------------------------")
    for strategy, group in cost_details.groupby("strategy", sort=False):
        zero = group.loc[group["cost_bps"].eq(0), "sharpe"].iloc[0]
        hundred = group.loc[group["cost_bps"].eq(100), "sharpe"].iloc[0]
        print(f"{strategy}: Sharpe {zero:.4f} at 0 bps, {hundred:.4f} at 100 bps")

    execution = execution_timing_sensitivity(
        backtests, prices, benchmark_returns, common.index
    )
    save_performance_report(
        execution, "execution_sensitivity.csv", index=False
    )
    print("\nExecution Timing Sensitivity")
    print("----------------------------")
    for strategy, group in execution.groupby("strategy", sort=False):
        same = group[group["execution_scenario"].eq("same_close")].iloc[0]
        delayed = group[group["execution_scenario"].eq("one_period_delay")].iloc[0]
        print(
            f"{strategy}: Sharpe {same['sharpe']:.4f} same-close vs "
            f"{delayed['sharpe']:.4f} delayed "
            f"(delta {delayed['sharpe_delta']:+.4f})"
        )

    plot_leave_one_out_sharpe(
        loo_details, REPORT_DIR / "leave_one_out_sharpe.png", show=False
    )
    plot_subperiod_sharpe(
        subperiod, REPORT_DIR / "subperiod_sharpe.png", show=False
    )
    plot_cost_sensitivity_sharpe(
        cost_details, REPORT_DIR / "cost_sensitivity_sharpe.png", show=False
    )
    plot_execution_sensitivity_sharpe(
        execution, REPORT_DIR / "execution_sensitivity_sharpe.png", show=False
    )

    print("\nOutputs saved under reports/")
    print("=" * 60)


if __name__ == "__main__":
    main()
