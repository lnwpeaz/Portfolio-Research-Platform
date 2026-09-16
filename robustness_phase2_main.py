"""Run Phase 2 statistical robustness and stability diagnostics."""

import numpy as np
import pandas as pd

from src.data.loader import load_benchmark, load_prices
from src.evaluation.bootstrap import bootstrap_block_sensitivity, bootstrap_sampling_stability
from src.evaluation.contribution_stress import contribution_concentration, contribution_counterfactuals
from src.evaluation.ml_stability import ml_ranking_stability, ml_score_dispersion, ml_signal_decay
from src.evaluation.optimizer_diagnostics import optimizer_estimator_diagnostics
from src.evaluation.optimizer_stability import optimizer_weight_stability
from src.evaluation.report import REPORT_DIR, common_period_returns, save_performance_report
from src.evaluation.research_pipeline import run_strategy_suite, strategy_return_series
from src.evaluation.return_outliers import return_outlier_analysis
from src.evaluation.robustness_common import evaluate_return_series
from src.evaluation.robustness_scorecard import build_robustness_scorecard
from src.evaluation.rolling_robustness import rolling_robustness
from src.evaluation.visualization import (
    plot_bootstrap_sharpe_intervals, plot_contribution_concentration,
    plot_ml_rolling_rank_ic, plot_ml_signal_decay, plot_optimizer_effective_holdings,
    plot_optimizer_max_weight, plot_outlier_dependence, plot_rolling_24m_sharpe,
)


def _save(frame: pd.DataFrame, filename: str) -> None:
    save_performance_report(frame, filename, index=False)


def _verify_ml(common: pd.DataFrame) -> None:
    metrics = evaluate_return_series(common.ml, common.benchmark)
    expected = {"annual_return": .3083, "annual_volatility": .2299, "sharpe": 1.2909,
                "max_drawdown": -.1986, "information_ratio": 1.2458}
    for name, value in expected.items():
        if not np.isclose(metrics[name], value, atol=5e-4):
            raise RuntimeError(f"Official ML regression in {name}: {metrics[name]:.6f}")


from src.evaluation.provenance import record_research_run


@record_research_run
def main() -> None:
    print("=" * 60); print("ROBUSTNESS & BIAS VALIDATION — PHASE 2"); print("=" * 60)
    prices, benchmark_prices = load_prices(), load_benchmark()
    backtests, scores, benchmark_returns = run_strategy_suite(prices, benchmark_prices)
    strategies = {name: strategy_return_series(value) for name, value in backtests.items()}
    common = common_period_returns(strategies, benchmark_returns)
    _verify_ml(common)

    bootstrap, intervals = bootstrap_sampling_stability(common)
    block_sensitivity = bootstrap_block_sensitivity(common)
    _save(bootstrap, "bootstrap_summary.csv"); _save(intervals, "bootstrap_metric_intervals.csv")
    _save(block_sensitivity, "bootstrap_block_sensitivity.csv")

    common_backtests = {name: value[value.date.isin(common.index)].copy() for name, value in backtests.items()}
    concentration, security = contribution_concentration(common_backtests, common.index)
    _save(concentration, "contribution_concentration_stress.csv")
    _save(security, "security_contribution_summary.csv")

    reruns, rerun_cache = {}, {}
    for strategy in backtests:
        ranked = security[security.strategy.eq(strategy)].sort_values("rank")
        for n in (1, 2, 3):
            removed = ranked.head(n).ticker.tolist()
            cache_key = tuple(sorted(removed))
            if cache_key not in rerun_cache:
                experimental = prices.drop(columns=removed).copy()
                rerun_cache[cache_key] = run_strategy_suite(
                    experimental, benchmark_prices.copy()
                )[0]
            reruns[(strategy, n)] = rerun_cache[cache_key][strategy]
    counterfactuals = contribution_counterfactuals(
        common_backtests, security, benchmark_returns, common.index, reruns
    )
    _save(counterfactuals, "contribution_counterfactuals.csv")

    outliers, outlier_counterfactuals = return_outlier_analysis(common)
    _save(outliers, "return_outlier_diagnostics.csv")
    _save(outlier_counterfactuals, "return_outlier_counterfactuals.csv")

    opt_ts, opt_summary, opt_tickers = optimizer_weight_stability(backtests)
    estimator, estimator_summary = optimizer_estimator_diagnostics(backtests, prices)
    _save(opt_ts, "optimizer_stability_timeseries.csv"); _save(opt_summary, "optimizer_stability_summary.csv")
    _save(opt_tickers, "optimizer_ticker_stability.csv"); _save(estimator, "optimizer_estimator_diagnostics.csv")
    _save(estimator_summary, "optimizer_estimator_summary.csv")

    yearly, rolling_ic = ml_ranking_stability(scores)
    decay = ml_signal_decay(scores, prices, benchmark_prices)
    score_dispersion, score_summary = ml_score_dispersion(scores)
    _save(yearly, "ml_ranking_yearly_stability.csv"); _save(rolling_ic, "ml_ranking_rolling_stability.csv")
    _save(decay, "ml_signal_decay.csv"); _save(score_dispersion, "ml_score_dispersion.csv")
    _save(score_summary, "ml_score_diagnostic_summary.csv")

    rolling24, rolling_summary = rolling_robustness(common)
    _save(rolling24, "rolling_24m_robustness.csv"); _save(rolling_summary, "rolling_24m_summary.csv")

    baseline = pd.DataFrame({name: evaluate_return_series(common[name], common.benchmark) for name in strategies}).T
    loo = pd.read_csv(REPORT_DIR / "leave_one_out_summary.csv")
    subperiod = pd.read_csv(REPORT_DIR / "subperiod_stability_summary.csv")
    costs = pd.read_csv(REPORT_DIR / "transaction_cost_sensitivity.csv")
    execution = pd.read_csv(REPORT_DIR / "execution_sensitivity.csv")
    scorecard = build_robustness_scorecard(
        baseline, loo, subperiod, costs, execution, bootstrap, concentration,
        outliers, rolling_summary, opt_summary,
    )
    _save(scorecard, "robustness_scorecard.csv")

    plot_bootstrap_sharpe_intervals(intervals, REPORT_DIR / "bootstrap_sharpe_intervals.png", False)
    plot_contribution_concentration(concentration, REPORT_DIR / "contribution_concentration.png", False)
    plot_outlier_dependence(outlier_counterfactuals, REPORT_DIR / "outlier_dependence.png", False)
    plot_optimizer_effective_holdings(opt_ts, REPORT_DIR / "optimizer_effective_holdings.png", False)
    plot_optimizer_max_weight(opt_ts, REPORT_DIR / "optimizer_max_weight.png", False)
    plot_ml_rolling_rank_ic(rolling_ic, REPORT_DIR / "ml_rolling_rank_ic.png", False)
    plot_ml_signal_decay(decay, REPORT_DIR / "ml_signal_decay.png", False)
    plot_rolling_24m_sharpe(rolling24, REPORT_DIR / "rolling_24m_sharpe.png", False)

    print("Bootstrap Sampling Stability\n----------------------------")
    for row in bootstrap.itertuples(index=False):
        print(f"{row.strategy}: observed {row.observed_sharpe:.3f}, median {row.median_sharpe:.3f}, 5%-95% {row.sharpe_p5:.3f} to {row.sharpe_p95:.3f}, P(active>0) {row.p_active_return_positive:.1%}")
    print("\nContribution Concentration\n--------------------------")
    for row in concentration.itertuples(index=False):
        print(f"{row.strategy}: {row.largest_contributor} largest; top-1 {row.top_1_share:.1%}, top-3 {row.top_3_share:.1%}")
    print("\nReturn Outlier Dependence\n-------------------------")
    for strategy in strategies:
        values = outlier_counterfactuals[outlier_counterfactuals.strategy.eq(strategy)].set_index("counterfactual")
        print(f"{strategy}: Sharpe {values.loc['baseline','sharpe']:.3f} -> {values.loc['remove_best_3_months','sharpe']:.3f} without best 3 months")
    print("\nOptimizer Stability\n-------------------")
    for row in opt_summary.itertuples(index=False): print(f"{row.strategy}: effective holdings {row.average_effective_holdings:.2f}, average max weight {row.average_max_weight:.1%}")
    print("\nML Ranking Stability\n--------------------")
    print(yearly[["year", "mean_rank_ic"]].to_string(index=False))
    print("\nSignal Decay\n------------")
    print(decay[["horizon_months", "mean_rank_ic", "average_top_n_spread"]].to_string(index=False))
    print("\nRolling 24M Robustness\n----------------------")
    for row in rolling_summary.itertuples(index=False): print(f"{row.strategy}: positive active {row.positive_active_return_ratio:.1%}, Sharpe > benchmark {row.sharpe_above_benchmark_ratio:.1%}")
    print("\nOutputs saved under reports/"); print("=" * 60)


if __name__ == "__main__":
    main()
