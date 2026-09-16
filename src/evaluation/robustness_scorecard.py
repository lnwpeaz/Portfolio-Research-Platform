"""Human-readable Phase 1/2 robustness scorecard; no synthetic rating."""

import numpy as np
import pandas as pd


def build_robustness_scorecard(
    baseline: pd.DataFrame, loo: pd.DataFrame, subperiod: pd.DataFrame,
    costs: pd.DataFrame, execution: pd.DataFrame, bootstrap: pd.DataFrame,
    contribution: pd.DataFrame, outliers: pd.DataFrame, rolling: pd.DataFrame,
    optimizer: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for strategy in baseline.index:
        def one(frame, condition):
            match = frame.loc[condition]
            return match.iloc[0] if len(match) else None
        l = one(loo, loo.strategy.eq(strategy)); s = one(subperiod, subperiod.strategy.eq(strategy))
        c = one(costs, costs.strategy.eq(strategy) & costs.cost_bps.eq(100))
        e = one(execution, execution.strategy.eq(strategy) & execution.execution_scenario.eq("one_period_delay"))
        b = one(bootstrap, bootstrap.strategy.eq(strategy)); cn = one(contribution, contribution.strategy.eq(strategy))
        o = one(outliers, outliers.strategy.eq(strategy)); r = one(rolling, rolling.strategy.eq(strategy))
        op = one(optimizer, optimizer.strategy.eq(strategy))
        rows.append({
            "strategy": strategy, "baseline_sharpe": baseline.loc[strategy, "sharpe"],
            "baseline_information_ratio": baseline.loc[strategy, "information_ratio"],
            "worst_leave_one_out_sharpe": l.minimum_leave_one_out_sharpe if l is not None else np.nan,
            "leave_one_out_sharpe_std": l.std_leave_one_out_sharpe if l is not None else np.nan,
            "positive_active_subperiod_ratio": s.positive_active_return_ratio if s is not None else np.nan,
            "sharpe_100bps": c.sharpe if c is not None else np.nan,
            "delayed_execution_sharpe": e.sharpe if e is not None else np.nan,
            "delayed_execution_sharpe_delta": e.sharpe_delta if e is not None else np.nan,
            "bootstrap_p_active_return_positive": b.p_active_return_positive if b is not None else np.nan,
            "bootstrap_p_sharpe_above_benchmark": b.p_sharpe_above_benchmark if b is not None else np.nan,
            "bootstrap_sharpe_p5": b.sharpe_p5 if b is not None else np.nan,
            "bootstrap_sharpe_median": b.median_sharpe if b is not None else np.nan,
            "top1_contribution_share": cn.top_1_share if cn is not None else np.nan,
            "top3_contribution_share": cn.top_3_share if cn is not None else np.nan,
            "best3_month_arithmetic_share": o.best_3_month_arithmetic_share if o is not None else np.nan,
            "rolling_24m_positive_active_ratio": r.positive_active_return_ratio if r is not None else np.nan,
            "rolling_24m_sharpe_above_benchmark_ratio": r.sharpe_above_benchmark_ratio if r is not None else np.nan,
            "average_effective_holdings": op.average_effective_holdings if op is not None else np.nan,
            "average_max_weight": op.average_max_weight if op is not None else np.nan,
        })
    return pd.DataFrame(rows)
