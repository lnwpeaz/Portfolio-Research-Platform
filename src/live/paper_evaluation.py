"""Prospective evaluation using frozen historical paper weights."""

from pathlib import Path
import pandas as pd

from src.live.paper_state import load_previous_recommendations


def evaluate_previous_paper_decision(
    paper_dir: Path, prices: pd.DataFrame, benchmark: pd.Series,
    realization_date,
) -> pd.DataFrame:
    previous = load_previous_recommendations(paper_dir)
    if previous.empty:
        return pd.DataFrame()
    realization_date = pd.Timestamp(realization_date)
    signal_date = pd.Timestamp(previous.signal_date.max())
    if realization_date <= signal_date:
        return pd.DataFrame()
    performance_path = paper_dir / "performance.csv"
    existing = pd.read_csv(performance_path, parse_dates=["signal_date", "realization_date"]) if performance_path.exists() else pd.DataFrame()
    rows = []
    benchmark_return = float(benchmark.loc[realization_date] / benchmark.loc[signal_date] - 1)
    for strategy, group in previous.groupby("strategy"):
        if not existing.empty and ((existing.strategy == strategy) & (existing.signal_date == signal_date) & (existing.realization_date == realization_date)).any():
            continue
        frozen = group.set_index("ticker").target_weight
        security_returns = prices.loc[realization_date, frozen.index] / prices.loc[signal_date, frozen.index] - 1
        gross = float(frozen @ security_returns)
        cost = float(group.estimated_cost.sum())
        net = gross - cost
        prior = existing[existing.strategy.eq(strategy)].sort_values("realization_date") if not existing.empty else pd.DataFrame()
        portfolio_value = (prior.portfolio_value.iloc[-1] if len(prior) else 1.0) * (1 + net)
        benchmark_value = (prior.benchmark_value.iloc[-1] if len(prior) else 1.0) * (1 + benchmark_return)
        rows.append({"strategy": strategy, "signal_date": signal_date, "realization_date": realization_date, "gross_return": gross, "transaction_cost": cost, "net_return": net, "benchmark_return": benchmark_return, "active_return": net - benchmark_return, "portfolio_value": portfolio_value, "benchmark_value": benchmark_value})
    result = pd.DataFrame(rows)
    if not result.empty:
        result.to_csv(performance_path, mode="a", header=not performance_path.exists(), index=False)
    return result
