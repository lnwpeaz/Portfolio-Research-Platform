"""Persistent immutable paper-decision state. No brokerage interaction."""

import json
from pathlib import Path
import pandas as pd


TRANSACTION_COLUMNS = [
    "paper_run_id", "strategy", "signal_date", "ticker",
    "pretrade_weight", "target_weight", "trade_weight",
    "estimated_transaction_cost",
]


def normalize_transaction_history(path: Path) -> None:
    """Migrate the earlier target-only paper ledger before future appends."""
    if not path.exists():
        return
    existing = pd.read_csv(path)
    if "pretrade_weight" not in existing:
        existing["pretrade_weight"] = 0.0
    if "trade_weight" not in existing:
        existing["trade_weight"] = existing["target_weight"]
    existing.reindex(columns=TRANSACTION_COLUMNS).to_csv(path, index=False)


def load_paper_state(paper_dir: Path) -> dict:
    path = paper_dir / "state.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_previous_recommendations(paper_dir: Path) -> pd.DataFrame:
    state = load_paper_state(paper_dir)
    path = state.get("latest_recommendations")
    return pd.read_csv(path, parse_dates=["signal_date"]) if path and Path(path).exists() else pd.DataFrame()


def previous_target_weights(
    paper_dir: Path, prices: pd.DataFrame | None = None,
    current_date: pd.Timestamp | None = None,
) -> dict[str, dict]:
    """Return prior targets or their as-of-safe drifted pre-trade weights."""
    previous = load_previous_recommendations(paper_dir)
    if previous.empty:
        return {}
    result = {}
    for strategy, group in previous.groupby("strategy"):
        weights = group.set_index("ticker").target_weight.astype(float)
        if prices is not None and current_date is not None:
            signal_date = pd.Timestamp(group.signal_date.max())
            current_date = pd.Timestamp(current_date)
            if current_date > signal_date:
                relatives = prices.loc[current_date, weights.index] / prices.loc[signal_date, weights.index]
                values = weights * relatives
                weights = values / values.sum()
        result[strategy] = weights.to_dict()
    return result


def record_paper_decisions(
    paper_dir: Path, run_id: str, created_at: str, portfolios: pd.DataFrame,
    summaries: pd.DataFrame, prices: pd.DataFrame,
    previous_weights: dict[str, dict] | None = None,
) -> pd.DataFrame:
    run_dir = paper_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    signal_date = pd.Timestamp(portfolios.signal_date.max())
    costs = summaries.set_index("strategy").estimated_transaction_cost
    previous_weights = previous_weights or {}
    transaction_rows = []
    for strategy, group in portfolios.groupby("strategy", sort=False):
        target = group.set_index("ticker").target_weight.astype(float)
        pretrade = pd.Series(previous_weights.get(strategy, {}), dtype=float)
        tickers = target.index.union(pretrade.index)
        target = target.reindex(tickers, fill_value=0.0)
        pretrade = pretrade.reindex(tickers, fill_value=0.0)
        trades = target - pretrade
        total_abs_trade = float(trades.abs().sum())
        strategy_cost = float(costs[strategy])
        for ticker in tickers:
            allocated_cost = (
                strategy_cost * abs(trades[ticker]) / total_abs_trade
                if total_abs_trade > 0 else 0.0
            )
            transaction_rows.append({
                "paper_run_id": run_id, "strategy": strategy,
                "signal_date": signal_date, "ticker": ticker,
                "pretrade_weight": pretrade[ticker],
                "target_weight": target[ticker],
                "trade_weight": trades[ticker],
                "estimated_transaction_cost": allocated_cost,
            })
    transactions = pd.DataFrame(transaction_rows)
    allocated_costs = transactions.set_index(["strategy", "ticker"])[
        "estimated_transaction_cost"
    ]

    rows = []
    for row in portfolios.itertuples(index=False):
        rows.append({
            "paper_run_id": run_id, "strategy": row.strategy, "signal_date": signal_date,
            "decision_created_at": created_at, "ticker": row.ticker,
            "target_weight": row.target_weight,
            "reference_price": prices.loc[signal_date, row.ticker],
            "estimated_cost": allocated_costs.get((row.strategy, row.ticker), 0.0),
            "status": "research_reference_price_not_execution_fill",
        })
    recommendations = pd.DataFrame(rows)
    recommendation_path = run_dir / "recommendations.csv"
    recommendations.to_csv(recommendation_path, index=False)
    holdings_path = paper_dir / "holdings.csv"
    recommendations.to_csv(holdings_path, mode="a", header=not holdings_path.exists(), index=False)
    transaction_path = paper_dir / "transactions.csv"
    normalize_transaction_history(transaction_path)
    transactions = transactions.reindex(columns=TRANSACTION_COLUMNS)
    transactions.to_csv(transaction_path, mode="a", header=not transaction_path.exists(), index=False)
    state = {"latest_run_id": run_id, "latest_signal_date": signal_date.isoformat(), "latest_recommendations": str(recommendation_path.resolve())}
    (paper_dir / "state.json").write_text(json.dumps(state, indent=2))
    return recommendations
