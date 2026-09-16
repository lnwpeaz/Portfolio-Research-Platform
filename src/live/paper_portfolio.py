"""Orchestration helpers for research-only paper recommendations."""

from pathlib import Path
import pandas as pd

from src.live.paper_evaluation import evaluate_previous_paper_decision
from src.live.paper_state import record_paper_decisions


def update_paper_portfolio(paper_dir: Path, run_id: str, created_at: str,
                           portfolios: pd.DataFrame, summaries: pd.DataFrame,
                           prices: pd.DataFrame, benchmark: pd.Series,
                           previous_weights: dict[str, dict] | None = None):
    paper_dir.mkdir(parents=True, exist_ok=True)
    signal_date = pd.Timestamp(portfolios.signal_date.max())
    evaluated = evaluate_previous_paper_decision(paper_dir, prices, benchmark, signal_date)
    recommendations = record_paper_decisions(
        paper_dir, run_id, created_at, portfolios, summaries, prices,
        previous_weights=previous_weights,
    )
    return recommendations, evaluated
