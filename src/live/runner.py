"""Shared orchestration for live and historical-replay research modes."""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import pandas as pd

from src.data.loader import load_benchmark, load_prices
from src.live.consensus import build_cross_strategy_consensus
from src.live.data_snapshot import create_data_snapshot
from src.live.live_features import generate_current_features
from src.live.live_ml import generate_live_ml_predictions
from src.live.live_portfolio import construct_current_portfolios
from src.live.live_signals import generate_current_momentum_signals
from src.live.paper_portfolio import update_paper_portfolio
from src.live.paper_state import previous_target_weights
from src.live.research_report import generate_research_report
from src.live.risk_flags import generate_risk_flags
from src.live.run_config import LiveRunConfig
from src.live.run_manifest import build_manifest, write_json


def _historical_context():
    path = Path("reports/robustness_scorecard.csv")
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def run_live_research(config: LiveRunConfig, paper_update=True, paper_dir=Path("paper")):
    prices, benchmark = load_prices(), load_benchmark()
    created = datetime.now(timezone.utc).isoformat()
    snapshot = create_data_snapshot(prices, benchmark, config, created)
    suffix = snapshot.metadata["input_hash"][:8]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + suffix
    run_dir = Path(config.output_directory) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    features = generate_current_features(snapshot)
    signals = generate_current_momentum_signals(snapshot)
    predictions, model_metadata = generate_live_ml_predictions(snapshot, features, config.top_n)
    previous = previous_target_weights(
        paper_dir, snapshot.prices, features.signal_date.max()
    ) if paper_update else {}
    portfolios, summaries = construct_current_portfolios(snapshot.prices, signals, predictions, config.top_n, config.transaction_cost_bps, previous)
    consensus = build_cross_strategy_consensus(portfolios, signals, predictions)
    contribution_path = Path("reports/contribution_concentration_stress.csv")
    contribution = pd.read_csv(contribution_path) if contribution_path.exists() else None
    flags = generate_risk_flags(summaries, snapshot.status, signals, predictions, contribution)
    files = {
        "data_snapshot_metadata.json": snapshot.metadata, "ml_model_metadata.json": model_metadata,
    }
    write_json(run_dir / "data_snapshot_metadata.json", snapshot.metadata)
    write_json(run_dir / "ml_model_metadata.json", model_metadata)
    frames = {"universe_status.csv": snapshot.status, "current_features.csv": features,
              "momentum_signals.csv": signals, "ml_predictions.csv": predictions,
              "target_portfolios.csv": portfolios, "portfolio_summary.csv": summaries,
              "cross_strategy_consensus.csv": consensus, "risk_flags.csv": flags}
    for filename, frame in frames.items(): frame.to_csv(run_dir / filename, index=False)
    outputs = list(files) + list(frames) + ["research_report.md", "research_summary.csv", "manifest.json"]
    manifest = build_manifest(run_id, created, config, snapshot, features.signal_date.max(), model_metadata, outputs)
    write_json(run_dir / "manifest.json", manifest)
    generate_research_report(run_dir, manifest, signals, predictions, portfolios, summaries, consensus, flags, _historical_context())
    Path(config.output_directory).mkdir(parents=True, exist_ok=True)
    write_json(Path(config.output_directory) / "latest.json", {"run_id": run_id, "run_directory": str(run_dir.resolve()), "manifest": str((run_dir / 'manifest.json').resolve())})
    recommendations = evaluated = pd.DataFrame()
    if paper_update:
        recommendations, evaluated = update_paper_portfolio(
            paper_dir, run_id, created, portfolios, summaries,
            snapshot.prices, snapshot.benchmark, previous_weights=previous,
        )
    return {"run_id": run_id, "run_dir": run_dir, "snapshot": snapshot, "features": features,
            "signals": signals, "predictions": predictions, "model_metadata": model_metadata,
            "portfolios": portfolios, "summaries": summaries, "consensus": consensus,
            "flags": flags, "recommendations": recommendations, "evaluated": evaluated, "manifest": manifest}
