"""Current large-universe research using the unchanged model and signals."""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import tracemalloc
import uuid

import numpy as np
import pandas as pd

from src.config import ML_MINIMUM_TRAIN_MONTHS
from src.data.frequency import last_observation_by_month
from src.data.institutional.coverage import resolve_common_market_date
from src.data.institutional.validation import validate_market_panel
from src.evaluation.cross_sectional import assign_score_quantiles
from src.evaluation.institutional_pipeline import _current_scores
from src.features.ml_dataset import FEATURE_COLUMNS, build_ml_dataset, build_ml_feature_panel
from src.features.momentum import calculate_monthly_momentum_signal
from src.live.institutional_paper import write_institutional_paper_target
from src.models.ml_ranker import MLStockRanker
from src.universe.investability import InvestabilityConfig, screen_investability
from src.universe.models import UniverseSnapshot


def _run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]


def _market_snapshot(snapshot, resolved):
    return UniverseSnapshot(
        index_name=snapshot.index_name,
        requested_date=snapshot.requested_date,
        resolved_date=resolved,
        members=snapshot.members,
        provider=snapshot.provider,
        point_in_time_verified=False,
        source=snapshot.source,
        warnings=snapshot.warnings,
        retrieved_at=snapshot.retrieved_at,
    )


def _pivot_panel(panel, tickers, resolved):
    subset = panel[
        panel.canonical_ticker.isin(tuple(tickers) + ("^GSPC",))
        & panel.date.le(pd.Timestamp(resolved))
    ]
    wide = subset.pivot(index="date", columns="canonical_ticker", values="adjusted_close").sort_index()
    if "^GSPC" not in wide:
        raise ValueError("Cached institutional panel does not contain ^GSPC.")
    benchmark = wide.pop("^GSPC").rename("benchmark")
    return wide.reindex(columns=[ticker for ticker in tickers if ticker in wide]), benchmark


def _feature_missingness(raw_features, tickers, signal_date):
    current = raw_features[raw_features.date.eq(signal_date)].set_index("ticker")
    rows = []
    for feature in FEATURE_COLUMNS:
        valid = int(current.reindex(tickers)[feature].notna().sum())
        rows.append({
            "feature": feature, "valid_count": valid,
            "missing_count": len(tickers) - valid,
            "coverage_pct": valid / len(tickers) if tickers else np.nan,
        })
    return pd.DataFrame(rows)


def _momentum_rankings(prices, signal_date):
    monthly = last_observation_by_month(prices)
    three = calculate_monthly_momentum_signal(monthly, 3).loc[signal_date].rename("momentum_3m")
    twelve = calculate_monthly_momentum_signal(monthly, 12, skip_recent_months=1).loc[signal_date].rename("momentum_12_1")
    result = pd.concat([three, twelve], axis=1).rename_axis("ticker").reset_index()
    result["rank_3m"] = result.momentum_3m.rank(method="first", ascending=False, na_option="bottom").astype("Int64")
    result["rank_12_1"] = result.momentum_12_1.rank(method="first", ascending=False, na_option="bottom").astype("Int64")
    return result.sort_values(["rank_3m", "ticker"])


def _sector_exposure(rankings, membership, quantiles, top_n):
    eligible = rankings.copy()
    if "sector" not in eligible.columns:
        eligible = eligible.merge(membership[["ticker", "sector"]], on="ticker", how="left")
    eligible["sector"] = eligible.sector.fillna("Unknown")
    q5 = quantiles[quantiles["quantile"].eq("Q5")]
    top = rankings.nsmallest(top_n, "ml_rank")
    sectors = sorted(set(eligible.sector) | set(q5.sector) | set(top.sector))
    rows = []
    for sector in sectors:
        universe_weight = eligible.sector.eq(sector).mean()
        q5_weight = q5.sector.eq(sector).mean() if len(q5) else 0.0
        top_weight = top.sector.eq(sector).mean() if len(top) else 0.0
        rows.append({
            "sector": sector, "eligible_equal_weight": universe_weight,
            "q5_equal_weight": q5_weight, "topn_equal_weight": top_weight,
            "topn_active_equal_weight": top_weight - universe_weight,
            "benchmark_cap_weight": np.nan,
            "benchmark_weight_status": "UNAVAILABLE_NOT_FABRICATED",
        })
    return pd.DataFrame(rows)


def run_institutional_live(
    snapshot, panel, market_data_manifest, *, top_n=25,
    minimum_coverage=0.95, paper_update=False,
    output_root="reports/institutional", paper_root="paper",
):
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    tracemalloc.start()
    total_started = time.perf_counter()
    timings = {}
    stage = time.perf_counter()
    validation_flags = validate_market_panel(panel, snapshot.requested_date)
    if (validation_flags.severity == "error").any():
        raise ValueError("Institutional panel contains validation errors; inspect validation_flags.csv.")
    resolution = resolve_common_market_date(
        panel, snapshot.tickers, snapshot.requested_date, minimum_coverage
    )
    market_snapshot = _market_snapshot(snapshot, resolution["resolved_market_date"])
    prices, benchmark = _pivot_panel(panel, snapshot.tickers, market_snapshot.resolved_date)
    timings["data_load_validation_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    raw_features = build_ml_feature_panel(prices)
    dataset = build_ml_dataset(prices, benchmark)
    signal_date = dataset.date.max()
    current_features = dataset[dataset.date.eq(signal_date)]
    screening = screen_investability(
        market_snapshot, prices, current_features,
        InvestabilityConfig(
            minimum_history_observations=252, stale_calendar_days=3,
            required_features=tuple(FEATURE_COLUMNS),
        ),
    )
    eligible = screening.eligible_tickers
    timings["feature_generation_seconds"] = time.perf_counter() - stage
    if not eligible:
        raise ValueError("No S&P 500 securities are feature eligible.")

    stage = time.perf_counter()
    scored, training_rows, model_timings = _current_scores(
        dataset, eligible, signal_date, return_timings=True
    )
    timings.update(model_timings)
    timings["ml_training_scoring_seconds"] = time.perf_counter() - stage
    scored = scored.rename(columns={"rank": "ml_rank"})
    membership = market_snapshot.to_frame().rename(columns={"company_name": "company"})
    rankings = scored[["date", "ticker", "ml_score", "ml_rank"]].merge(
        membership[["ticker", "company", "sector", "industry"]], on="ticker", how="left"
    )
    rankings["feature_valid"] = True
    latest_target = dataset[
        dataset.target_date.notna() & dataset.target_date.lt(signal_date) & dataset.target.notna()
    ].target_date.max()
    rankings["prediction_date"] = signal_date
    rankings["training_start"] = dataset.date.min()
    rankings["latest_training_target_date"] = latest_target
    rankings["training_rows"] = training_rows
    rankings["eligible_securities"] = len(eligible)
    rankings["scored_securities"] = len(rankings)

    stage = time.perf_counter()
    momentum = _momentum_rankings(prices.loc[:, prices.columns.intersection(eligible)], signal_date)
    cross = rankings.merge(
        momentum[["ticker", "momentum_3m", "rank_3m", "momentum_12_1", "rank_12_1"]],
        on="ticker", how="left",
    )
    cross["consensus_count"] = (
        cross.ml_rank.le(top_n).astype(int)
        + cross.rank_3m.le(top_n).astype(int)
        + cross.rank_12_1.le(top_n).astype(int)
    )
    quantile_input = scored[["date", "ticker", "ml_score", "future_return"]]
    quantiles = assign_score_quantiles(quantile_input).merge(
        membership[["ticker", "sector"]], on="ticker", how="left"
    ).rename(columns={"ml_score": "score"})
    quantiles["sector"] = quantiles.sector.fillna("Unknown")
    sector = _sector_exposure(rankings, membership, quantiles, top_n)
    selected = rankings.nsmallest(top_n, "ml_rank").copy()
    selected["target_weight"] = 1 / len(selected)
    selected["portfolio"] = "sp500_ml_equal_weight_research"
    target = selected[["ticker", "company", "sector", "ml_score", "ml_rank", "target_weight", "portfolio"]]
    feature_missingness = _feature_missingness(raw_features, market_snapshot.tickers, signal_date)
    eligibility = screening.report.merge(
        membership[["ticker", "company", "sector", "industry"]], on="ticker", how="left"
    )
    price_failure_codes = {
        "HAS_PRICE_HISTORY", "MISSING_PRICE", "SUFFICIENT_HISTORY",
        "INSUFFICIENT_LOOKBACK", "CURRENT_PRICE_AVAILABLE", "STALE_PRICE",
    }
    price_ready = eligibility.reason.fillna("").map(
        lambda reason: not bool(price_failure_codes.intersection(reason.split("|")))
    )
    feature_ready = price_ready & eligibility.feature_ready
    coverage = pd.DataFrame([{
        "date": signal_date, "universe_members": market_snapshot.member_count,
        "price_ready": int(price_ready.sum()),
        "feature_ready": int(feature_ready.sum()), "prediction_ready": len(rankings),
        "rejected": market_snapshot.member_count - len(eligible),
        "coverage_pct": len(rankings) / market_snapshot.member_count,
    }])
    timings["reports_seconds"] = time.perf_counter() - stage

    run_id = _run_id()
    run_dir = Path(output_root) / "runs" / run_id
    current_dir = Path(output_root) / "current"
    run_dir.mkdir(parents=True, exist_ok=False)
    current_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "eligibility_report.csv": eligibility,
        "feature_coverage.csv": coverage,
        "feature_missingness.csv": feature_missingness,
        "ml_rankings.csv": rankings,
        "momentum_rankings.csv": momentum,
        "cross_sectional_summary.csv": cross,
        "current_ml_quantiles.csv": quantiles[["ticker", "score", "quantile", "sector"]],
        "sector_exposure.csv": sector,
        "sp500_ml_target.csv": target,
        "validation_flags.csv": validation_flags,
    }
    for filename, frame in frames.items():
        frame.to_csv(run_dir / filename, index=False)
        frame.to_csv(current_dir / filename, index=False)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings["total_seconds"] = time.perf_counter() - total_started
    model = MLStockRanker().model
    params = model.get_params()
    manifest = {
        "run_id": run_id, "universe": "sp500",
        "universe_snapshot_hash": snapshot.snapshot_hash,
        "requested_as_of_date": str(snapshot.requested_date),
        "resolved_market_date": str(market_snapshot.resolved_date),
        "coverage_on_resolved_date": resolution["coverage_on_resolved_date"],
        "provider": "yahoo", "market_data_manifest": market_data_manifest,
        "eligible_count": len(eligible), "prediction_count": len(rankings),
        "feature_columns": FEATURE_COLUMNS, "model_class": type(model).__name__,
        "hyperparameters": {key: params[key] for key in ("max_iter", "learning_rate", "max_depth", "random_state")},
        "minimum_train_months": ML_MINIMUM_TRAIN_MONTHS,
        "top_n": top_n, "paper_update": paper_update,
        "warnings": [
            "Current-constituent prospective research only; historical survivorship bias is not solved.",
            "Yahoo adjusted history is revisable and not point-in-time institutional data.",
            "^GSPC is a price index while adjusted security prices include dividends.",
            "Research output only; no trade execution.",
        ],
        "outputs": [str((run_dir / filename).resolve()) for filename in frames],
        "timings": timings, "peak_memory_mb": peak / 1024 ** 2,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    (current_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    if paper_update:
        paper_dir = write_institutional_paper_target("sp500", run_id, target, manifest, paper_root)
        manifest["paper_run_directory"] = str(paper_dir.resolve())
    return {
        "manifest": manifest, "rankings": rankings, "momentum": momentum,
        "cross_sectional": cross, "quantiles": quantiles, "sector": sector,
        "target": target, "eligibility": eligibility, "coverage": coverage,
    }
