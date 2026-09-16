"""Orchestration helpers for current and PIT institutional research."""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np
import pandas as pd

from src.config import ML_MINIMUM_TRAIN_MONTHS, ONE_WAY_TURNOVER_COST_BPS
from src.data.frequency import last_observation_by_month
from src.evaluation.benchmark import SP500_PRICE_INDEX
from src.evaluation.cross_sectional import (
    cross_sectional_diagnostics,
    long_short_diagnostic,
    quantile_monotonicity,
    quantile_returns,
    quantile_summary,
    sector_attribution_summary,
    sector_exposure,
    topn_sensitivity,
)
from src.evaluation.institutional_plots import (
    plot_long_short, plot_quantile_performance, plot_rank_ic,
    plot_sector_exposure, plot_universe_coverage,
)
from src.features.ml_dataset import FEATURE_COLUMNS, build_ml_dataset
from src.features.momentum import calculate_monthly_momentum_signal
from src.models.ml_ranker import MLStockRanker
from src.models.walk_forward import generate_walk_forward_scores
from src.universe.investability import InvestabilityConfig, screen_investability


def _write_json(path, value):
    def default(item):
        if isinstance(item, (pd.Timestamp, Path)):
            return str(item)
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, default=default) + "\n")


def feature_coverage(dataset, snapshot, prices):
    monthly = last_observation_by_month(prices.loc[:, prices.columns.intersection(snapshot.tickers)])
    counts = dataset.groupby("date").ticker.nunique().rename("feature_complete")
    rows = []
    for date in monthly.index:
        available = int(monthly.loc[date].notna().sum())
        complete = int(counts.get(date, 0))
        rows.append({
            "date": date, "universe_members": snapshot.member_count,
            "eligible_members": available, "feature_complete": complete,
            "feature_missing": max(0, available - complete),
            "prediction_eligible": complete,
        })
    return pd.DataFrame(rows)


def ticker_feature_diagnostics(dataset, snapshot, prices, tickers=None):
    expected_dates = dataset.date.nunique()
    counts = dataset.groupby("ticker").size()
    rows = []
    for ticker in (snapshot.tickers if tickers is None else tuple(tickers)):
        rows.append({
            "ticker": ticker,
            "price_observations": int(prices[ticker].notna().sum()) if ticker in prices else 0,
            "feature_complete_rows": int(counts.get(ticker, 0)),
            "missing_feature_rows_vs_panel": max(0, expected_dates - int(counts.get(ticker, 0))),
        })
    return pd.DataFrame(rows)


def point_in_time_feature_coverage(dataset, membership, prices):
    monthly = last_observation_by_month(prices)
    complete_counts = dataset.groupby("date").ticker.nunique()
    rows = []
    for date in monthly.index:
        active = membership[
            membership.effective_from.le(date)
            & (membership.effective_to.isna() | membership.effective_to.gt(date))
        ]
        tickers = active.ticker.drop_duplicates().tolist()
        present = monthly.columns.intersection(tickers)
        eligible = int(monthly.loc[date, present].notna().sum())
        complete = int(complete_counts.get(date, 0))
        rows.append({
            "date": date, "universe_members": len(tickers),
            "eligible_members": eligible, "feature_complete": complete,
            "feature_missing": max(0, eligible - complete),
            "prediction_eligible": complete,
        })
    return pd.DataFrame(rows)


def apply_point_in_time_membership(dataset, membership):
    """Retain signal rows satisfying effective_from <= date < effective_to."""
    intervals = membership[
        ["ticker", "effective_from", "effective_to"]
    ].copy()
    intervals["effective_from"] = pd.to_datetime(intervals.effective_from)
    intervals["effective_to"] = pd.to_datetime(intervals.effective_to)
    joined = dataset.merge(intervals, on="ticker", how="inner", validate="many_to_many")
    active = joined.date.ge(joined.effective_from) & (
        joined.effective_to.isna() | joined.date.lt(joined.effective_to)
    )
    result = joined.loc[active, dataset.columns]
    if result.duplicated(["date", "ticker"]).any():
        raise ValueError("Overlapping membership intervals create duplicate active rows.")
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def ml_dataset_summary(dataset, scores):
    rows_per_date = dataset.groupby("date").size()
    feature_missing = dataset[FEATURE_COLUMNS].isna().to_numpy().mean()
    return pd.DataFrame([{
        "total_rows": len(dataset),
        "number_of_securities": dataset.ticker.nunique(),
        "start_date": dataset.date.min(), "end_date": dataset.date.max(),
        "average_rows_per_date": rows_per_date.mean(),
        "minimum_rows_per_date": rows_per_date.min(),
        "maximum_rows_per_date": rows_per_date.max(),
        "positive_target_fraction": dataset.target.dropna().astype(float).mean(),
        "missing_feature_rate": feature_missing,
        "prediction_rows": len(scores),
        "prediction_dates": scores.date.nunique(),
    }])


def walk_forward_training_rows(dataset, prediction_dates):
    labeled = dataset[dataset.target_date.notna() & dataset.target.notna()].copy()
    target_dates = np.sort(labeled.target_date.to_numpy(dtype="datetime64[ns]"))
    rows = []
    for date in sorted(pd.to_datetime(pd.Series(prediction_dates).dropna().unique())):
        rows.append({
            "prediction_date": date,
            "training_rows": int(np.searchsorted(target_dates, np.datetime64(date), side="left")),
            "prediction_rows": int(dataset.date.eq(date).sum()),
        })
    return pd.DataFrame(rows)


def _current_scores(dataset, eligible, prediction_date, return_timings=False):
    training = dataset[
        dataset.ticker.isin(eligible) & dataset.target_date.notna()
        & dataset.target_date.lt(prediction_date) & dataset.target.notna()
    ]
    prediction = dataset[
        dataset.ticker.isin(eligible) & dataset.date.eq(prediction_date)
    ].copy()
    if training.date.nunique() < ML_MINIMUM_TRAIN_MONTHS or training.target.nunique() < 2:
        raise ValueError("Insufficient eligible history for current ML ranking.")
    model = MLStockRanker()
    fit_started = time.perf_counter()
    model.fit(training)
    fit_seconds = time.perf_counter() - fit_started
    score_started = time.perf_counter()
    prediction["ml_score"] = model.predict_scores(prediction)
    scoring_seconds = time.perf_counter() - score_started
    prediction = prediction.sort_values(["ml_score", "ticker"], ascending=[False, True])
    prediction["rank"] = range(1, len(prediction) + 1)
    if return_timings:
        return prediction, len(training), {
            "ml_training_seconds": fit_seconds,
            "scoring_seconds": scoring_seconds,
        }
    return prediction, len(training)


def run_current_research(provider, snapshot, prices, benchmark, output_dir, top_n=25):
    """Rank the current intersection only; never backcast current membership."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prices = prices.loc[prices.index <= snapshot.resolved_date]
    benchmark = benchmark.loc[benchmark.index <= snapshot.resolved_date]
    dataset = build_ml_dataset(prices, benchmark)
    current_date = dataset.date.max()
    current_features = dataset[dataset.date.eq(current_date)]
    screening = screen_investability(
        snapshot, prices, current_features,
        InvestabilityConfig(required_features=tuple(FEATURE_COLUMNS)),
    )
    eligible = screening.eligible_tickers
    if eligible:
        rankings, training_rows = _current_scores(dataset, eligible, current_date)
    else:
        rankings = pd.DataFrame(columns=["date", "ticker", "ml_score", "rank"])
        training_rows = 0
    monthly = last_observation_by_month(prices.loc[:, prices.columns.intersection(eligible)])
    momentum = calculate_monthly_momentum_signal(monthly, lookback_months=3).loc[current_date]
    momentum = momentum.dropna().sort_values(ascending=False).rename("momentum_3m").reset_index()
    momentum.columns = ["ticker", "momentum_3m"]
    momentum["rank"] = range(1, len(momentum) + 1)
    membership = snapshot.to_frame()
    ranking_sectors = rankings.merge(membership[["ticker", "sector"]], on="ticker", how="left")
    if ranking_sectors.empty:
        sector = pd.DataFrame(columns=["sector", "eligible_count", "topn_count", "eligible_equal_weight", "topn_equal_weight", "active_equal_weight"])
    else:
        sector = ranking_sectors.assign(selected=lambda x: x["rank"].le(top_n)).groupby("sector", dropna=False).agg(
            eligible_count=("ticker", "size"), topn_count=("selected", "sum")
        ).reset_index()
        sector["sector"] = sector.sector.fillna("Unknown")
        sector["eligible_equal_weight"] = sector.eligible_count / len(rankings)
        sector["topn_equal_weight"] = sector.topn_count / min(top_n, len(rankings))
        sector["active_equal_weight"] = sector.topn_equal_weight - sector.eligible_equal_weight
    coverage = feature_coverage(dataset, snapshot, prices)
    snapshot.to_frame().to_csv(output_dir / "universe_snapshot.csv", index=False)
    screening.report.to_csv(output_dir / "eligibility_report.csv", index=False)
    coverage.tail(1).to_csv(output_dir / "feature_coverage.csv", index=False)
    rankings[["date", "ticker", "ml_score", "rank"]].to_csv(output_dir / "ml_rankings.csv", index=False)
    momentum.to_csv(output_dir / "momentum_rankings.csv", index=False)
    sector.to_csv(output_dir / "sector_exposure.csv", index=False)
    manifest = {
        "mode": "current_prospective_only",
        "provider": provider.name,
        "requested_date": snapshot.requested_date,
        "resolved_date": snapshot.resolved_date,
        "point_in_time_verified": snapshot.point_in_time_verified,
        "snapshot_hash": snapshot.snapshot_hash,
        "member_count": snapshot.member_count,
        "eligible_count": len(eligible),
        "rejected_count": len(screening.rejected_tickers),
        "training_rows": training_rows,
        "prediction_rows": len(rankings),
        "top_n": top_n,
        "benchmark": SP500_PRICE_INDEX.__dict__,
        "warnings": list(snapshot.warnings) + [SP500_PRICE_INDEX.warning],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest


def run_historical_research(provider, snapshot, prices, benchmark, output_dir, top_n=25):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    started = time.perf_counter()
    timings = {}
    stage = time.perf_counter()
    if provider.name == "fixed20":
        research_prices = prices.loc[:, prices.columns.intersection(snapshot.tickers)]
    else:
        research_prices = prices
    dataset = build_ml_dataset(research_prices, benchmark)
    membership_history = None
    if provider.name != "fixed20":
        membership_history = provider._historical()
        if membership_history is None or membership_history.empty:
            raise ValueError("Historical research requires dated membership records.")
        dataset = apply_point_in_time_membership(dataset, membership_history)
    timings["feature_generation_seconds"] = time.perf_counter() - stage
    stage = time.perf_counter()
    scores = generate_walk_forward_scores(dataset, minimum_train_months=ML_MINIMUM_TRAIN_MONTHS)
    timings["ml_training_seconds"] = time.perf_counter() - stage
    stage = time.perf_counter()
    ic = cross_sectional_diagnostics(scores, top_n=top_n)
    quantiles = quantile_returns(scores)
    qsummary = quantile_summary(quantiles)
    monotonicity = quantile_monotonicity(quantiles)
    long_short = long_short_diagnostic(quantiles)
    membership = snapshot.to_frame() if membership_history is None else membership_history
    sectors = sector_exposure(scores, membership, top_n=top_n)
    sector_summary = sector_attribution_summary(sectors, scores, top_n=top_n)
    sensitivity = topn_sensitivity(scores, cost_bps=ONE_WAY_TURNOVER_COST_BPS)
    dataset_stats = ml_dataset_summary(dataset, scores)
    training_rows = walk_forward_training_rows(dataset, scores.date.unique())
    coverage = (
        feature_coverage(dataset, snapshot, research_prices)
        if membership_history is None
        else point_in_time_feature_coverage(dataset, membership_history, research_prices)
    )
    if membership_history is None:
        ticker_missing = ticker_feature_diagnostics(dataset, snapshot, prices)
    else:
        historical_tickers = tuple(sorted(membership_history.ticker.unique()))
        ticker_missing = ticker_feature_diagnostics(
            dataset, snapshot, prices, tickers=historical_tickers
        )
    timings["portfolio_and_diagnostics_seconds"] = time.perf_counter() - stage
    outputs = {
        "cross_sectional_ic.csv": ic, "ml_quantile_returns.csv": quantiles,
        "ml_quantile_summary.csv": qsummary,
        "ml_quantile_monotonicity.csv": monotonicity,
        "ml_long_short_diagnostic.csv": long_short,
        "ml_sector_exposure.csv": sectors,
        "ml_sector_attribution.csv": sector_summary,
        "topn_sensitivity.csv": sensitivity,
        "feature_coverage.csv": coverage,
        "ticker_feature_missingness.csv": ticker_missing,
        "ml_dataset_summary.csv": dataset_stats,
        "walk_forward_training_rows.csv": training_rows,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    plot_quantile_performance(quantiles, output_dir / "ml_quantile_performance.png")
    plot_long_short(quantiles, output_dir / "ml_q5_q1.png")
    plot_rank_ic(ic, output_dir / "ml_rank_ic.png")
    plot_universe_coverage(coverage, output_dir / "universe_coverage.png")
    plot_sector_exposure(sectors, output_dir / "ml_sector_exposure.png")
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    timings["total_seconds"] = time.perf_counter() - started
    latest_features = dataset[dataset.date.eq(dataset.date.max())]
    current_screen = screen_investability(snapshot, prices, latest_features)
    manifest = {
        "mode": "historical_reference" if provider.name == "fixed20" else "historical_point_in_time",
        "provider": provider.name,
        "point_in_time_verified": snapshot.point_in_time_verified,
        "snapshot_hash": snapshot.snapshot_hash,
        "universe_members": snapshot.member_count,
        "eligible_count": len(current_screen.eligible_tickers),
        "rejected_count": len(current_screen.rejected_tickers),
        "ml_rows": len(dataset), "prediction_rows": len(scores),
        "rebalance_dates": scores.date.nunique(),
        "average_securities_per_prediction_date": scores.groupby("date").size().mean(),
        "peak_memory_mb": peak / 1024 ** 2,
        "timings": timings,
        "top_n": top_n,
        "warnings": list(snapshot.warnings),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    return manifest
