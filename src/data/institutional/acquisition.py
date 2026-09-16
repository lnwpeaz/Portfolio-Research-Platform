"""Incremental acquisition orchestration and reproducible run manifests."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import pandas as pd

from src.data.institutional.cache import InstitutionalDataCache, merge_incremental
from src.data.institutional.corporate_actions import corporate_action_flags
from src.data.institutional.coverage import resolve_common_market_date
from src.data.institutional.metadata import acquisition_status, utc_now
from src.data.institutional.validation import security_readiness, validate_market_panel


@dataclass(frozen=True)
class AcquisitionRun:
    panel: pd.DataFrame
    status: pd.DataFrame
    validation_flags: pd.DataFrame
    corporate_action_flags: pd.DataFrame
    manifest: dict
    timings: dict


def _run_id():
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:8]


def acquire_market_data(
    provider, cache: InstitutionalDataCache, tickers, start_date, end_date,
    universe_snapshot_hash, *, refresh=False, minimum_coverage=0.95,
    benchmark_ticker="^GSPC", minimum_observations=252,
):
    started = time.perf_counter()
    requested = tuple(dict.fromkeys(tickers))
    all_names = requested + ((benchmark_ticker,) if benchmark_ticker not in requested else ())
    mapping = provider.normalizer.mapping_frame(all_names)
    existing, previous_manifest = cache.load_latest(provider.name)
    if not existing.empty:
        existing = existing[existing.canonical_ticker.isin(all_names)]
    fetches = []
    cached_names = set(existing.canonical_ticker) if not existing.empty else set()
    missing = tuple(ticker for ticker in all_names if ticker not in cached_names)
    if missing:
        fetches.append((missing, pd.Timestamp(start_date), pd.Timestamp(end_date)))
    # Extend an earlier requested window without discarding the cached suffix.
    if not existing.empty:
        prior_start = pd.Timestamp(previous_manifest.get("requested_start")) if previous_manifest else pd.NaT
        if pd.isna(prior_start) or pd.Timestamp(start_date) < prior_start:
            prefix_end = prior_start - pd.Timedelta(days=1) if pd.notna(prior_start) else existing.date.min() - pd.Timedelta(days=1)
            fetches.append((tuple(t for t in all_names if t in cached_names), pd.Timestamp(start_date), prefix_end))
        prior_end = pd.Timestamp(previous_manifest.get("requested_end")) if previous_manifest else pd.NaT
        if refresh:
            refresh_start = max(pd.Timestamp(start_date), existing.date.max() - pd.Timedelta(days=10))
            fetches.append((tuple(t for t in all_names if t in cached_names), refresh_start, pd.Timestamp(end_date)))
        elif pd.isna(prior_end) or prior_end < pd.Timestamp(end_date):
            latest = existing.groupby("canonical_ticker").date.max()
            extension = tuple(t for t in all_names if t in latest)
            if extension:
                fetches.append((extension, latest.loc[list(extension)].min() + pd.Timedelta(days=1), pd.Timestamp(end_date)))

    fetch_started = time.perf_counter()
    results, raw_parts, failures = [], [], {}
    for names, fetch_start, fetch_end in fetches:
        if not names or fetch_start > fetch_end:
            continue
        result = provider.fetch_prices(names, fetch_start, fetch_end)
        results.append(result)
        if not result.panel.empty:
            raw_parts.append(result.panel.copy())
        failures.update(result.failure_reasons)
    fetch_seconds = time.perf_counter() - fetch_started
    panel = merge_incremental(existing, [result.panel for result in results])
    panel = panel[panel.date.le(pd.Timestamp(end_date))].copy() if not panel.empty else panel
    if panel.empty:
        raise RuntimeError("Market-data acquisition produced no observations.")

    validation_started = time.perf_counter()
    flags = validate_market_panel(panel, requested_date=end_date)
    resolution = resolve_common_market_date(
        panel, requested, end_date, minimum_coverage=minimum_coverage
    )
    status = acquisition_status(
        panel, requested, mapping[mapping.canonical_ticker.isin(requested)],
        resolution["resolved_market_date"], failures,
    )
    readiness = security_readiness(
        panel, requested, resolution["resolved_market_date"],
        minimum_observations=minimum_observations,
    )
    status = status.merge(readiness, left_on="canonical_ticker", right_on="ticker", how="left").drop(columns="ticker")
    actions = corporate_action_flags(panel[panel.canonical_ticker.isin(requested)])
    validation_seconds = time.perf_counter() - validation_started

    run_id = _run_id()
    successful = status.loc[status.status.eq("successful"), "canonical_ticker"].tolist()
    failed = status.loc[status.status.eq("failed"), "canonical_ticker"].tolist()
    partial = status.loc[status.status.eq("partial"), "canonical_ticker"].tolist()
    provider_parameters = results[-1].provider_parameters if results else (
        previous_manifest.get("provider_parameters", {}) if previous_manifest else {}
    )
    manifest = {
        "run_id": run_id, "provider": provider.name,
        "requested_start": str(pd.Timestamp(start_date).date()),
        "requested_end": str(pd.Timestamp(end_date).date()),
        "requested_tickers": list(requested),
        "successful_tickers": successful, "failed_tickers": failed,
        "partial_tickers": partial, "created_at": utc_now(),
        "provider_parameters": provider_parameters,
        "input_universe_snapshot_hash": universe_snapshot_hash,
        "benchmark_ticker": benchmark_ticker,
        "resolved_market_date": str(resolution["resolved_market_date"].date()),
        "coverage_on_resolved_date": resolution["coverage_on_resolved_date"],
        "downloaded_tickers": sorted({t for result in results for t in result.successful_tickers}),
        "cached_tickers": sorted(cached_names.intersection(requested)),
        "warnings": [
            "Yahoo adjusted data are revisable and are not point-in-time institutional data.",
            "Current constituents must not be used as historical membership.",
        ],
        "timings": {
            "data_fetch_load_seconds": fetch_seconds,
            "validation_seconds": validation_seconds,
        },
    }
    if raw_parts:
        manifest, cache_manifest_path = cache.write_version(
            provider.name, run_id, raw_parts, panel, manifest
        )
        manifest["cache_manifest"] = str(cache_manifest_path.resolve())
    elif previous_manifest:
        manifest.update({
            "raw_files": [],
            "normalized_file": previous_manifest["normalized_file"],
            "normalized_fingerprint": previous_manifest["normalized_fingerprint"],
            "cache_reused_without_download": True,
        })
    else:
        raise RuntimeError("No cache version or downloaded data are available.")
    timings = {
        "data_fetch_load_seconds": fetch_seconds,
        "validation_seconds": validation_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    manifest["timings"] = timings
    return AcquisitionRun(panel, status, flags, actions, manifest, timings)


def write_acquisition_reports(run, output_dir="reports/institutional/data"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run.status.to_csv(output / "data_acquisition_status.csv", index=False)
    run.validation_flags.to_csv(output / "validation_flags.csv", index=False)
    run.corporate_action_flags.to_csv(output / "corporate_action_flags.csv", index=False)
    manifest = dict(run.manifest)
    manifest["files_written"] = [
        str((output / name).resolve()) for name in (
            "data_acquisition_status.csv", "validation_flags.csv",
            "corporate_action_flags.csv", "data_manifest.json",
        )
    ]
    (output / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
    )
    return manifest
