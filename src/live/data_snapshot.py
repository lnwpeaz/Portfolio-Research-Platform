"""Explicit and reproducible market-data snapshots."""

from dataclasses import dataclass
import hashlib
import json
import pandas as pd

from src.live.run_config import LiveRunConfig


@dataclass(frozen=True)
class DataSnapshot:
    prices: pd.DataFrame
    benchmark: pd.Series
    status: pd.DataFrame
    metadata: dict


def assert_no_future_market_data(prices, benchmark, as_of_date):
    cutoff = pd.Timestamp(as_of_date)
    if (not prices.empty and prices.index.max() > cutoff) or (not benchmark.empty and benchmark.index.max() > cutoff):
        raise ValueError("Future market observation detected in live snapshot.")


def _hash_snapshot(prices, benchmark, as_of_date, universe):
    digest = hashlib.sha256()
    digest.update(str(pd.Timestamp(as_of_date).isoformat()).encode())
    digest.update(json.dumps(list(universe), separators=(",", ":")).encode())
    digest.update(pd.util.hash_pandas_object(prices, index=True).values.tobytes())
    digest.update(pd.util.hash_pandas_object(benchmark, index=True).values.tobytes())
    return digest.hexdigest()


def create_data_snapshot(
    prices: pd.DataFrame, benchmark: pd.Series, config: LiveRunConfig,
    created_at: str, stale_calendar_days: int = 7,
) -> DataSnapshot:
    requested = list(config.universe)
    missing_columns = [ticker for ticker in requested if ticker not in prices]
    requested_as_of = config.as_of_date or min(prices.index.max(), benchmark.index.max())
    cutoff = pd.Timestamp(requested_as_of)
    # Truncation happens once, visibly, at the snapshot boundary.
    sliced_prices = prices.loc[prices.index <= cutoff, [t for t in requested if t in prices]].copy()
    sliced_benchmark = benchmark.loc[benchmark.index <= cutoff].copy()
    if sliced_prices.empty or sliced_benchmark.empty:
        raise ValueError("No market data exist on or before requested as-of date.")
    latest_market = min(sliced_prices.index.max(), sliced_benchmark.index.max())
    sliced_prices = sliced_prices.loc[:latest_market]
    sliced_benchmark = sliced_benchmark.loc[:latest_market]
    assert_no_future_market_data(sliced_prices, sliced_benchmark, cutoff)
    status_rows, eligible = [], []
    for ticker in requested:
        if ticker in missing_columns:
            status_rows.append({"ticker": ticker, "requested": True, "eligible": False, "latest_price_date": pd.NaT, "staleness_days": pd.NA, "status": "ineligible", "reason": "missing_ticker"})
            continue
        valid = sliced_prices[ticker].dropna()
        latest = valid.index.max() if not valid.empty else pd.NaT
        staleness = (cutoff.normalize() - latest.normalize()).days if pd.notna(latest) else pd.NA
        reason = None if pd.notna(latest) and staleness <= stale_calendar_days else ("missing_price" if pd.isna(latest) else "stale_price")
        is_eligible = reason is None
        if is_eligible: eligible.append(ticker)
        status_rows.append({"ticker": ticker, "requested": True, "eligible": is_eligible, "latest_price_date": latest, "staleness_days": staleness, "status": "eligible" if is_eligible else "ineligible", "reason": reason})
    input_hash = _hash_snapshot(sliced_prices, sliced_benchmark, cutoff, requested)
    metadata = {
        "snapshot_id": input_hash[:16], "as_of_date": cutoff.isoformat(), "created_at": created_at,
        "requested_universe": requested, "eligible_universe": eligible,
        "ineligible_tickers": [r["ticker"] for r in status_rows if not r["eligible"]],
        "latest_market_date": latest_market.isoformat(),
        "latest_benchmark_date": sliced_benchmark.dropna().index.max().isoformat(),
        "benchmark_staleness_days": int((cutoff.normalize() - sliced_benchmark.dropna().index.max().normalize()).days),
        "data_source": config.data_source, "row_count": len(sliced_prices), "input_hash": input_hash,
    }
    return DataSnapshot(sliced_prices, sliced_benchmark, pd.DataFrame(status_rows), metadata)
