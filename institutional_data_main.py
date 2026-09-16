"""Acquire, cache, and validate current institutional-universe market data."""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import pandas as pd

from src.data.institutional.acquisition import acquire_market_data, write_acquisition_reports
from src.data.institutional.cache import InstitutionalDataCache
from src.data.institutional.yahoo_provider import YahooMarketDataProvider
from src.universe.sp500 import SP500UniverseProvider, fetch_current_sp500


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("sp500",), default="sp500")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--current-membership", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path("data/institutional"))
    parser.add_argument("--batch-size", type=int, default=75)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.years < 1:
        raise ValueError("--years must be positive.")
    universe_started = time.perf_counter()
    current = None if args.current_membership else fetch_current_sp500()
    provider = SP500UniverseProvider(current_path=args.current_membership, current_frame=current)
    requested_end = pd.Timestamp(args.end or pd.Timestamp.today()).normalize()
    snapshot = provider.get_snapshot(requested_end)
    universe_seconds = time.perf_counter() - universe_started
    requested_start = pd.Timestamp(args.start) if args.start else requested_end - pd.DateOffset(years=args.years) - pd.Timedelta(days=31)
    market_provider = YahooMarketDataProvider(batch_size=args.batch_size, retries=args.retries)
    cache = InstitutionalDataCache(args.cache_root)
    run = acquire_market_data(
        market_provider, cache, snapshot.tickers, requested_start, requested_end,
        snapshot.snapshot_hash, refresh=args.refresh,
        minimum_coverage=args.minimum_coverage,
    )
    run.manifest["timings"]["universe_seconds"] = universe_seconds
    manifest = write_acquisition_reports(run)
    Path("reports/institutional/current").mkdir(parents=True, exist_ok=True)
    snapshot.to_frame().to_csv("reports/institutional/current/universe_snapshot.csv", index=False)
    status = run.status
    print("=" * 60)
    print("INSTITUTIONAL MARKET DATA")
    print("=" * 60)
    print(f"Universe: S&P500\nMembers requested: {snapshot.member_count}\nProvider: yahoo")
    print(f"Requested date: {requested_end.date()}\nResolved market date: {manifest['resolved_market_date']}")
    print(f"Downloaded: {len(manifest['downloaded_tickers'])}\nCached: {len(manifest['cached_tickers'])}")
    print(f"Failed: {len(manifest['failed_tickers'])}\nPartial: {len(manifest['partial_tickers'])}")
    print(f"Price ready: {int(status.price_ready.sum())}\nCoverage: {manifest['coverage_on_resolved_date']:.2%}")
    print("Outputs: reports/institutional/data")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
