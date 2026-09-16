"""Institutional universe research entry point."""

import argparse
from pathlib import Path
import sys
import time
import pandas as pd

from src.data.loader import load_benchmark, load_prices
from src.evaluation.institutional_pipeline import run_current_research, run_historical_research
from src.universe import POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE, PointInTimeMembershipUnavailable
from src.universe.registry import create_universe_provider
from src.universe.sp500 import fetch_current_sp500


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("fixed20", "sp500", "set50"), required=True)
    parser.add_argument("--as-of")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--historical-membership", type=Path)
    parser.add_argument("--current-membership", type=Path)
    parser.add_argument("--top-n", type=int)
    parser.add_argument("--prices", type=Path, help="Parquet price panel for the selected universe")
    parser.add_argument("--benchmark", type=Path, help="Parquet benchmark series")
    parser.add_argument("--output", type=Path, default=Path("reports/institutional"))
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    top_n = args.top_n if args.top_n is not None else (5 if args.universe == "fixed20" else 25)
    if top_n < 1:
        raise ValueError("--top-n must be positive.")
    market_started = time.perf_counter()
    prices, benchmark = load_prices(args.prices), load_benchmark(args.benchmark)
    print(f"Market data load: {time.perf_counter() - market_started:.3f}s; {prices.shape[1]} securities")
    if args.as_of or args.end:
        requested = pd.Timestamp(args.as_of or args.end)
    elif args.universe == "fixed20":
        requested = pd.Timestamp(min(prices.index.max(), benchmark.index.max()))
    else:
        requested = pd.Timestamp.today().normalize()
    current_frame = None
    if args.universe == "sp500" and args.current_membership is None and args.historical_membership is None:
        print("Universe load: fetching published current S&P 500 constituents...")
        try:
            current_frame = fetch_current_sp500()
        except Exception as error:
            print(f"{POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE}: current source unavailable: {error}")
            return 2
    kwargs = {}
    if args.universe != "fixed20":
        kwargs = {
            "historical_path": args.historical_membership,
            "current_path": args.current_membership,
            "current_frame": current_frame,
        }
    provider = create_universe_provider(args.universe, **kwargs)
    historical_requested = bool(args.start or args.end)
    try:
        snapshot = provider.get_snapshot(
            requested, require_point_in_time=historical_requested and args.universe != "fixed20"
        )
    except PointInTimeMembershipUnavailable as error:
        print(error)
        return 2
    print(f"Universe load: {snapshot.member_count} members; hash={snapshot.snapshot_hash[:12]}")
    if historical_requested or args.universe == "fixed20":
        if args.start:
            prices, benchmark = prices.loc[pd.Timestamp(args.start):], benchmark.loc[pd.Timestamp(args.start):]
        prices, benchmark = prices.loc[:requested], benchmark.loc[:requested]
        manifest = run_historical_research(provider, snapshot, prices, benchmark, args.output, top_n)
    else:
        manifest = run_current_research(
            provider, snapshot, prices, benchmark, args.output / "current", top_n
        )
    print(pd.Series(manifest, dtype=object).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
