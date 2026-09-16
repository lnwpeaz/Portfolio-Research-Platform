"""Generate current S&P 500 research from the validated institutional cache."""

import argparse
import json
from pathlib import Path
import sys
import pandas as pd

from src.live.institutional_runner import run_institutional_live
from src.universe.sp500 import SP500UniverseProvider


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", choices=("sp500",), default="sp500")
    parser.add_argument("--current-membership", type=Path, default=Path("reports/institutional/current/universe_snapshot.csv"))
    parser.add_argument("--data-manifest", type=Path, default=Path("reports/institutional/data/data_manifest.json"))
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--minimum-coverage", type=float, default=0.95)
    parser.add_argument("--paper-update", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.data_manifest.exists():
        raise FileNotFoundError("Run institutional_data_main.py before institutional_live_main.py.")
    market_manifest = json.loads(args.data_manifest.read_text())
    requested = pd.Timestamp(market_manifest["requested_end"])
    provider = SP500UniverseProvider(current_path=args.current_membership)
    snapshot = provider.get_snapshot(requested)
    if snapshot.snapshot_hash != market_manifest["input_universe_snapshot_hash"]:
        raise ValueError("Current universe snapshot does not match the market-data manifest.")
    panel = pd.read_parquet(market_manifest["normalized_file"])
    result = run_institutional_live(
        snapshot, panel, market_manifest, top_n=args.top_n,
        minimum_coverage=args.minimum_coverage, paper_update=args.paper_update,
    )
    manifest = result["manifest"]
    print("=" * 60)
    print("INSTITUTIONAL LIVE RESEARCH")
    print("=" * 60)
    print(f"Universe: S&P500\nResolved market date: {manifest['resolved_market_date']}")
    print(f"Eligible: {manifest['eligible_count']}\nScored: {manifest['prediction_count']}\nTop-N: {manifest['top_n']}")
    print("Paper update: " + ("enabled (paper/sp500 only)" if args.paper_update else "disabled"))
    print("Outputs: reports/institutional/current")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
