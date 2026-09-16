"""As-of-safe historical replay of the prospective research architecture."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from src.data.frequency import last_observation_by_month
from src.data.loader import load_prices
from src.live.run_config import LiveRunConfig
from src.live.runner import run_live_research


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of")
    parser.add_argument("--start"); parser.add_argument("--end")
    parser.add_argument("--frequency", choices=["monthly"], default="monthly")
    args = parser.parse_args()
    if bool(args.as_of) == bool(args.start or args.end):
        parser.error("Specify either --as-of or both --start and --end.")
    if (args.start is None) != (args.end is None): parser.error("--start and --end are required together.")
    prices = load_prices(); universe = tuple(prices.columns)
    replay_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    root = Path("reports/replay") / replay_id
    if args.as_of:
        dates = [pd.Timestamp(args.as_of)]
        paper_update = False
    else:
        monthly_dates = last_observation_by_month(prices).index
        dates = list(monthly_dates[(monthly_dates >= pd.Timestamp(args.start)) & (monthly_dates <= pd.Timestamp(args.end))])
        paper_update = True
    results = []
    for date in dates:
        config = LiveRunConfig(as_of_date=date, universe=universe, output_directory=root / "runs")
        result = run_live_research(config, paper_update=paper_update, paper_dir=root / "paper")
        results.append(result)
    summary = pd.DataFrame([{"requested_as_of": date, "resolved_market_date": r["snapshot"].metadata["latest_market_date"], "run_id": r["run_id"], "eligible_securities": len(r["snapshot"].metadata["eligible_universe"]), "evaluated_previous": not r["evaluated"].empty} for date, r in zip(dates, results)])
    root.mkdir(parents=True, exist_ok=True); summary.to_csv(root / "replay_summary.csv", index=False)
    print("=" * 60); print("PROSPECTIVE RESEARCH REPLAY"); print("=" * 60)
    print(summary.to_string(index=False)); print(f"Outputs: {root}")
    print("Future observations were excluded at every snapshot boundary.")


if __name__ == "__main__": main()
