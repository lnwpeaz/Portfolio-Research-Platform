"""CLI for prospective research output and research-only paper portfolios."""

import argparse
from pathlib import Path

from src.data.loader import load_prices
from src.live.run_config import LiveRunConfig
from src.live.runner import run_live_research


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-of", help="Research cutoff; defaults to latest bundled market observation.")
    parser.add_argument("--no-paper-update", action="store_true")
    args = parser.parse_args()
    universe = tuple(load_prices().columns)
    config = LiveRunConfig(as_of_date=args.as_of, universe=universe)
    result = run_live_research(config, paper_update=not args.no_paper_update, paper_dir=Path("paper"))
    snapshot = result["snapshot"].metadata
    print("=" * 60); print("LIVE PORTFOLIO RESEARCH"); print("=" * 60)
    print(f"Run ID: {result['run_id']}")
    print(f"As-of date: {snapshot['as_of_date']}")
    print(f"Latest market observation: {snapshot['latest_market_date']}")
    print(f"Eligible securities: {len(snapshot['eligible_universe'])}")
    for title, column in (("3M Momentum", "rank_3m"), ("12-1 Momentum", "rank_12_1")):
        print(f"\n{title}\n{'-' * len(title)}")
        print(", ".join(result["signals"].nsmallest(5, column).ticker))
    print("\nML Ranking\n----------"); print(", ".join(result["predictions"].nsmallest(5, "ml_rank").ticker))
    print("\nTarget Portfolios\n-----------------")
    for strategy, group in result["portfolios"].groupby("strategy", sort=False):
        print(f"{strategy}: " + ", ".join(f"{r.ticker} {r.target_weight:.1%}" for r in group.itertuples()))
    print("\nCross-Strategy Consensus\n------------------------")
    print(result["consensus"].head(5)[["ticker", "number_strategies_selecting_or_holding"]].to_string(index=False))
    print("\nRisk Flags\n----------")
    print(result["flags"][["scope", "flag"]].to_string(index=False) if not result["flags"].empty else "None")
    print("\nPaper Portfolio\n---------------")
    if args.no_paper_update: print("Paper state update disabled.")
    else:
        print(f"Previous recommendation evaluated: {'yes' if not result['evaluated'].empty else 'no'}")
        print(f"New recommendation recorded: {'yes' if not result['recommendations'].empty else 'no'}")
    print(f"Research outputs: {result['run_dir']}")
    print("=" * 60); print("RESEARCH OUTPUT — NOT LIVE TRADE EXECUTION"); print("=" * 60)


if __name__ == "__main__": main()
