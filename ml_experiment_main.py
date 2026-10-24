"""Run the predeclared ML regularization experiment without altering v1.0."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from threadpoolctl import threadpool_limits

from src.data.loader import load_benchmark, load_prices
from src.evaluation.ml_experiment import ROOT, run_ml_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        help="New run directory; existing directories are never overwritten.",
    )
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8]
    output = args.output_dir or ROOT / "reports/ml_regularization" / run_id

    def progress(stage, candidate):
        print(f"{stage}: {candidate}", flush=True)

    # Small monthly panels benefit from avoiding OpenMP oversubscription.
    with threadpool_limits(limits=1):
        result = run_ml_experiment(load_prices(), load_benchmark(), output, progress)
    print(f"\nFrozen validation selection: {result['selection']['selected_candidate']}")
    for stage in ("validation", "evaluation"):
        print(f"\n{stage.upper()} — chronological, cash-reset, net of modeled costs")
        print(result[stage][[
            "candidate", "is_selected", "annual_alpha", "sharpe",
            "annual_volatility", "max_drawdown",
        ]].to_string(index=False))
    print(f"\nReport: {output.resolve() / 'report.md'}")
    print("Retrospective evaluation; previously inspected history, not an unseen holdout.")


if __name__ == "__main__":
    main()
