from pathlib import Path
import pandas as pd

from src.live.run_manifest import RESEARCH_WARNINGS


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact table without adding an optional tabulate dependency."""
    values = frame.copy().fillna("N/A").astype(str)
    header = "| " + " | ".join(values.columns) + " |"
    separator = "|" + "|".join(["---"] * len(values.columns)) + "|"
    rows = ["| " + " | ".join(row) + " |" for row in values.to_numpy()]
    return "\n".join([header, separator, *rows])


def generate_research_report(run_dir: Path, manifest, signals, predictions, portfolios,
                             summaries, consensus, flags, historical_context=None):
    lines = ["# Portfolio Research Report", "", "## Run Information", "",
             f"- As-of date: {manifest['as_of_date']}", f"- Signal date: {manifest['signal_date']}",
             f"- Eligible universe: {len(manifest['eligible_universe'])}",
             f"- Data source: {manifest['data_source']}", ""]
    lines += ["## Current Rankings", "", "### 3M Momentum", "", _markdown_table(signals.nsmallest(5, "rank_3m")[["ticker", "momentum_3m", "rank_3m"]]), "", "### 12–1 Momentum", "", _markdown_table(signals.nsmallest(5, "rank_12_1")[["ticker", "momentum_12_1", "rank_12_1"]]), "", "### ML Ranking", "", _markdown_table(predictions.nsmallest(5, "ml_rank")[["ticker", "ml_score", "ml_rank"]]), ""]
    lines += ["## Current Portfolios", "", _markdown_table(portfolios[["strategy", "ticker", "target_weight", "optimization_status"]]), "", "## Cross-Strategy Agreement", "", _markdown_table(consensus.head(10)), "", "## Portfolio Risk", "", _markdown_table(summaries), ""]
    if historical_context is not None and not historical_context.empty:
        lines += ["## Historical Context", "", "Historical diagnostics are context only and are not forecasts of future return.", "", _markdown_table(historical_context), ""]
    lines += ["## Risk Flags", "", _markdown_table(flags) if not flags.empty else "No configured flags triggered.", "", "## Research Warnings", ""]
    lines += [f"- {warning}" for warning in RESEARCH_WARNINGS]
    (run_dir / "research_report.md").write_text("\n".join(lines) + "\n")
    summary = summaries.copy()
    summary.insert(0, "run_id", manifest["run_id"]); summary.insert(1, "as_of_date", manifest["as_of_date"]); summary.insert(2, "signal_date", manifest["signal_date"])
    summary.to_csv(run_dir / "research_summary.csv", index=False)
