# Research output map

Root CSV/PNG files are compact representative outputs, retained at existing
paths for compatibility. They are illustrative ex-post fixed20 research,
with a price-index benchmark and same-close execution. No file is evidence of
unbiased historical S&P 500 performance.

| Producer | Main artifacts |
|---|---|
| `comparison_main.py` | `common_period_performance.csv`, `common_period_yearly_returns.csv` |
| `research_main.py` | Those tables plus `common_period_returns.csv`, `common_period_turnover.csv`, `rolling_risk_metrics.csv`, `drawdown_summary.csv`, `concentration*`, `contribution*`, `ranking_diagnostics_timeseries.csv`; wealth/drawdown/rolling/yearly/IC figures |
| `robustness_main.py` | `leave_one_out*`, `subperiod*`, `transaction_cost*`, `execution_sensitivity*`, corresponding figures |
| `robustness_phase2_main.py` | `bootstrap*`, `contribution_*stress/counterfactuals`, `return_outlier*`, `optimizer*`, `ml_*`, `rolling_24m*`, `robustness_scorecard.csv` and figures |
| Historical CLI wrapper | `provenance/<entrypoint>_<id>.json`: settings, hashes, package versions and success/failure |
| `live_main.py` | `live/<run_id>/`: manifest, features, predictions, targets, consensus, flags, research report; `live/latest.json` pointer |
| `live_replay_main.py` | `replay/<replay_id>/runs/`, replay summary, isolated paper state for date ranges |
| Institutional CLIs | `institutional/`: acquisition coverage/actions, current scoring, sector/quantile/Top-N diagnostics and dated run directories |

Root report filenames are mutable on rerun. Immutable provenance manifests do
not archive the old report bytes; retain both when sharing an experiment.
Phase 2 depends on Phase 1 CSVs. Benchmark-relative values inherit the
`^GSPC` price-return versus stock adjusted-return mismatch.

Repeated live/replay/institutional runs, validation logs and provenance files
are ignored to keep Git review small. The JSON environment/input/source example
in `docs/EXAMPLE_RUN_MANIFEST.json` is a retained representative execution.
`data/institutional/` contains ignored versioned raw/normalized market data;
`paper/` contains ignored mutable ledger state and immutable recommendation runs.

Start with the [README figures](../README.md#key-research-outputs), then the
[research report](../docs/RESEARCH_REPORT.md) and full CSVs. Historical optimizer
stability uses its available strategy history; concentration summaries in the
common-period report use the shared 88 months. Do not compare them as if their
sample lengths were identical.
