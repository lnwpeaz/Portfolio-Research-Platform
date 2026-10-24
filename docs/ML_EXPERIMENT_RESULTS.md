# ML regularization: results and reproduction

The experiment changes only the existing classifier's tree depth and L2
regularization. The [protocol](ML_REGULARIZATION.md) was fixed before running
the four candidates. The original model remains the default so the original
research and live workflows remain reproducible.

## Selection and evaluation

Selection used 24 realization months in 2021–2022, with the original baseline
eligible to win. The highest finite net Sharpe selected **depth 3, L2 = 10**:
validation Sharpe 0.853 versus baseline 0.691. Its configuration was saved
before evaluating the 36 realization months in 2023–2025.

| 2023–2025 metric | Original ML | Selected ML | Change |
|---|---:|---:|---:|
| CAGR, net of modeled costs | 35.04% | 36.01% | +0.97 percentage points |
| Annual single-benchmark alpha | 9.71% | 11.23% | +1.52 percentage points |
| Net Sharpe | 1.861 | 1.996 | +0.136 before rounding |
| Annual volatility | 17.09% | 16.22% | −0.87 percentage points |
| Maximum drawdown | −13.29% | −10.57% | 2.72 percentage points less severe |

Both segments start in cash. The original five features, expanding monthly
training, strict `target_date < prediction_date` rule, Top-5 equal weights,
same-close execution and 10 bps one-way-turnover costs are retained.

Validation volatility did **not** improve: 21.38% for baseline versus 21.44%
for the selected model. The selection objective was net Sharpe, not simultaneous
improvement on every metric. All metrics and all candidates are retained in
the [complete report](../reports/ml_regularization/study_v1/report.md).

Another candidate, depth 3 / L2 = 1, has higher evaluation Sharpe (2.133), but
it was not the validation winner and was not substituted after seeing those
results. This study ends with the preselected configuration, rather than
starting another search on the same evaluation history.

The 2023–2025 history had already been inspected. This is a retrospective
chronological evaluation, not an unseen holdout, a significance claim or a
prospective performance record. Twenty ex-post survivors, absent delistings,
the adjusted-stock versus price-index benchmark mismatch and same-close
assumptions continue to limit interpretation. Alpha is the existing annualized
single-benchmark regression intercept at zero risk-free rate; it is not excess
CAGR or factor-adjusted alpha. Results above must not be compared directly to
the continuous 2018–2025 baseline table.

## Run the study

From the repository root in the existing Python environment:

```bash
export MPLBACKEND=Agg
python ml_experiment_main.py
```

Each run creates a new directory under `reports/ml_regularization/`. Use
`--output-dir /absolute/path/to/new-run` to choose another new directory.
Existing directories are rejected. The CLI uses one numerical thread and
does not require a download or a new dependency installation in the recorded
environment.

The selected variant is available through the existing ML code by passing
`model_parameters={"max_depth": 3, "l2_regularization": 10.0}` to
`generate_walk_forward_scores`, or `l2_regularization=10.0` to
`MLStockRanker`. Omitted arguments retain the original baseline. The experiment
CLI handles date splits, comparison, freezing and report retention together.

## Evidence to retain

- `protocol.md` and `config.json`: predeclared candidates and complete model parameters.
- `selection.json`: validation-only decision and validation/config hashes.
- `validation_*` and `evaluation_*`: every candidate's metrics, net returns,
  scores and cash-reset portfolio ledgers.
- `comparison.png` and `report.md`: presentation view and full interpretation.
- `manifest.json`: data-frame/source/output hashes, Python/package versions,
  status and timestamps.

Source hashes include uncommitted Python sources, the protocol and dependency
files. Preserve all files together; rerunning the CLI never overwrites this
study. Original reports and input files are separately checked for changes.

## Verification record

The final full suite passed **105 tests** (91 existing and 14 new) in 20.00
seconds with `OMP_NUM_THREADS=1 MPLBACKEND=Agg python -m pytest -q` in the
recorded environment. New cases cover validation-only selection, exact ties,
incomplete data, undefined Sharpe, baseline compatibility, regularization,
future-label isolation, aligned months, cash resets, frozen selection,
existing-directory protection and failure manifests.

Two complete study executions produced byte-identical results in all **eight
CSV files**, plus identical configuration and selection JSON files. The final
run's **90 source hashes** and **13 output hashes** were checked against actual
files. All **217 pre-existing CSV/input files** retained their original hashes.
An independent read-only review also reconciled portfolio accounting and
regenerated the baseline scores across all 60 segment signal months.
