# Validation record

Validated 2026-09-16 using the existing Python 3.11.4 environment on macOS,
with `MPLBACKEND=Agg` and a writable temporary Matplotlib config directory.
A fresh environment installation and external vendor downloads were not tested.

## Tests and executed entrypoints

Baseline: **66 passed**, one joblib warning. After adding three provenance
cases: **69 passed**, one joblib warning. The final post-edit full suite also
passed **69 tests in 46.61 seconds** (47.08 seconds process wall time). Output is retained
locally at `reports/validation/final_tests.log`.

All commands below exited successfully. Durations are observed wall time, not
benchmarks; other validation processes sometimes ran concurrently.

| Executed command | Exit | Seconds |
|---|---:|---:|
| `python -m pytest -q` | 0 | 47.16 |
| `python main.py` | 0 | 3.72 |
| `python ml_main.py` | 0 | 17.30 |
| `python comparison_main.py` | 0 | 16.52 |
| `python research_main.py` | 0 | 18.36 |
| `python robustness_main.py` | 0 | 429.08 |
| `python robustness_phase2_main.py` | 0 | 241.63 |
| `python live_main.py` | 0 | 1.87 |
| `python live_replay_main.py --as-of 2024-12-31` | 0 | 1.63 |
| `python live_replay_main.py --as-of 2024-12-31` | 0 | 1.79 |
| `python institutional_main.py --universe fixed20 --output reports/institutional/fixed20` | 0 | 44.53 |

`research_main.py` was rerun after the final turnover export/figure changes.
The six historical entrypoints wrote provenance manifests. A representative
research run is preserved in [EXAMPLE_RUN_MANIFEST.json](EXAMPLE_RUN_MANIFEST.json).
The raw command logs and machine-readable results are retained locally under
`reports/validation/` and ignored for Git review.

## Integrity checks

- **37 of 37 existing root historical CSVs reproduce byte-for-byte**, compared
  against SHA-256 hashes saved before changes. New common-return/turnover
  exports are additional artifacts; official historical numbers did not change.
- Backtest, portfolio, feature, model-fit, cost and configuration source files
  were unchanged. `walk_forward.py` changed only its explanatory docstring.
- Two CLI replays at 2024-12-31 produced identical input hashes and identical
  CSV bytes for features, momentum signals, ML predictions, target portfolios,
  portfolio summaries and consensus. Their run IDs/timestamps naturally differ.
- Replay cutoff was 2024-12-31; latest training label date was **2024-11-29**.
  Feature dates were no later than the cutoff. Existing tests additionally
  exercise future-observation exclusion and immutable frozen-weight evaluation.
- Replay input hash:
  `82906d8b273391f14dcab433804575b58c5b192c3d65954e849b598a9c585816`.
- Default live mode resolved the latest bundled date **2025-12-31**, with 20
  eligible securities. It created a new immutable research recommendation;
  no later period existed to evaluate. This is not a new market-data download.
- The original 24 root figures and five institutional figures were reviewed.
  Historical figures were regenerated with visible data-limit warnings; one
  turnover chart was added. README image and local documentation paths were
  checked. The fixed20 sector figure remains uninformative because metadata
  are all `Unknown`; the report explicitly says so.
- The shareable files total approximately **7 MB**, including the two bundled
  data inputs. No visible file exceeds 10 MB. Caches, virtual environments,
  repeated runs, paper state and validation logs are ignored.
- No project-local Git repository exists in this workspace and no project files
  were staged, committed or published. New historical manifests correctly
  record a null Git commit and a deterministic source hash.

## Warnings and boundaries

The only test/runtime environment warning observed was joblib's inability to
read physical CPU count in this sandbox; it falls back to logical core counts.
This warning includes stack-location text but did not cause a command failure.
Live concentration flags are research diagnostics, not execution errors.

Network-dependent `institutional_data_main.py` and
`institutional_live_main.py` were not rerun. Current membership/market-data
status retained in the old audit is dated context, not a new acquisition claim.
Setup instructions are provided, but fresh package installation is unverified.

The historical limitations remain: ex-post survivors, no delistings/PIT
membership, a price-index benchmark, revisable vendor history, same-close
execution and simple costs. Legacy replay also loads separately labeled
historical context/risk diagnostics from root reports, which can cover later
periods; those are not as-of evidence and do not feed predictions or weights.
Legacy live market panels are frozen in memory but not persisted in their run
directories. These reporting/provenance boundaries are explicit in the
[methodology](METHODOLOGY.md).
