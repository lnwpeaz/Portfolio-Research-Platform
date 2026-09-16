# Reproduce the research

Run commands from the repository root. Validation used Python 3.11.4 on macOS
and the existing `venv`; [validation results](VALIDATION.md) distinguish executed
commands from setup instructions. No network download is needed for the bundled
historical workflows. A fresh installation was not exercised during this review.

## Environment setup

For the shortest first run, follow the [README Quick Start](../README.md#quick-start).
The setup below uses the recorded package snapshot for closer reproduction.
Activate the project environment before running tests: an unrelated global
Python can lack `pypfopt` even when the existing project `venv` is complete.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/portfolio-research-mpl
```

`requirements-lock.txt` is the actual installed environment snapshot, including
solver/transitive packages and notebook tools; it is not a cross-platform,
hash-verified lock. `requirements.txt` is the original unpinned dependency list.
The snapshot excludes unused `tqdm`, which is in that original list but was not
installed in the tested environment. Prefer the snapshot for reproduction;
package/solver/platform differences can change numerical results. Do not adjust
model parameters to satisfy a regression guard after an environment change.

## Frozen data requirements

Keep `data/raw/us_stock.parquet` and `data/raw/benchmark.parquet` unchanged.
The loader accepts the bundled Yahoo-style MultiIndex `Close` frames and sorts
the date index. They resolve to 2,766 daily observations, 20 stocks, and one
benchmark, from 2015-01-02 through 2025-12-31. The two files total about 2.5 MB
and are explicit exceptions to the Parquet ignore rule. Their exact SHA-256
hashes and tickers are in [the input manifest](../data/BUNDLED_DATA.json).

The old download notebook can fetch revised data and overwrite these files;
it is **not** an exact-reproduction step. Raw acquisition timestamps and original
vendor request manifests were not preserved for these legacy inputs. Hashing
them now establishes file identity, not point-in-time vendor provenance. The
input manifest's period describes observed rows, not a claimed acquisition date.

## Validated execution order

For a first review, run **tests → comparison → research**. The comparison
prints/saves the common-period table; research adds figures and diagnostics.
The longer sequence below covers standalone reports, robustness and prospective
workflows. Institutional/current-data commands are optional and separate.

```bash
python -m pytest -q
python main.py
python ml_main.py
python comparison_main.py
python research_main.py
python robustness_main.py
python robustness_phase2_main.py
python live_main.py
python live_replay_main.py --as-of 2024-12-31
python institutional_main.py --universe fixed20 --output reports/institutional/fixed20
```

The first two strategy CLIs primarily print results; `main.py` also displays a
figure when using an interactive backend. Comparison and research save root
`reports/` tables/figures. Research also saves common-period returns and turnover.
Phase 1 retrains leave-one-out ML models. Phase 2 runs bootstrap and contributor
removal reruns and **reads Phase 1 reports**; keep this order. These are the
longest steps. Both contain existing checks against the bundled ML baseline;
they are not generic performance requirements for new data.

`live_main.py` uses the latest **bundled market date**, 2025-12-31, not today's
calendar date. It updates the research-only `paper/` ledger by default. Existing
same-date recommendations are not rewritten. Its `--no-paper-update` option
supports a report-only run. Replay writes under its own dated namespace and
single-date replay does not update paper state. Repeated replay run IDs and
timestamps differ; rankings, features, target weights, and input hashes should
match with identical data and environment.

## Provenance and report retention

The six historical CLIs save exclusive JSON files under `reports/provenance/`.
They record frozen settings, full model parameters, input hashes, source hashes,
package versions, status, and report hashes. Stable report filenames remain
compatible with existing consumers; they are overwritten on rerun. Archive
reports with their manifest when preserving an experiment. A console-only run
can legitimately have no report hashes.

Live and institutional manifests retain their existing conventions. Legacy
live snapshots are in-memory copies; only metadata and derived reports are
saved. Institutional acquisition separately versions raw and normalized data.
The project now has its own Git repository. New historical manifests record
its commit alongside source hashes, which also identify uncommitted Python
changes. The retained example manifest predates Git initialization, so its null
Git field is historical provenance, not a description of the current checkout.

## External-data operations

The [operations reference](OPERATIONS.md) retains institutional acquisition,
current membership, replay ranges, and PIT input-schema details. Those
network-dependent commands were not rerun as part of the bundled-data validation;
the listed command syntax is not a promise that a vendor is currently available.
S&P current research needs a constituent source and matching cached prices;
SET50 and historical S&P research remain blocked without genuine dated data.
No current acquisition was used to refresh or improve historical performance.
