# Portfolio project delivery

This records the initial presentation delivery. The subsequent GitHub pass is
recorded in [validation](VALIDATION.md#github-hardening-pass); use the concise
[portfolio guide](PORTFOLIO.md) for current resume and interview material.

## Important files changed

| Files | Change |
|---|---|
| `README.md` | Finance-readable landing page, architecture diagrams, all-strategy results, safeguards, limitations, representative figures and navigation |
| `AUDIT.md` | Updated provenance/optimizer status and presentation-stage notes; retained prior findings and historical validation |
| `docs/RESEARCH_REPORT.md` | Research question, data, signals, ML, portfolio protocol, results, diagnostics, robustness, prospective architecture and limitations |
| `docs/METHODOLOGY.md` | Exact feature/date/target conventions, strict label embargo, costs, constraints, missingness, metrics and cutoff boundaries |
| `docs/REPRODUCIBILITY.md` | Environment, frozen inputs, execution order, output retention and verified/unverified boundaries |
| `docs/REPOSITORY_AUDIT.md` | Initial inventory, legacy inconsistencies and implemented plan |
| `docs/OPERATIONS.md` | Retained detailed former README operational reference, with validation caveats |
| `docs/VALIDATION.md` | Actual commands, tests, reproducibility/determinism checks, warnings and limitations |
| `docs/EXAMPLE_RUN_MANIFEST.json` | Representative successful research configuration, model parameters and hashes |
| `docs/DELIVERY.md` | This inventory, final tree, research status and resume options |
| `reports/README.md` | Producer-to-artifact map; sample-period and retention conventions |
| `data/BUNDLED_DATA.json` | Frozen input identities, dates, tickers, sizes and hashes |
| `requirements-lock.txt` | Snapshot of the tested installed environment; no fresh-install claim |
| `.gitignore` | Keep two small reproduction inputs; ignore local/cache/download/repeated-run state |
| `src/evaluation/provenance.py` | Frozen configuration plus historical CLI success/failure manifests |
| `tests/test_provenance.py` | Source/input change detection and success/failure output provenance coverage |
| `main.py`, `ml_main.py`, `comparison_main.py`, `robustness_main.py`, `robustness_phase2_main.py` | Historical provenance wrapper only; calculations preserved |
| `research_main.py` | Provenance wrapper, common-period return/turnover exports and turnover figure |
| `src/evaluation/visualization.py` | Readable legends, visible bias/benchmark footnotes, date labels and turnover figure |
| `src/evaluation/institutional_plots.py` | Visible historical data-limit/manifest footnotes |
| `src/models/walk_forward.py` | Corrected explanatory embargo example only; executable logic unchanged |
| `notebooks/01_download_data.ipynb` through `05_validate_backtest.ipynb` | Added prominent superseded-notebook notice; retained development history |
| `reports/common_period_returns.csv`, `reports/common_period_turnover.csv`, `reports/common_period_turnover.png` | Additional small representative exports/figure |
| Existing root report PNGs and institutional PNGs | Regenerated with research-limit labels; original historical CSV bytes unchanged |

Dated provenance, live/replay, institutional validation runs and local logs
were also generated, but are ignored for Git. No commits were made during that
initial presentation task; the project was subsequently initialized and published.

## Final architecture

```text
portfolio-research-platform/
├── README.md
├── AUDIT.md
├── requirements.txt
├── requirements-lock.txt
├── main.py / ml_main.py
├── comparison_main.py / research_main.py
├── robustness_main.py / robustness_phase2_main.py
├── live_main.py / live_replay_main.py
├── institutional_main.py / institutional_data_main.py / institutional_live_main.py
├── src/
│   ├── config.py
│   ├── data/                  # loader, calendar, providers, cache versions
│   ├── universe/              # fixed20, S&P, SET50 contract, investability
│   ├── features/              # momentum, explicit ML feature/label panel
│   ├── models/                # classifier and embargoed walk-forward
│   ├── portfolio/             # selection, optimization, cap projection
│   ├── backtest/              # realized returns, drift and costs
│   ├── evaluation/            # analytics, robustness, figures, provenance
│   └── live/                  # cutoff snapshots, ranking, paper evaluation
├── tests/                     # 14 modules, 69 tests
├── docs/                      # report, methodology, operations and validation
├── data/
│   ├── BUNDLED_DATA.json
│   ├── raw/                   # two frozen bundled reproduction files
│   ├── processed/             # ignored obsolete notebook output
│   └── institutional/         # ignored acquired/versioned data
├── reports/
│   ├── README.md
│   ├── *.csv / *.png          # small representative research outputs
│   ├── provenance/           # ignored exclusive historical manifests
│   ├── live/ / replay/        # ignored repeated prospective runs
│   ├── institutional/        # ignored data/provider diagnostics
│   └── validation/           # ignored execution logs and check records
├── paper/                     # ignored research ledger and frozen decisions
└── notebooks/                 # five explicitly superseded explorations
```

Existing report paths and module boundaries were preserved. A cosmetic
`outputs/` migration would have broken consumers without improving research.

## Research status

**Completed for this portfolio version:** five unchanged official strategies,
embargoed ML, audited date/selection/cost/constraint behavior, active/ranking/
risk analytics, both robustness phases, prospective paper/replay architecture,
provider contracts, historical provenance, professional documentation and
validated example outputs. See [validation](VALIDATION.md) for all executions.

**Remaining external-data limitations:** genuine PIT constituents, inactive
securities, identifier history, delisting returns and a consistent total-return
benchmark. Revisable Yahoo history and the ex-post universe still prevent an
unbiased historical investment claim. Better engineering does not create
missing historical data.

**Optional future work:** execution prices and realistic liquidity/capacity
costs, formal factor attribution and tail-risk/regime analysis, persisted legacy
live price snapshots, and strictly cutoff-aware historical-context reporting.
None was used to retune or replace the official strategies.

## Recruiter view

A short README review should convey an end-to-end equity research project:
financial signals become constrained portfolios, returns are evaluated under
explicit timing/cost rules, ML trains only on available labels, and robustness
analyses challenge the findings. Its central strength is transparent research
engineering and reproducibility. Its high historical returns are not claimed
as unbiased evidence of investment skill.

## Resume and interview material

See [PORTFOLIO.md](PORTFOLIO.md) for three resume options, short and detailed
interview explanations, and technical questions. This is the maintained version
of the recruiter material previously listed here.
