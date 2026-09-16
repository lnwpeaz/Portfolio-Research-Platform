# Validation record

The first sections retain the initial presentation validation. See the
[GitHub hardening pass](#github-hardening-pass) for the latest review.

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
- At that earlier validation, no project-local Git repository existed and no
  project files were staged, committed or published by the task. The retained
  example manifest therefore has a null Git commit and a source hash. The
  project has since been initialized and published; new historical manifests
  can record the project commit.

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


## GitHub hardening pass

This subsequent review began on a clean `main` checkout at `e8bfdb2`, with a
project-local repository and the expected `lnwpeaz/Portfolio-Research-Platform`
origin. Local tracking reported synchronization with `origin/main`; no remote
fetch or GitHub rendering test was performed. No staging, commit or push was
performed in this pass.

The change is limited to README hierarchy/Quick Start, the portfolio interview
guide, documentation consistency and environment-file ignore rules. Python
source, parameters, dependencies, market inputs and research calculations were
not changed. Existing charts, filenames and entrypoint architecture were retained.

### Execution and reproducibility

- The first unactivated `python -m pytest -q` used global Python and failed
  collection with seven missing-`pypfopt` import errors. The existing project
  environment resolved this; no dependencies were installed or changed.
- Baseline after `source venv/bin/activate`: **69 passed**, one warning,
  **51.74 seconds**.
- Post-edit tests in that environment: **69 passed**, one warning,
  **35.97 seconds**. The warning was joblib falling back to logical CPU counts.
- `python comparison_main.py`: exit 0, **16.72 seconds**.
- `python research_main.py`: exit 0, **22.74 seconds**.
- All **93 pre-existing historical CSV files** matched their saved pre-edit
  SHA-256 hashes, including **all 39 tracked CSV files**. Comparison/research
  regenerated their outputs; robustness and ignored institutional CSVs were
  checked for preservation, not regenerated in this documentation-only pass.
- Logs, baseline hashes and machine-readable check results are retained locally
  under ignored `reports/validation/github-hardening/`.

Fresh installation remains untested. `requirements.txt` is an unpinned list;
`requirements-lock.txt` captures the previously tested installation and is not a
portable or hash-verified lock. The minimal first-run path is now prominent in
README, with detailed execution order retained in the reproduction guide.

### Presentation and hygiene

All local Markdown links, fragments and README image paths were checked.
Mermaid diagrams were retained with their existing simple flowchart syntax;
this pass does not claim a remote GitHub rendering test. The five notebooks
already carry prominent superseded notices and were retained as development
history. Resume material now has one maintained home in `PORTFOLIO.md`.

The initial 194 tracked files totalled 7,095,559 bytes. The largest was the
intentional stock input at 2,397,460 bytes; no tracked file exceeded 10 MB.
Tracked text searches found no matches for personal home-directory paths,
the local user email marker, or uppercase credential markers. Additional common
private-key, cloud/GitHub-key and credential-assignment patterns produced no
suspected credentials. This is a working-tree scan, not a certification of the
entire Git history. No ignored cache/live/institutional files are tracked.
Environment secret files are now ignored, with a sanitized example exception.
`git diff --check` passed; the final changes are seven modified documentation/
ignore files plus the new portfolio guide, with no staged changes.

### Deliberate boundaries

No license was added because redistribution rights for bundled vendor data are
not established in the repository. Citation metadata and a contribution guide
were not added: this personal portfolio has no separate citation/release or
contributor workflow requiring them. No author identity or affiliation was
invented. The old example manifest remains intact as pre-Git provenance.

The strict embargo, feature allow-list, signal/realization dates, Top-5 rules,
252-price optimizer window, caps/fallbacks, drift-aware turnover, initial
turnover and 10 bps cost convention agree with implementation. The existing
methodology correctly distinguishes the generic equal-weight ML engine from
the configurable momentum cap. The audit now repeats the limitation that
legacy replay historical-context panels are not as-of-safe, although they do
not feed rankings or target weights. No strategy behavior was changed to
resolve documentation wording.

All ex-post-universe, PIT/delisting, benchmark, vendor and same-close limitations
remain prominently disclosed. The project is suitable for a resume, GitHub pin
and LinkedIn Projects as quantitative research engineering and portfolio
analytics; it does not establish unbiased alpha or production readiness.
