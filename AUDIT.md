# Quantitative Research and Engineering Audit

Audit scope: the Python research pipeline and bundled data, reviewed for
methodological correctness rather than return improvement. No model replacement
or hyperparameter tuning was performed.

## Executive assessment

The original results were not suitable for an investment claim. The most
important implementation biases are now corrected: return rows and benchmark
months align, walk-forward labels have an availability embargo, future return
availability cannot alter the selected portfolio, constraints remain binding
after normalization, and turnover is computed from drifted pre-trade weights.

The largest remaining limitation is the data. The 20 current, highly successful
large-cap names form an ex-post universe with complete histories. There is no
point-in-time membership or delisting return data. In addition, the benchmark is
the S&P 500 price index (`^GSPC`), while stock prices are dividend-adjusted. Until
these are replaced with point-in-time total-return data, reported performance is
illustrative research output, not an unbiased backtest.

## Ranked findings

| Rank | Severity | Finding | Why it matters | Resolution / concrete fix |
|---:|:---:|---|---|---|
| 1 | Critical | Fixed ex-post 20-stock universe; no delisted securities | Creates severe survivorship and selection bias. The strategy only chooses among names known to have survived and become large winners. | **Architecture ready; historical PIT data still unavailable.** Providers, half-open dated membership filtering, immutable snapshots, and explicit failure are implemented. Genuine dated records, inactive-security prices, identifier history, and delisting returns are still required. |
| 2 | Critical | Strategy returns were stored at the signal month, then compared with that month's benchmark return | Strategy return was *t to t+1* while benchmark return was *t-1 to t*, invalidating active comparisons and plots. | **Fixed.** Backtests now emit both `signal_date` and the following realization `date`; benchmark returns align to realization dates. |
| 3 | Critical | Selected names were dropped using next-month return availability before weights were formed | The ex-ante portfolio depended on a future-data field. This is look-ahead and survivorship leakage. | **Fixed.** Selection and weighting are completed first. A partially missing selected outcome now fails fast and requests delisting-aware data rather than replacing the name. |
| 4 | Critical | Walk-forward training included a label whose outcome ends at the prediction cutoff | That label is not safely available early enough to train and execute at the same cutoff. | **Fixed.** Each row has `target_date`; training requires `target_date < prediction_date` and the minimum window counts actually available labeled months. |
| 5 | High | Month-end signal uses the same close assumed for execution | A close-derived signal cannot generally be computed, trained, optimized, and filled at that exact close. This can create optimistic timing. | **Explicit remaining assumption.** For live research, either compute inputs before the market-on-close cutoff or add a next-session execution price and slippage model. The assumption is documented in code and README. |
| 6 | High | Clipping weights and renormalizing could violate the maximum weight again | A nominal 40% cap could become a 100% position; max-Sharpe results were materially overstated. | **Fixed.** Weights are projected onto a feasible capped simplex, feasibility is checked, the optimizer receives the bound directly, and weights are validated. |
| 7 | High | Turnover compared consecutive target weights rather than current drifted weights | Trades and costs were wrong after assets earned different returns. Equal holdings can require a rebalance even when the target list is unchanged. | **Fixed.** End-of-period asset values produce the next pre-trade weights; turnover is one-way `0.5 * L1`, with initial deployment set to 1. |
| 8 | High | Benchmark is `^GSPC` price return while securities use adjusted total-return prices | Dividends accrue to stocks but not the index, overstating active performance. | **Data blocked.** Replace with a point-in-time S&P 500 total-return series, or a clearly documented investable adjusted ETF proxy with fees and inception limitations. |
| 9 | High | Missing/stale prices could be silently accepted by `resample().last()` | Pandas selects the last non-null value per column, potentially valuing a suspended/missing asset at a stale price. | **Fixed.** Monthly conversion selects the final observed row and preserves cell-level missingness. Stock and benchmark month-end dates must align. |
| 10 | High | Optimizers used short startup samples despite a 252-day requested window; failures were silent | Estimates changed definition over time and fallback behavior could not be audited. | **Fixed.** Full requested history is required by default, incomplete optimizer universes fall back to equal weight, and every row records `optimization_status`; fallback rate is reported. |
| 11 | Medium | Linear costs omit spread variation, slippage, market impact, and liquidity/capacity | Constant 10 bps is not realistic across securities or regimes and can understate costs for high-turnover ranking strategies. | Keep the current transparent baseline, then add per-asset bid/ask or volume-based costs and execution delay once volume/quote data are available. Do not calibrate from final backtest returns. |
| 12 | Medium | Optimization uses sample covariance and historical mean without estimator diagnostics | Max-Sharpe is numerically and statistically unstable; silent extreme solutions are common. | Constraints and fallbacks are fixed. **Diagnostics now implemented** for condition numbers, eigenvalues, ex-ante volatility, expected returns and recorded fallback status. Historical solver objective values remain unavailable. Estimator changes are intentionally deferred because tuning/redesign was out of scope. |
| 13 | Medium | Target and future returns share the dataset object with features | This is safe only because the model uses an explicit feature allow-list; ad-hoc notebook code could accidentally include them. | Retain the explicit `FEATURE_COLUMNS` contract. For production, expose separate feature, label, and evaluation frames or a typed dataset object. |
| 14 | Medium | Adjusted Yahoo data have vendor/revision and corporate-action reproducibility risk | Re-downloads may change history; the data are not point-in-time. | **Improved, not resolved.** Raw and normalized versions are immutable, manifests preserve parameters/hashes/symbol mappings, and split/dividend/extreme-return diagnostics are emitted. Yahoo remains revisable non-PIT research data. |
| 15 | Medium | Portfolio analytics were limited to standalone return ratios | No active risk, alpha/beta, year stability, IC, ranking hit rate, or optimizer fallback reporting. | **Improved.** Added active return/tracking error/information ratio/beta/alpha, calendar-year analysis, rank IC/top-N diagnostics, maximum turnover, and fallback rate. |
| 16 | Medium | Metric edge cases were numerically incorrect | Initial-period losses were omitted from max drawdown; Sortino used the standard deviation of negative observations instead of full-sample downside deviation. | **Fixed** with finite/short-sample guards and regression tests. |
| 17 | Medium | ML dataset construction used nested date-by-ticker Python loops | Scales poorly and encourages tightly coupled feature/label logic. | **Fixed.** Feature panels are stacked/vectorized; monthly sampling is shared in a dedicated module. Walk-forward retraining remains the main expected cost. |
| 18 | Low | Architecture has hard-coded script parameters and weak run provenance | Results cannot be reproduced from a saved configuration; `src/config.py` is mostly bypassed. | **Improved.** Historical CLI manifests now record frozen settings, the fixed20 universe and input dates, input/source/output hashes, full model parameters and environment versions. Reports retain mutable filenames; preserve each report set with its manifest. Live and institutional provenance retain their existing conventions. |
| 19 | Low | There were no unit tests | Timing, leakage, cap, turnover, and metric regressions could recur unnoticed. | **Fixed baseline.** Added 66 tests spanning the corrected engine, robustness phases, prospective cutoff safety, deterministic live predictions, universe/data providers, cache extension, market-date resolution, investability, quantiles, portfolio constraints, paper isolation, signed rebalance transactions, and frozen-weight evaluation. |
| 20 | Low | Several research diagnostics remain absent | Factor decay, quantile monotonicity, exposure attribution, tail risk, regime stability, and capacity are needed before production review. | Concentration, rolling risk, drawdown duration, ranking time series, attribution, current quantiles, and equal-weight sector exposures are present. Historical PIT exposure, style factors, VaR/ES, liquidity capacity, and formal factor attribution remain. |

## Review by requested area

- **Look-ahead, target leakage, feature leakage, walk-forward:** critical date
  and outcome-availability paths are corrected. Explicit feature selection
  prevents label columns entering the model. Same-close execution remains a
  declared assumption.
- **Portfolio construction, optimization, weights:** ranking precedes outcome
  inspection; optimizer inputs end at the signal date; caps are feasible and
  binding; incomplete optimization histories are visible fallbacks.
- **Costs and turnover:** initial deployment, one-way turnover, drifted weights,
  and deterministic holdings are implemented. A market-aware cost model awaits
  liquidity data.
- **Benchmark:** dates now align, but total-return comparability is blocked by
  the bundled `^GSPC` price-index dataset.
- **Survivorship and missing data:** future-dependent dropping is removed and
  stale month-end values are prevented. Point-in-time membership and delisting
  data are still required.
- **Numerical stability:** finite weights, feasibility, short samples, drawdown,
  downside deviation, and optimizer fallback visibility are improved. Max-
  Sharpe input and weight stability diagnostics are now available; historical
  solver objective values remain unavailable.
- **Architecture and performance:** common calendar logic and portfolio state
  helpers reduce duplication; ML panel creation is vectorized. Shared settings
  and historical/live manifests are implemented; typed result objects remain
  an architectural opportunity.
- **Tests, diagnostics, analytics:** a focused regression suite and first-level
  ranking/active/yearly diagnostics now exist. Exposure, capacity, attribution,
  and regime analytics remain necessary for an institutional-grade platform.

## Validation

- `python -m pytest -q`: 66 passed after the institutional market-data extension.
- Both `main.py` and `ml_main.py` completed against the bundled data using a
  non-interactive plotting backend.
- `comparison_main.py` produces and saves common-period performance and yearly
  return tables after inner-joining every strategy and the benchmark.
- `research_main.py` adds a conventional 12-1 momentum baseline and saves
  rolling risk, concentration, drawdown, attribution, yearly, ranking, and
  visualization outputs over the common period.
- `robustness_main.py` runs leave-one-out universe, fixed-subperiod,
  turnover-based transaction-cost, and one-period execution-delay experiments
  without changing the official strategies or configuration.
- `robustness_phase2_main.py` adds paired moving-block sampling stability,
  contribution/outlier counterfactuals, optimizer and ML stability diagnostics,
  rolling 24-month consistency, and a non-scored robustness scorecard.
- `live_main.py` completed at the latest bundled market observation and created
  an immutable initial paper recommendation for all five strategies.
- `live_replay_main.py --as-of 2024-12-31` resolved to 2024-12-31, retained all
  20 eligible securities, and excluded all later market and label observations.
- No model class or hyperparameter was changed, and no return optimization or
  strategy tuning was performed.

## Final implementation hardening

- Important run parameters now come from `src/config.py`, including an explicit
  `same_close` execution mode and one-way turnover cost convention.
- Optimizer fallbacks distinguish insufficient history, missing data, and
  expected optimizer failure; each backtest row preserves the concise reason.
- Non-optimizer reports display optimizer fallback as `N/A`.
- The drawdown chart and numerical metric share the same initial-capital
  convention.
- Ranking diagnostic definitions are explicit and unchanged.
- The 12-1 signal uses `price[t-1] / price[t-12] - 1`; realized contributions
  reconcile to period gross returns and transaction costs remain separate.
- The fixed-universe, delisting, benchmark total-return, and point-in-time data
  limitations described above remain unresolved by design.

## Phase 2 statistical robustness findings

### Corrected implementation biases

The timing, leakage, outcome-availability, weight-cap, turnover, cost, and
benchmark-date alignment corrections above remain regression-tested. Phase 2
does not change any official signal, portfolio, optimizer, cost, or execution
assumption.

### Remaining data biases

The fixed ex-post universe remains the critical limitation. Moving-block
bootstrap intervals describe stability conditional on this already selected
sample; they do not repair survivorship bias. Missing delisting returns,
point-in-time membership, and the price-index/total-return mismatch likewise
remain unresolved.

### Statistical uncertainty

Paired block resampling, contribution concentration, removal of extreme months,
rolling 24-month windows, optimizer input diagnostics, and fixed-score ML Rank
IC decay are now reported separately from investable backtests. Contribution
subtraction and best-month removal are explicitly labeled non-investable.
Optimizer objective values are unavailable from stored historical results and
are reported as missing rather than reconstructed or fabricated. The unified
scorecard intentionally contains no synthetic robustness rating.

With six-month blocks and 5,000 simulations, median bootstrap Sharpes range
from 1.124 (Min Vol) to 1.361 (3M), while 5th-percentile Sharpes remain positive
(0.658 to 0.745). Estimated probabilities of positive active return range from
93.8% (Min Vol) to 100.0% after rounding (12-1 and ML). These are conditional
sampling-stability results, not bias-free significance claims.

Positive contribution is concentrated: the leading security supplies 24.1% of
3M, 24.5% of 12-1, 28.6% of Max Sharpe, 16.6% of Min Vol, and 31.0% of ML
positive attribution. Removing the best three realized months reduces Sharpe
to 1.168, 1.010, 0.945, 0.895, and 1.110 respectively, but does not eliminate
performance. Max Sharpe and Min Vol average only 3.21 and 3.35 effective
holdings, with average maximum weights near 37.9%; covariance condition numbers
are moderate (median 18.3, 95th percentile 54.2) rather than numerically extreme.

ML mean Rank IC varies substantially by year, from -0.109 in the partial 2018
sample to 0.120 in 2024. Fixed-score mean Rank IC is 0.0386, 0.0332, and 0.0331
at cumulative one-, two-, and three-month horizons. Across 65 complete rolling
24-month windows, ML has positive active return in 100% and exceeds benchmark
Sharpe in 95.4%; 12-1 records 100% and 90.8%. Min Vol is least consistent at
67.7% and 66.2%.

## Prospective Research Architecture

The live layer is separate from historical backtests. Every run resolves an
explicit as-of cutoff, snapshots prices and benchmark data once, records
freshness and a deterministic input hash, and passes the cutoff through feature
generation, ML training, portfolios, reporting, and paper state. Live features
exclude targets, future returns, and evaluation columns. ML training requires
every included `target_date` to be strictly before prediction.

Run directories and paper recommendations are immutable. Later paper evaluation
uses frozen target weights and only then-available realization prices; it does
not regenerate an old ranking. Benchmark dates match portfolio realization
dates and transaction costs remain separate. Historical replay calls the same
runner with a truncated snapshot, making later observations inaccessible.

Legacy reports can additionally load full-sample historical context and risk
flags. Those panels are not as-of-safe evidence and do not feed rankings or
portfolio weights; their limitation remains explicit in the methodology.

These controls prevent prospective look-ahead through market rows, features,
labels, benchmark observations, and portfolio returns. They do not fix or
downgrade the critical historical survivorship limitation. Live outputs remain
research recommendations—not orders, fills, or expected profits—and carry that
warning in every manifest and report.

## Institutional universe architecture status

The strategy layer can consume fixed20, S&P 500, and future SET50 providers
through a common immutable snapshot. S&P 500 current membership can come from a
dated external CSV or the published current table. Historical runs require
genuine `effective_from`/`effective_to` records and filter every feature row at
its signal date using the documented half-open boundary. Missing history raises
`POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE`.

Investability and feature coverage distinguish membership from usable market
data. Quantiles, Q5-Q1, Rank IC, equal-weight sector exposure, sector association,
and fixed Top-N sensitivity are diagnostics; none changes the model or selects
a parameter. The bundled data remain the same 20 ex-post survivors. Therefore
finding 1 is not resolved: architecture ready; historical PIT data still
unavailable.

## Current S&P 500 market-data status

The current provider resolved 503 S&P securities. The Yahoo research adapter
downloaded all constituent histories plus `^GSPC` over the configured five-year
window, while retaining canonical and vendor tickers. The broad-market resolver
selected 2026-08-21 at 100% same-date price coverage. Five hundred securities
met the 252-observation and current-feature requirements and were scored by the
unchanged embargoed ML model; three shorter histories remained explicit
rejections.

This enables reproducible current/prospective cross-sectional decisions, not a
historical S&P 500 backtest. The acquired histories are for today's constituents,
Yahoo is revisable, delisting returns are absent, and historical membership was
not obtained. Survivorship bias remains unresolved and no unbiased historical
S&P 500 performance is reported.

## Portfolio presentation and reproducibility update

The landing page now links a research report, exact methodology, reproduction
guide, output map, repository audit and actual validation record. Early
notebooks are explicitly marked as superseded; their old calculations are not
current engine outputs. Historical figures carry ex-post-universe and
price-benchmark warnings. The small bundled input files are explicit Git-ignore
exceptions, while downloaded caches and repeated runs remain ignored.

Historical CLI provenance supplements existing live manifests with frozen
settings and input/source/output hashes. Legacy live snapshots are frozen in
memory and save metadata, not complete persisted price panels; exact replay
requires preserved source inputs. Institutional acquisition does retain
versioned raw and normalized panels. Source hashing works even without a
project-local Git repository, as in the earlier pre-publication review. The
project now has its own repository; new manifests can also record its commit.

The test suite now includes provenance coverage (69 passing tests). See
[validation](docs/VALIDATION.md) for executed commands, numeric reproduction
checks and warnings. Official strategies, hyperparameters, portfolio selection,
costs, constraints and execution assumptions were unchanged. The earlier
66-test validation above records the preceding development stage.

The current S&P 500 status above describes the stored 2026-08-21 market
snapshot; it is not a newly acquired or continuously current membership claim.
All historical survivorship, delisting, vendor and benchmark limitations remain.
