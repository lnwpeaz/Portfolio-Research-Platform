# Methodology and timing contract

This document describes the official fixed20 engine, not an unbiased S&P 500
backtest. See [the audit](../AUDIT.md) for the historical corrections and
[research report](RESEARCH_REPORT.md) for interpretation. No strategy or model
parameter was selected during the portfolio presentation update.

## Calendar and information availability

`src/data/frequency.py` selects the final observed **row** in each calendar
month. Dates remain actual observation dates; individual missing cells are
preserved. No forward fill is applied. Stock and benchmark monthly observation
dates must agree when constructing the ML dataset.

For adjusted price P at signal month t, the holding-period outcome is
`r[i,t+1] = P[i,t+1] / P[i,t] - 1`. Features and ranking use observations through
t only. The backtest row stores `signal_date = t` and `date = t+1`, the
realization date. In the feature/score panel, `date = t` and `target_date = t+1`.
The paper evaluator spells the latter `realization_date`.

**Same-close execution** is the sole official execution mode. A signal derived
from the closing price is assumed filled at that close; this is optimistic.
The separate one-period-delay experiment holds previously formed weights one
month later. It is not a next-session fill or a realistic slippage simulation.

## Features and target

The explicit `FEATURE_COLUMNS` allow-list contains exactly:

| Feature | Formula at month t |
|---|---|
| momentum_1m | P[t] / P[t−1] − 1 |
| momentum_3m | P[t] / P[t−3] − 1 |
| momentum_6m | P[t] / P[t−6] − 1 |
| volatility_3m | Sample standard deviation of the last 3 monthly returns × √12 |
| drawdown_3m | P[t] / max(P[t], P[t−1], P[t−2]) − 1 |

The binary target is 1 exactly when next-month stock return exceeds next-month
`^GSPC` price return, and 0 otherwise. It is nullable when either outcome is
unavailable. Dataset rows are retained according to feature completeness, not
future returns. The terminal cross-section can therefore be scored without
having a realized target.

`HistGradientBoostingClassifier` uses 200 iterations, learning rate 0.05,
maximum depth 3, and random state 42; other settings are library defaults,
recorded in historical provenance. The fitted probability of class 1 is a
ranking score, not a calibrated expected return.

## Walk-forward embargo

At each prediction date d, training admits only rows satisfying:

`target_date is not null AND target_date < d AND target is not null`.

At least 36 distinct labeled **feature months** and both classes are required.
The window expands; a fresh model is fitted at every prediction month. For an
August month-end prediction, July features whose outcomes end in August are
excluded; June features whose outcomes end in July can be admitted. Merely
checking that a training feature date precedes d would be insufficient.

Fitting and prediction index only the five allowed features. `target`,
`target_date`, `future_return`, and benchmark evaluation returns never enter X.
This addresses implementation leakage; revisable adjusted data and the ex-post
universe remain separate sources of bias.

## Selection and construction

3M momentum ranks `P[t]/P[t−3]−1`. The 12–1 signal ranks
`P[t−1]/P[t−12]−1`. Both select the five largest valid scores. ML selects the
five highest probabilities, with ticker ascending as its explicit tie-break.
Momentum uses the existing pandas score sort; no tie rule was changed.

The three equal-weight strategies allocate 1/N. The momentum optimizer variants
use the same 3M selection and the last 252 daily **price observations** through
t (thus typically 251 daily returns). PyPortfolioOpt supplies annualized sample
covariance; max Sharpe also uses compounded historical mean returns and a zero
risk-free rate. Portfolios are long-only, fully invested, with a 40% position
cap in the momentum construction path. Five-stock ML equal weights are 20%;
the generic score engine does not itself enforce a configurable position cap.

The optimizer waits for the full price window. Insufficient history, missing
historical cells, an expected solver failure, or an incomplete returned universe
produce explicit equal-weight fallback statuses. Feasibility is checked;
projection onto a capped simplex and final validation keep the cap binding.
Zero-weight names can remain in the selected holdings list; effective holdings
`1 / sum(w²)` conveys concentration better than its count.

Selection precedes realized-return inspection. A selected missing outcome
raises rather than replacing the security. The score engine skips entirely
unlabeled terminal cross-sections; it rejects partially missing selected
outcomes. No delisting return is inferred from a missing value.

## Turnover and costs

Pre-trade weights after a holding period are:

`w_pre[i] = w_old[i] × (1 + r[i]) / sum_j(w_old[j] × (1 + r[j]))`.

One-way turnover is `T = 0.5 × sum_i |w_target[i] − w_pre[i]|`, over the union
of old and new names. Initial deployment is explicitly T = 1. Monthly net
return is `sum_i w_target[i] × r_next[i] − T × 10 / 10,000`.

This linear cost deduction is a transparent research convention. It does not
model execution cash flows, variable spreads, volume, impact, or capacity.
Rebalancing to identical target weights can still incur turnover after drift.

## Evaluation

All five strategy net-return series and benchmark are inner-joined on their
realization dates. The common period has 88 months, September 2018–December
2025. Earlier standalone strategy runs have different startup periods; compare
strategies using the common-period table. Slicing to the common period does not
reset an older strategy's portfolio or charge it a second initial deployment.

CAGR is `product(1+r)^(12/n)−1`; annual volatility uses sample standard deviation
× √12. Sharpe uses arithmetic mean excess return / sample standard deviation
× √12, with risk-free rate zero. Sortino uses the square root of the full-sample
mean squared negative excess return. Drawdown includes initial wealth of one.
Tracking error is annualized sample standard deviation of active monthly
returns; information ratio uses mean active return / its standard deviation
× √12. Annual alpha is the arithmetic regression intercept × 12. Short or
degenerate samples yield unavailable values where the metrics require them.

Rank IC is monthly Spearman(score, next-period stock return). IC IR is
mean(IC)/std(IC), **unannualized**. Top-N hit rate counts selected names beating
the benchmark; Top-N spread subtracts the full scored universe's mean return.
Yearly ranking groups use signal years; performance tables use realization
years. Their partial-year counts therefore differ.

## Prospective and provider boundaries

Legacy live/replay truncates price and benchmark panels once at the explicit
cutoff and resolves their latest shared market observation. Monthly sampling
can include the last observed row of a partial current month: it is an as-of
research snapshot, not necessarily a completed calendar-month signal. Live
feature output excludes target/evaluation columns, and training uses the same
strict embargo. Eligibility and staleness are reported.

Legacy snapshots are copied in memory; saved metadata includes input hashes,
not a complete persisted price panel. Reproduction needs the preserved input
files and environment. Run directories are exclusive, paper recommendations
are immutable, and evaluation uses frozen weights. Replay uses its own output
and paper namespace. Historical contextual report tables can be loaded by the
legacy runner; they are not as-of-safe evidence, and do not feed predictions
or portfolio construction.

The institutional provider adds dated immutable membership snapshots,
canonical/vendor identifiers, raw and normalized cache versions, coverage
checks, current quantiles, and sector diagnostics. Dated membership uses
`effective_from <= signal_date < effective_to`, with an open upper end allowed.
Absent genuine history raises `POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE`.
The provider's verification flag means supplied dated records passed its
contract; it does not independently certify the vendor's historical accuracy.
Current constituents plus old prices do not constitute historical membership.

## Reproducibility boundary

Historical CLI manifests record frozen configuration, model defaults, input
file hashes, source hashes (including uncommitted Python), package versions,
completion/failure status, and hashes of reports written by that command.
A source hash supplements Git identity; a parent directory's repository is not
accepted as this project's commit. Reports keep their stable filenames and can
be overwritten on rerun; manifests identify the bytes from each execution but
do not archive every report version. Preserve a report set with its manifest
when sharing a result. See [reproduction](REPRODUCIBILITY.md).
