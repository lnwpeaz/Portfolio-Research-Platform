# Cross-sectional equity ranking: research report

## Research question

How do simple momentum ranking, constrained portfolio optimization, and a
walk-forward classifier behave under a shared monthly evaluation protocol?
Cross-sectional ranking allocates a limited portfolio among available stocks;
it lets us study signal quality, concentration, turnover, and implementation
sensitivity together. The project emphasizes auditable research engineering.

**All historical results are illustrative, conditional on the available ex-post
20-stock universe. They are not an unbiased S&P 500 backtest or evidence of
investable alpha.** No model replacement, parameter tuning, or best-scenario
selection was performed during this portfolio update.

## Dataset and universe

The frozen inputs span 2015-01-02–2025-12-31, with 2,766 daily rows and 20
surviving large-cap securities. Adjusted stock prices incorporate distributions;
`^GSPC` benchmark prices do not. No inactive-security history or delisting return
series is available. [Input hashes](../data/BUNDLED_DATA.json) identify the
actual files; they do not establish that the vendor history was point-in-time.

The universe abstraction supports fixed20, current S&P 500, and a future SET50
provider. External dated membership can be filtered with half-open effective
intervals. The repository has no genuine historical S&P constituent dataset.
Current S&P scoring is a separate prospective workflow; contemporary members'
past prices must not be relabeled as historical index membership.

## Features, signals and ML

Momentum baselines use three months or the 12–1 definition that skips the
latest month. ML uses 1M/3M/6M momentum, three-month annualized volatility, and
three-month drawdown. A HistGradientBoostingClassifier predicts the probability
of beating the next-month price benchmark. Its settings remain 200 iterations,
0.05 learning rate, depth 3 and seed 42.

A fresh expanding-window model requires 36 available labeled feature months.
Training labels must end **strictly before** prediction. The feature allow-list
and explicit target dates separate information available at signal formation
from later evaluation. [Methodology](METHODOLOGY.md) gives formulas and a
month-by-month embargo example.

## Portfolio, backtest protocol and costs

All five official portfolios select five names monthly. 3M, 12–1 and ML use
equal weights. The other two use the 3M selection with maximum-Sharpe or
minimum-volatility optimization over 252 daily price observations. The momentum
construction path enforces long-only weights summing to one and a 40% cap.
Optimizer problems produce recorded equal-weight fallbacks.

Signals are formed at month-end observed closes and assumed executable at the
same close. Returns are stored at the following realization date. Selection
never consults next-month availability; missing selected outcomes fail rather
than altering the portfolio. One-way turnover compares target weights with
drifted pre-trade holdings. Costs are 10 bps times turnover, including initial
deployment turnover of one. All results below are net of this modeled cost.

## Evaluation and results

The five strategies and benchmark share 88 monthly realizations from
2018-09-28 through 2025-12-31. The first calendar year is partial. CAGR,
volatility, Sharpe, Sortino, drawdown, Calmar, active return, tracking error,
information ratio, beta and alpha are all retained in the
[complete metric table](../reports/common_period_performance.csv), with
[calendar-year detail](../reports/common_period_yearly_returns.csv).

| Portfolio | CAGR | Volatility | Sharpe | Sortino | Max drawdown | Tracking error | Information ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3M equal | 34.04% | 24.23% | 1.338 | 2.879 | −20.57% | 17.33% | 1.110 |
| 12–1 equal | 30.32% | 24.49% | 1.209 | 2.583 | −20.76% | 15.25% | 1.077 |
| 3M max Sharpe | 33.33% | 28.79% | 1.141 | 2.815 | −24.03% | 21.75% | 0.904 |
| 3M min volatility | 20.13% | 18.06% | 1.110 | 2.242 | −14.56% | 13.03% | 0.527 |
| ML equal | 30.83% | 22.99% | 1.291 | 2.735 | −19.86% | 13.24% | 1.246 |
| S&P 500 price index | 12.42% | 16.93% | 0.779 | 1.205 | −24.77% | N/A | N/A |

These strong numbers are conditional on a small selected set of survivors.
They cannot establish that momentum or ML would outperform in an investable
historical universe. Active metrics are additionally distorted by missing
benchmark dividends. Optimization does not dominate equal weighting in this
sample; its concentration and estimator sensitivity warrant examination.
[Turnover by month](../reports/common_period_turnover.csv) provides the trading
intensity behind the fixed-cost assumption.

## Ranking diagnostics

Fixed-score ML mean Rank IC is 0.0386, 0.0332, and 0.0331 over cumulative one-,
two-, and three-month horizons. The one-month IC IR is approximately 0.168,
unannualized; monthly IC is positive in about 51.1% of evaluated periods.
Top-5 benchmark-outperformance hit rate is about 53.2%. These modest and variable
ranking statistics provide useful context alongside the portfolio returns.
Yearly mean IC ranges from −0.109 in partial 2018 to 0.120 in 2024.
See [decay](../reports/ml_signal_decay.csv) and
[yearly stability](../reports/ml_ranking_yearly_stability.csv).

Institutional diagnostics additionally support equal-count score quantiles,
Q5−Q1, coverage, fixed Top-N sensitivity, and equal-weight sector exposure.
Current quantiles describe today's ranking distribution only. The legacy fixed20
provider has no sector metadata, so its sector output is entirely `Unknown` and
cannot support an economic sector interpretation. Historical
quantile returns require realized data, and are not a separate official strategy.

## Robustness tests and interpretation

| Diagnostic | Question it challenges |
|---|---|
| Leave-one-out reruns | Does a particular universe member dominate 3M, 12–1 or ML results? |
| Fixed calendar subperiods | Are results concentrated in a particular period? These are not fitted economic regimes. |
| Costs from 0 to 100 bps | How sensitive are net outcomes to trading intensity? |
| One-period execution delay | How dependent are results on immediate implementation? |
| Paired circular moving-block bootstrap | How variable are metrics when local return dependence is preserved? |
| Contributor concentration and reruns | How much depends on the leading one, two or three contributors? |
| Extreme-month removal | How much comes from a small set of realized best months? |
| Optimizer input/weight diagnostics | Are covariance estimates or allocations unstable/concentrated? |
| Fixed-score ML decay and yearly IC | Does predictive ranking persist across horizons and years? |
| Rolling 24-month windows | Are conclusions stable beyond coarse subperiod boundaries? |

Default bootstrap uses 5,000 simulations and six-month blocks; three- and
twelve-month blocks are separately reported, never selected as an official
setting. Fifth-percentile Sharpe estimates are 0.658–0.745 across strategies.
This measures conditional sampling stability, not survivorship-free significance.

The leading security supplies 16.6%–31.0% of positive arithmetic contribution.
Removing the best three months leaves Sharpe values of roughly 0.895–1.168.
Contributor subtraction and best-month removal are explicitly **non-investable
counterfactuals**. Even rerunning after selecting contributors ex post remains
a sensitivity study, not an implementable selection rule.

Across 65 complete rolling 24-month windows, ML's active return is positive in
all windows; minimum volatility's is positive in 67.7%. Overlapping windows
are dependent, and these fractions inherit the data biases. The
[scorecard](../reports/robustness_scorecard.csv) retains separate diagnostics
with missing values where unavailable; there is no composite robustness score.

## Prospective architecture

The live runner freezes the as-of data view, computes features and scores,
constructs target weights, and writes an exclusive report directory. Paper
recommendations retain their weights; later evaluation does not regenerate
past rankings. Historical replay reuses the same cutoff runner in an isolated
namespace. Legacy snapshots save hashes/metadata; institutional acquisition
also saves versioned raw/normalized panels. Neither is a brokerage interface.

**Research recommendation ≠ trade execution ≠ expected profit.** A replay is a
simulation of the prospective process, not a recommendation actually issued in
the historical period. Current successful-member filtering still limits replay.

## Limitations and future work

The required external improvements are genuine dated membership, identifier
history, inactive securities, delisting returns, and a consistent total-return
benchmark. Architecture alone cannot supply them. Yahoo revisions and
corporate-action handling remain provenance risks; saved versions improve
reproduction without making the original source point-in-time.

Execution remains same-close, with linear costs and no liquidity/capacity
model. Optional later research includes next-session execution with suitable
prices, factor/style attribution, formal regime analysis, tail-risk summaries,
and richer live snapshot persistence. These are future extensions, not
conditions concealed by selecting a more favorable historical result.
