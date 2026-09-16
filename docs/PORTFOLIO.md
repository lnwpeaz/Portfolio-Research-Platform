# Portfolio and interview guide

A Python quantitative research project comparing five monthly equity portfolios
under shared timing, cost and evaluation rules. It connects momentum and ML
rankings to constrained allocations, robustness diagnostics and prospective
paper recommendations, with explicit controls against implementation leakage.

**Stack:** Python, pandas, NumPy, scikit-learn, SciPy, PyPortfolioOpt/CVXPY,
Matplotlib, Parquet/PyArrow, pytest; Yahoo research-data adapters.

**Methods:** 3M and 12–1 momentum; expanding-window gradient-boosted classification;
maximum-Sharpe and minimum-variance optimization; Rank IC, active risk, rolling
analysis, paired moving-block bootstrap, and cost/execution sensitivity.

**Engineering:** explicit feature allow-list and label embargo; separate signal
and realization dates; capped weights and visible fallbacks; drift-aware costs;
input/source provenance; cutoff replay and immutable paper targets; 69 tests.

**Limits:** ex-post 20-stock survivors, absent historical PIT membership and
delistings, stock-adjusted/price-index benchmark mismatch, revisable data,
same-close execution and constant costs. This is research engineering, not
proof of profitable alpha or a production trading system.

## Resume bullet options

- Built a Python equity research pipeline comparing five momentum, optimized
  and ML portfolios with walk-forward evaluation, constrained weights and
  drift-aware transaction costs.
- Implemented label-availability embargoes, deterministic as-of replay and
  immutable paper targets, supported by 69 regression and contract tests.
- Developed ranking, active-risk and bootstrap diagnostics with reproducible
  data/source manifests and explicit survivorship and benchmark limitations.

## 30-second explanation

“I built an equity research pipeline that compares momentum, portfolio
optimization and machine-learning rankings under one monthly protocol. The
engineering focus is making the decisions auditable: labels must be available
before training, selection cannot depend on future returns, and costs use
drifted holdings. It also supports deterministic replay and frozen paper
recommendations. The historical sample contains only 20 ex-post survivors, so
I present its performance as illustrative and emphasize the research controls.”

## 2-minute explanation

“The question is how different equity-ranking and allocation methods behave
under the same evaluation rules. I compare five portfolios: equal-weight 3M
and 12–1 momentum, two 3M portfolios using constrained optimization, and an
ML-ranked equal-weight portfolio. All select five names monthly. The optimizers
use 252 daily price observations, long-only weights and a 40% cap.

“The classifier uses five momentum, volatility and drawdown features to predict
next-month benchmark outperformance. Training expands through time, but a
label is included only if its target date is strictly earlier than prediction.
That excludes labels ending exactly at the prediction cutoff. An explicit
allow-list keeps future outcomes out of the model inputs.

“I separate signal dates from realization dates and select the portfolio before
checking outcomes. Missing selected returns raise an error instead of replacing
the stock. Turnover compares targets with drifted holdings, and net returns
subtract 10 basis points times one-way turnover. Optimization failures remain
visible rather than disappearing into silent fallback behavior.

“I evaluate returns alongside Rank IC, concentration, active risk, bootstrap
stability and cost/execution sensitivity. The prospective layer freezes paper
targets and evaluates those weights later. Tests and manifests make the work
inspectable and reproducible. The important qualification is that the historical
universe is ex post, delistings are missing, and the price benchmark excludes
dividends. The project demonstrates research engineering; it does not establish
unbiased investment performance.”

## Five likely technical questions

1. **Why is `feature_date < prediction_date` insufficient?** A feature row's
   label may end at or after prediction. Require `target_date < prediction_date`
   and count the 36-month minimum using available labeled feature months.
2. **How are turnover and trading costs calculated?** Drift old weights using
   realized asset returns, normalize, then take half the L1 distance to targets
   over all old/new names. Initial turnover is one; cost is turnover × 0.001.
3. **How do constraints survive optimizer failures?** Full history is required;
   missing inputs or expected solver failures produce labeled equal weights.
   Feasibility checks, capped-simplex projection and validation enforce the cap.
   The generic ML score engine uses equal weights, not a configurable cap.
4. **What does a positive bootstrap result establish?** Sampling stability
   conditional on these data, preserving local dependence and paired benchmark
   observations. It cannot remove universe selection bias or establish alpha.
5. **What would be required before prospective deployment?** Genuine PIT and
   delisting-aware total-return data, consistent benchmark returns, execution
   prices and realistic liquidity costs. Legacy live snapshots save metadata,
   not full price panels; historical context in replay reports is separately
   loaded and must not be treated as as-of evidence. No brokerage execution exists.

For exact formulas see [methodology](METHODOLOGY.md); for executed commands and
reproducibility evidence see [validation](VALIDATION.md#github-hardening-pass).
