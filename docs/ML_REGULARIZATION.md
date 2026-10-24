# ML regularization experiment

## Protocol fixed before execution

This is a bounded extension of the existing five-feature classifier, not a
replacement for the v1.0 research results. The objective is validation Sharpe
after the existing 10 bps one-way-turnover cost. No promise is made that alpha,
drawdown and volatility improve together.

The following candidates are fixed before running the experiment:

| Candidate | Maximum tree depth | L2 regularization |
|---|---:|---:|
| baseline | 3 | 0 |
| depth2_l2_1 | 2 | 1 |
| depth2_l2_10 | 2 | 10 |
| depth3_l2_1 | 3 | 1 |
| depth3_l2_10 | 3 | 10 |

Every candidate retains 200 iterations, learning rate 0.05, random seed 42,
the existing five features, at least 36 available labeled feature months,
and the strict `target_date < prediction_date` embargo. Models are refitted
monthly on expanding histories. There is no random train/test split.

Selection uses **realization dates in 2021–2022**, on identical months for
all candidates and the benchmark. Highest finite net Sharpe wins, including
the baseline; exact ties prefer the candidate listed first. Data used to
construct validation features and labels stop at 2022-12-31. The selected
configuration is written before evaluation predictions are computed.

Evaluation uses **realization dates in 2023–2025**, with the selected
hyperparameters fixed. Expanding refits may use labels from earlier evaluation
months once they become available under the same embargo. This is not
reselection. Every candidate is reported, and a worse evaluation result is
retained. Evaluation results must not drive a second search in this study.

For each segment, all portfolios start in cash at its first signal and incur
initial deployment turnover of one. Signals are at the preceding month's
last observed close. Monthly Top-5 equal weights, same-close execution and
the 10 bps cost convention are unchanged. Segment drawdowns start from one.
These cash-reset segments are not the official continuous-history results.

Report CAGR, annual alpha (the existing single-benchmark regression intercept
annualized at zero risk-free rate), Sharpe, annual volatility, maximum drawdown,
turnover and cost totals. Alpha is not CAGR minus benchmark CAGR and is not
factor-adjusted alpha.

The 2023–2025 historical results have already been inspected during earlier
research. Evaluation is a chronological retrospective check, **not an unseen
holdout**. The ex-post 20-stock universe, absent delistings, adjusted-stock
versus price-index benchmark mismatch, and same-close assumptions remain.
Future unseen data would be required for a clean prospective assessment.

Outputs belong only to a new experiment run directory. Preserve its protocol,
configuration, input/source hashes, selection record and reports together.
