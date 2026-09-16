# Operations reference

Preserved detailed CLI reference from the earlier README. For the validated
bundled-data sequence, environment and current caveats, use
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Network-dependent institutional
examples below were not rerun during the portfolio update.

# Portfolio Research Pipeline

Quantitative research project for building systematic investment strategies using machine learning, portfolio optimization, and backtesting.

The code implements monthly cross-sectional momentum, constrained portfolio
optimization, expanding-window ML ranking, transaction costs, and historical
backtesting.

Timing convention: features and signals are formed from month-end adjusted
closes, the portfolio return is measured to the next month end, and backtest
rows are indexed by that realization date. The implementation assumes trading
at the signal close. This is configured explicitly as
`EXECUTION_TIMING_MODE = "same_close"`; unsupported modes fail rather than
silently changing results. A live implementation must either compute the signal
before the market-on-close cutoff or add execution prices, a consistent lag,
and slippage to both the momentum and ML paths.

Transaction cost convention: turnover is one-way turnover and the configured
cost is `ONE_WAY_TURNOVER_COST_BPS`. Net returns use:

`net_return = gross_return - one_way_turnover * cost_bps / 10,000`

Run `python comparison_main.py` to build, print, and save the performance and
calendar-year tables for momentum equal weight, momentum max Sharpe, momentum
minimum volatility, conventional 12-1 momentum, ML, and the benchmark over
their identical common realization dates. CSV outputs are written under
`reports/`.

Run `python research_main.py` for the extended research phase. It adds rolling
risk, concentration, drawdown episodes, security contribution attribution,
calendar-year leadership/excess returns, ML Rank IC time series, and static
matplotlib research figures. The 12-1 signal is defined as
`price[t-1] / price[t-12] - 1`, explicitly excluding the most recent month.

Run `python robustness_main.py` for the separate Phase 1 robustness suite. It
re-runs 3M, 12-1, and ML strategies after removing each universe member, reports
fixed-calendar subperiod stability, applies turnover-based cost scenarios from
0 to 100 bps, and compares official same-close results with a one-month delayed
weight implementation. Outputs are written under `reports/`; none of these
experiments modifies the official configuration or strategy reports.

Research limitation: the bundled 20-stock universe is a fixed ex-post list and
is not suitable for unbiased performance claims. Production research requires
point-in-time membership, delisted securities and delisting returns. The
backtest deliberately raises on a partially missing selected return rather than
silently changing the portfolio using future availability.

## Statistical Robustness and Stability

Run `python robustness_phase2_main.py` for paired moving-block bootstrap,
security contribution concentration, return-outlier, optimizer stability, ML
ranking/decay, and rolling 24-month diagnostics. The bootstrap produces
sampling-stability intervals; it cannot remove the fixed-universe survivorship
bias embedded in every observation. Contribution concentration shows whether a
small number of names dominate arithmetic attribution, while removal of the
best months is explicitly a non-investable statistical counterfactual.

Optimizer diagnostics inspect recorded weights and the unchanged covariance
and expected-return inputs for concentration and numerical instability. ML
diagnostics reuse the original scores to measure yearly Rank IC, rolling
stability, and signal decay at longer forward horizons; they do not retrain or
create a new rule. Rolling windows complement coarse subperiods by showing how
consistently results survive through time. These diagnostics challenge the
official results but do not tune or replace any strategy.

## Prospective Research Mode

`python live_main.py` creates an immutable as-of research run under
`reports/live/<run_id>/`. If no date is supplied, the cutoff is the latest valid
market observation in the bundled dataset—not the local calendar date.

```text
python live_main.py
python live_main.py --as-of 2025-12-31
python live_main.py --as-of 2025-12-31 --no-paper-update
python live_replay_main.py --as-of 2024-12-31
python live_replay_main.py --start 2024-01-01 --end 2025-12-31 --frequency monthly
```

Prospective mode snapshots prices and benchmark data at the cutoff, emits
features without target/evaluation columns, trains the unchanged ML model only
on labels realized strictly before prediction, and uses official portfolio
settings. Each run records its input hash, manifest, rankings, portfolios,
consensus, risk flags, and research warnings.

Paper recommendations are frozen under `paper/runs/`. Later observations
evaluate those stored weights without recalculating the old signal. Reference
prices are not represented as executable fills, and transaction costs remain
separate. This provides infrastructure to record prospective research decisions; it does not place
trades or repair fixed-universe, delisting, vendor, benchmark, or historical
same-close limitations.

## Institutional Universe Research

`institutional_main.py` adds a provider boundary between strategy code and the
security universe. `fixed20` wraps the original ex-post list unchanged,
`sp500` supports a published current constituent table or externally supplied
dated history, and `set50` implements the same contract but remains data-blocked.
No provider reconstructs old constituents from today's list.

```text
python institutional_main.py --universe fixed20
python institutional_main.py --universe sp500
python institutional_main.py --universe sp500 --current-membership data/universe/sp500_current.csv --prices data/cache/sp500_prices.parquet
python institutional_main.py --universe sp500 --start 2018-01-01 --end 2025-12-31 --historical-membership /path/to/dated_membership.csv --prices /path/to/pit_prices.parquet
```

The default large-universe portfolio size is 25; fixed20 retains 5. The
10/20/25/50 comparison is a fixed sensitivity diagnostic and is never used to
select a winner. The model class, features, target, hyperparameters, label
embargo, and walk-forward method are unchanged.

Historical membership input uses these columns:

```text
index_name,ticker,company_name,sector,industry,effective_from,effective_to,source,source_date
```

Intervals are half-open: a member is active when
`effective_from <= signal_date < effective_to`; a null `effective_to` remains
active. Current constituent data are prospective only. Asking for history
without dated records stops with `POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE`.
Architecture readiness does not mean survivorship bias has been resolved.

Membership is followed by explicit price-history, current-price, staleness,
lookback, and feature-availability screening. Missing and partial coverage is
reported instead of silently removing securities. Batched downloads, when
explicitly used through `src.data.market_data`, write companion metadata with
requested/successful/failed names and an input hash; a partial cache is never
labeled complete.

Historical/PIT outputs under `reports/institutional/` include cross-sectional
IC, feature coverage, ticker missingness, equal-count score quantiles, the
research-only Q5-Q1 spread, sector exposure/attribution, fixed Top-N sensitivity,
a run manifest, and compact figures. Current-only output is written under
`reports/institutional/current/` and does not place trades or update the paper
ledger. Sector and universe weights are explicitly equal weight. The `^GSPC`
benchmark remains a price index and is inconsistent with dividend-adjusted
security prices; the abstraction allows a later total-return series without
changing strategy code.

### Institutional market-data foundation

Current S&P 500 research uses a vendor-neutral provider contract under
`src/data/institutional/`. Yahoo is the default research adapter and retrieves
adjusted close, raw close, volume, dividends, and stock splits in batches with
retry handling. Yahoo history is revisable and is not an institutional
point-in-time source; every manifest carries that warning.

```text
python institutional_data_main.py --universe sp500
python institutional_data_main.py --universe sp500 --refresh
python institutional_data_main.py --universe sp500 --start 2021-01-01
python institutional_live_main.py --universe sp500
python institutional_live_main.py --universe sp500 --paper-update
```

The default window is five years plus a 31-day warm-up buffer. It covers the
unchanged ML features, 12-1 momentum, 252-day optimizer/readiness requirement,
and 36-month walk-forward training minimum. Longer windows are an acquisition
choice, not a model parameter.

Canonical index symbols are retained in reports while vendor symbols are stored
separately (`BRK.B` maps to Yahoo `BRK-B`). Raw responses and normalized panels
are versioned immutably under `data/institutional/`; manifests identify each
version and incremental requests merge new observations without deleting old
ones. Re-running an already-covered request reuses the cache. `--refresh`
requests a small recent overlap so vendor revisions remain versioned rather
than overwriting old raw data.

The market date is the latest date on or before the requested calendar date
where at least 95% of current constituents have valid adjusted closes. This
prevents weekends, holidays, and isolated late prints from creating false stale
flags. Invalid prices/volume, duplicate or future dates, extreme returns,
reported actions, and unexplained adjusted/raw changes are reported without
silent deletion or winsorization.

`institutional_live_main.py` uses the cached panel, unchanged features, strict
label embargo, unchanged model and hyperparameters, unchanged momentum
definitions, and a documented Top 25 equal-weight research target. Current
quantiles are ranking snapshots only; no future-return performance is computed.
Reports are written both to an immutable run directory and the current view.
Paper state is unchanged unless `--paper-update` is explicit, and institutional
decisions use `paper/sp500/` rather than the legacy fixed20 ledger.

This is current-constituent prospective research. It does not provide dated
historical membership, delisting returns, or an unbiased historical S&P 500
backtest.
