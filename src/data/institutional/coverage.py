"""Broad-market date resolution and per-security readiness."""

import math
import pandas as pd


def resolve_common_market_date(panel, tickers, requested_date, minimum_coverage=0.95):
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1].")
    requested = pd.Timestamp(requested_date).normalize()
    names = tuple(dict.fromkeys(tickers))
    relevant = panel[
        panel.canonical_ticker.isin(names) & panel.date.le(requested)
        & panel.adjusted_close.notna() & panel.adjusted_close.gt(0)
    ]
    if relevant.empty:
        raise ValueError("No valid market observations exist by the requested date.")
    coverage = relevant.groupby("date").canonical_ticker.nunique().sort_index() / len(names)
    valid = coverage[coverage.ge(minimum_coverage)]
    if valid.empty:
        best = coverage.max()
        raise ValueError(
            f"No date meets {minimum_coverage:.1%} coverage; best available is {best:.1%}."
        )
    resolved = pd.Timestamp(valid.index[-1])
    return {
        "requested_date": requested,
        "resolved_market_date": resolved,
        "coverage_on_resolved_date": float(coverage.loc[resolved]),
        "covered_securities": int(round(coverage.loc[resolved] * len(names))),
        "required_securities": math.ceil(minimum_coverage * len(names)),
    }
