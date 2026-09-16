"""Non-destructive market-panel validation."""

import numpy as np
import pandas as pd


def validate_market_panel(panel, requested_date=None, extreme_return_threshold=0.50):
    flags = []
    if panel.empty:
        return pd.DataFrame(columns=["canonical_ticker", "date", "flag", "value", "severity"])
    for ticker, group in panel.groupby("canonical_ticker", sort=True):
        dates = pd.to_datetime(group.date)
        if dates.duplicated().any():
            for date in dates[dates.duplicated(keep=False)].unique():
                flags.append((ticker, date, "DUPLICATE_DATE", np.nan, "error"))
        if not dates.is_monotonic_increasing:
            flags.append((ticker, pd.NaT, "NON_MONOTONIC_DATES", np.nan, "error"))
        for column in ("adjusted_close", "raw_close"):
            values = pd.to_numeric(group[column], errors="coerce")
            invalid = values.notna() & (~np.isfinite(values) | values.le(0))
            for index in group.index[invalid]:
                flags.append((ticker, group.loc[index, "date"], f"INVALID_{column.upper()}", group.loc[index, column], "error"))
        volume = pd.to_numeric(group.volume, errors="coerce")
        invalid_volume = volume.notna() & (~np.isfinite(volume) | volume.lt(0))
        for index in group.index[invalid_volume]:
            flags.append((ticker, group.loc[index, "date"], "INVALID_VOLUME", group.loc[index, "volume"], "error"))
        returns = group.set_index("date").adjusted_close.pct_change(fill_method=None)
        for date, value in returns[returns.abs().gt(extreme_return_threshold)].items():
            flags.append((ticker, date, "EXTREME_ADJUSTED_RETURN", value, "warning"))
        if requested_date is not None and dates.gt(pd.Timestamp(requested_date)).any():
            for date in dates[dates.gt(pd.Timestamp(requested_date))]:
                flags.append((ticker, date, "FUTURE_OBSERVATION", np.nan, "error"))
    return pd.DataFrame(flags, columns=["canonical_ticker", "date", "flag", "value", "severity"])


def security_readiness(panel, tickers, resolved_date, minimum_observations=252, stale_calendar_days=3):
    resolved = pd.Timestamp(resolved_date)
    rows = []
    for ticker in tickers:
        group = panel[panel.canonical_ticker.eq(ticker)]
        valid = group[
            group.date.le(resolved) & group.adjusted_close.notna()
            & np.isfinite(group.adjusted_close) & group.adjusted_close.gt(0)
        ]
        observations = len(valid)
        latest = valid.date.max() if observations else pd.NaT
        staleness = (resolved - latest).days if observations else pd.NA
        reasons = []
        if observations == 0:
            reasons.append("MISSING_PRICE")
        elif observations < minimum_observations:
            reasons.append("INSUFFICIENT_HISTORY")
        if observations and staleness > stale_calendar_days:
            reasons.append("STALE_PRICE")
        if observations and not valid.date.eq(resolved).any():
            reasons.append("CURRENT_PRICE_UNAVAILABLE")
        rows.append({
            "ticker": ticker, "price_ready": not reasons,
            "latest_price_date": latest, "history_observations": observations,
            "staleness_days": staleness,
            "reason": "|".join(reasons) if reasons else None,
        })
    return pd.DataFrame(rows)
