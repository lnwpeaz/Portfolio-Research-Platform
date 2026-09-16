"""Manifest and per-security acquisition metadata."""

from datetime import datetime, timezone
import hashlib
import json
import pandas as pd


def panel_fingerprint(panel):
    ordered = panel.sort_values(["canonical_ticker", "date"]).reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(pd.util.hash_pandas_object(ordered, index=False).values.tobytes())
    return digest.hexdigest()


def acquisition_status(panel, requested, mapping, resolved_date=None, failures=None):
    failures = failures or {}
    expected_dates = panel.date.nunique() if not panel.empty else 0
    rows = []
    for row in mapping.itertuples():
        group = panel[panel.canonical_ticker.eq(row.canonical_ticker)]
        observed = group.adjusted_close.notna().sum() if not group.empty else 0
        expected = expected_dates
        last = group.loc[group.adjusted_close.notna(), "date"].max() if observed else pd.NaT
        status = "failed" if observed == 0 else (
            "partial" if resolved_date is not None and pd.Timestamp(last) < pd.Timestamp(resolved_date) else "successful"
        )
        rows.append({
            "canonical_ticker": row.canonical_ticker,
            "provider_ticker": row.provider_ticker,
            "first_available_date": group.date.min() if not group.empty else pd.NaT,
            "last_available_date": last,
            "observation_count": int(observed),
            "missing_ratio": float(1 - observed / expected) if expected else 1.0,
            "status": status,
            "failure_reason": failures.get(row.canonical_ticker) if status == "failed" else None,
        })
    return pd.DataFrame(rows)


def manifest_id(payload):
    stable = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(stable.encode()).hexdigest()[:16]


def utc_now():
    return datetime.now(timezone.utc).isoformat()
