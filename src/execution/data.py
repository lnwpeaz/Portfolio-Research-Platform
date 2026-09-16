"""Read existing prices/liquidity without downloads or synthetic observations."""
from pathlib import Path
import numpy as np
import pandas as pd


def load_execution_prices(path):
    frame = pd.read_parquet(path)
    if not isinstance(frame.columns, pd.MultiIndex) or not {"Open", "Close"}.issubset(frame.columns.get_level_values(0)):
        raise ValueError("EXECUTION_PRICES_UNAVAILABLE: adjusted Open and Close required")
    close, opening = frame["Close"].sort_index(), frame["Open"].sort_index()
    if close.index.has_duplicates or close.columns.has_duplicates:
        raise ValueError("Duplicate execution price dates or tickers")
    return close, opening.reindex(index=close.index, columns=close.columns)


def resolve_liquidity_path(path=None):
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise ValueError("LIQUIDITY_DATA_UNAVAILABLE: normalized provider Parquet required")
        return candidate
    candidates = sorted(Path("data/institutional/normalized").glob("*.parquet"))
    if len(candidates) != 1:
        raise ValueError("LIQUIDITY_DATA_UNAVAILABLE_OR_AMBIGUOUS: supply --liquidity-panel; no download or adjusted-price substitute is performed")
    return candidates[0]


def liquidity_panels(path, dates, tickers):
    panel = pd.read_parquet(path)
    required = {"date", "canonical_ticker", "raw_close", "volume"}
    if not required.issubset(panel):
        raise ValueError("LIQUIDITY_DATA_UNAVAILABLE: require date, canonical_ticker, raw_close, volume")
    panel = panel.loc[panel.canonical_ticker.isin(tickers)].copy()
    panel["date"] = pd.to_datetime(panel.date)
    if panel.duplicated(["date", "canonical_ticker"]).any():
        raise ValueError("Duplicate liquidity date/ticker observations")
    def pivot(column):
        return panel.pivot(index="date", columns="canonical_ticker", values=column).reindex(index=dates, columns=tickers).astype(float)
    return pivot("raw_close"), pivot("volume")


def calculate_adv(raw_close, volume, window=20):
    """Full-window raw-close × volume proxy, ending strictly before row date.

    Missing/nonpositive/nonfinite observations invalidate the full window.
    Reindex on the execution-session calendar before calling; no gap filling.
    """
    if window < 1 or not isinstance(window, int):
        raise ValueError("ADV window must be a positive integer")
    if not raw_close.index.equals(volume.index) or not raw_close.columns.equals(volume.columns):
        raise ValueError("Price/volume panels must align")
    if raw_close.index.has_duplicates or not raw_close.index.is_monotonic_increasing:
        raise ValueError("ADV dates must be unique and sorted")
    valid = np.isfinite(raw_close) & np.isfinite(volume) & raw_close.gt(0) & volume.gt(0)
    dollars = (raw_close * volume).where(valid)
    return dollars.rolling(window, min_periods=window).mean().shift(1)


def next_session(calendar, signal_date):
    """Next observed market session, never the same session or ticker-specific fill."""
    calendar = pd.DatetimeIndex(calendar)
    if calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("Execution calendar must be unique and sorted")
    position = calendar.searchsorted(pd.Timestamp(signal_date), side="right")
    return calendar[position] if position < len(calendar) else None
