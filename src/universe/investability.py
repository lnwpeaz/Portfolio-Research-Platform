"""Transparent investability rules applied after index membership."""

from dataclasses import dataclass
import pandas as pd

from src.features.ml_dataset import FEATURE_COLUMNS
from src.universe.models import UniverseSnapshot


@dataclass(frozen=True)
class InvestabilityConfig:
    minimum_history_observations: int = 126
    stale_calendar_days: int = 7
    required_features: tuple[str, ...] = tuple(FEATURE_COLUMNS)


@dataclass(frozen=True)
class InvestabilityResult:
    eligible_tickers: tuple[str, ...]
    rejected_tickers: tuple[str, ...]
    report: pd.DataFrame


def _latest_features(features: pd.DataFrame | None, ticker: str, as_of_date):
    if features is None or features.empty:
        return None
    frame = features
    if "ticker" in frame.columns:
        frame = frame[frame.ticker.eq(ticker)]
    elif ticker in frame.index:
        row = frame.loc[ticker]
        return row.iloc[-1] if isinstance(row, pd.DataFrame) else row
    else:
        return None
    date_column = "date" if "date" in frame.columns else "signal_date" if "signal_date" in frame.columns else None
    if date_column:
        frame = frame[pd.to_datetime(frame[date_column]).le(pd.Timestamp(as_of_date))]
        frame = frame.sort_values(date_column)
    return None if frame.empty else frame.iloc[-1]


def screen_investability(
    snapshot: UniverseSnapshot,
    prices: pd.DataFrame,
    features: pd.DataFrame | None = None,
    config: InvestabilityConfig | None = None,
) -> InvestabilityResult:
    """Screen members at the snapshot date without mutating provider output."""
    config = config or InvestabilityConfig()
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Prices must use a DatetimeIndex.")
    cutoff = snapshot.resolved_date
    available = prices.loc[prices.index <= cutoff]
    rows = []
    for ticker in snapshot.tickers:
        series = available[ticker].dropna() if ticker in available.columns else pd.Series(dtype=float)
        latest = series.index.max() if not series.empty else pd.NaT
        observations = int(len(series))
        staleness = (
            int((cutoff.normalize() - latest.normalize()).days)
            if pd.notna(latest) else pd.NA
        )
        has_history = observations > 0
        sufficient = observations >= config.minimum_history_observations
        current = pd.notna(latest) and staleness <= config.stale_calendar_days
        feature_row = _latest_features(features, ticker, cutoff)
        feature_ready = True if features is None else bool(
            feature_row is not None
            and all(column in feature_row.index for column in config.required_features)
            and feature_row[list(config.required_features)].notna().all()
        )
        reasons = []
        if not has_history:
            reasons.extend(["HAS_PRICE_HISTORY", "MISSING_PRICE"])
        elif not sufficient:
            reasons.extend(["SUFFICIENT_HISTORY", "INSUFFICIENT_LOOKBACK"])
        if has_history and not current:
            reasons.extend(["CURRENT_PRICE_AVAILABLE", "STALE_PRICE"])
        if not feature_ready:
            reasons.append("FEATURES_AVAILABLE")
        eligible = not reasons
        rows.append(
            {
                "ticker": ticker,
                "index_member": True,
                "eligible": eligible,
                "latest_price_date": latest,
                "history_observations": observations,
                "staleness_days": staleness,
                "feature_ready": feature_ready,
                "reason": "|".join(reasons) if reasons else None,
            }
        )
    report = pd.DataFrame(rows)
    eligible = tuple(report.loc[report.eligible, "ticker"])
    rejected = tuple(report.loc[~report.eligible, "ticker"])
    return InvestabilityResult(eligible, rejected, report)
