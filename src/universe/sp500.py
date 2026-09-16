"""S&P 500 provider with explicit current and historical source adapters."""

from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen
import pandas as pd

from src.universe.base import UniverseProvider
from src.universe.models import (
    POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE,
    PointInTimeMembershipUnavailable,
    UniverseMember,
    UniverseSnapshot,
)


HISTORICAL_COLUMNS = {
    "index_name", "ticker", "company_name", "sector", "industry",
    "effective_from", "effective_to", "source", "source_date",
}

CURRENT_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_current_sp500(url=CURRENT_SP500_URL) -> pd.DataFrame:
    """Fetch the published current table; this is never treated as history."""
    request = Request(url, headers={"User-Agent": "portfolio-research-platform/1.0"})
    with urlopen(request, timeout=30) as response:
        html = response.read().decode("utf-8")
    tables = pd.read_html(StringIO(html), attrs={"id": "constituents"})
    if not tables:
        raise RuntimeError("Current S&P 500 constituent table was not found.")
    table = tables[0].rename(
        columns={
            "Symbol": "ticker",
            "Security": "company_name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "industry",
        }
    )
    required = {"ticker", "company_name", "sector", "industry"}
    if not required.issubset(table.columns):
        raise ValueError("Current S&P 500 source schema changed unexpectedly.")
    result = table[list(required)].copy()
    result["source"] = url
    result["source_date"] = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
    return result


def load_membership_csv(path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = HISTORICAL_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Membership data missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["effective_from"] = pd.to_datetime(frame["effective_from"], errors="raise")
    frame["effective_to"] = pd.to_datetime(frame["effective_to"], errors="coerce")
    frame["source_date"] = pd.to_datetime(frame["source_date"], errors="raise")
    if frame.duplicated(["index_name", "ticker", "effective_from"]).any():
        raise ValueError("Membership data contains duplicate effective records.")
    invalid = frame.effective_to.notna() & (frame.effective_to <= frame.effective_from)
    if invalid.any():
        raise ValueError("effective_to must be later than effective_from.")
    return frame


class SP500UniverseProvider(UniverseProvider):
    name = "sp500"
    index_name = "S&P500"

    def __init__(self, historical_path=None, current_path=None, current_frame=None):
        self.historical_path = Path(historical_path) if historical_path else None
        self.current_path = Path(current_path) if current_path else None
        self._current_frame = current_frame.copy() if current_frame is not None else None

    def _historical(self):
        if self.historical_path is None or not self.historical_path.exists():
            return None
        frame = load_membership_csv(self.historical_path)
        return frame[frame.index_name.eq(self.index_name)].copy()

    def _current(self):
        if self._current_frame is not None:
            return self._current_frame.copy()
        if self.current_path is not None and self.current_path.exists():
            frame = pd.read_csv(self.current_path)
        else:
            return None
        required = {"ticker", "company_name", "sector", "industry", "source", "source_date"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(
                f"Current membership data missing provenance columns: {sorted(missing)}"
            )
        frame["source_date"] = pd.to_datetime(frame["source_date"], errors="raise")
        return frame

    @staticmethod
    def _members(frame, *, verified):
        members = []
        for row in frame.sort_values("ticker").to_dict("records"):
            members.append(
                UniverseMember(
                    ticker=row["ticker"],
                    company_name=row.get("company_name") or row["ticker"],
                    sector=row.get("sector") or "Unknown",
                    industry=row.get("industry") or "Unknown",
                    membership_start=row.get("effective_from"),
                    membership_end=row.get("effective_to"),
                    source=row.get("source") or "external_current_membership",
                    source_date=row.get("source_date"),
                    point_in_time_verified=verified,
                )
            )
        return tuple(members)

    def get_membership(self, as_of_date, *, require_point_in_time=False):
        requested = self.normalize_date(as_of_date)
        history = self._historical()
        if history is not None:
            # Half-open intervals: effective_from <= t < effective_to.
            active = history[
                history.effective_from.le(requested)
                & (history.effective_to.isna() | history.effective_to.gt(requested))
            ]
            if not active.empty:
                return self._members(active, verified=True)
        if require_point_in_time:
            raise PointInTimeMembershipUnavailable(
                f"{POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE}: no dated S&P 500 "
                f"membership records cover {requested.date()}."
            )
        current = self._current()
        if current is None or current.empty:
            raise PointInTimeMembershipUnavailable(
                f"{POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE}: no current S&P 500 "
                "constituent source is configured. Supply --current-membership."
            )
        return self._members(current, verified=False)

    def get_snapshot(self, as_of_date=None, *, require_point_in_time=False):
        requested = self.normalize_date(as_of_date or pd.Timestamp.today())
        history = self._historical()
        verified = False
        source = "external_current_membership"
        if history is not None:
            covered = history.effective_from.le(requested) & (
                history.effective_to.isna() | history.effective_to.gt(requested)
            )
            verified = bool(covered.any())
            if verified:
                source = ";".join(sorted(history.loc[covered, "source"].astype(str).unique()))
        members = self.get_membership(requested, require_point_in_time=require_point_in_time)
        if not verified and members:
            source_dates = [member.source_date for member in members if member.source_date is not None]
            if source_dates and requested < max(source_dates):
                raise PointInTimeMembershipUnavailable(
                    f"{POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE}: current constituents "
                    f"retrieved on {max(source_dates).date()} cannot represent "
                    f"requested date {requested.date()}."
                )
            source = ";".join(sorted({member.source for member in members}))
        warnings = () if verified else (
            "Current/prospective constituents only; historical survivorship bias is not solved.",
        )
        return UniverseSnapshot(
            index_name=self.index_name,
            requested_date=requested,
            resolved_date=requested,
            members=members,
            provider=self.name,
            point_in_time_verified=verified,
            source=source,
            warnings=warnings,
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
