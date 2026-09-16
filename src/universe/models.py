"""Immutable, deterministic universe data models."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import pandas as pd


POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE = "POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE"


class PointInTimeMembershipUnavailable(RuntimeError):
    """Raised rather than silently substituting current constituents."""

    status = POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE


def _timestamp(value):
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).normalize()


@dataclass(frozen=True, order=True)
class UniverseMember:
    ticker: str
    company_name: str = ""
    sector: str = "Unknown"
    industry: str = "Unknown"
    membership_start: pd.Timestamp | None = None
    membership_end: pd.Timestamp | None = None
    source: str = ""
    source_date: pd.Timestamp | None = None
    point_in_time_verified: bool = False

    def __post_init__(self):
        object.__setattr__(self, "ticker", str(self.ticker).strip().upper())
        object.__setattr__(self, "company_name", str(self.company_name or self.ticker))
        object.__setattr__(self, "sector", str(self.sector or "Unknown"))
        object.__setattr__(self, "industry", str(self.industry or "Unknown"))
        object.__setattr__(self, "membership_start", _timestamp(self.membership_start))
        object.__setattr__(self, "membership_end", _timestamp(self.membership_end))
        object.__setattr__(self, "source_date", _timestamp(self.source_date))

    def as_record(self) -> dict:
        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "industry": self.industry,
            "membership_start": self.membership_start,
            "membership_end": self.membership_end,
            "source": self.source,
            "source_date": self.source_date,
            "point_in_time_verified": self.point_in_time_verified,
        }


@dataclass(frozen=True)
class UniverseSnapshot:
    index_name: str
    requested_date: pd.Timestamp
    resolved_date: pd.Timestamp
    members: tuple[UniverseMember, ...]
    provider: str
    point_in_time_verified: bool
    source: str
    warnings: tuple[str, ...] = ()
    retrieved_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    snapshot_hash: str = field(init=False)
    metadata: Mapping = field(init=False, compare=False)

    def __post_init__(self):
        requested = pd.Timestamp(self.requested_date).normalize()
        resolved = pd.Timestamp(self.resolved_date).normalize()
        members = tuple(sorted(self.members, key=lambda item: item.ticker))
        if len({member.ticker for member in members}) != len(members):
            raise ValueError("Universe snapshot contains duplicate tickers.")
        object.__setattr__(self, "requested_date", requested)
        object.__setattr__(self, "resolved_date", resolved)
        object.__setattr__(self, "members", members)
        payload = {
            "index_name": self.index_name,
            "requested_date": requested.isoformat(),
            "resolved_date": resolved.isoformat(),
            "provider": self.provider,
            "point_in_time_verified": bool(self.point_in_time_verified),
            "source": self.source,
            "members": [
                {
                    **member.as_record(),
                    "membership_start": member.membership_start.isoformat()
                    if member.membership_start is not None else None,
                    "membership_end": member.membership_end.isoformat()
                    if member.membership_end is not None else None,
                    "source_date": member.source_date.isoformat()
                    if member.source_date is not None else None,
                }
                for member in members
            ],
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        object.__setattr__(self, "snapshot_hash", digest)
        status = "VERIFIED" if self.point_in_time_verified else "CURRENT_ONLY"
        metadata = MappingProxyType(
            {
                "provider": self.provider,
                "requested_as_of_date": requested,
                "resolved_as_of_date": resolved,
                "member_count": len(members),
                "source": self.source,
                "retrieved_at": self.retrieved_at,
                "point_in_time_status": status,
                "warnings": self.warnings,
            }
        )
        object.__setattr__(self, "metadata", metadata)

    @property
    def member_count(self) -> int:
        return len(self.members)

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(member.ticker for member in self.members)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([member.as_record() for member in self.members])
