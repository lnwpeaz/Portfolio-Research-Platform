"""Legacy fixed universe provider."""

from datetime import datetime, timezone
import pandas as pd

from src.universe.base import UniverseProvider
from src.universe.models import UniverseMember, UniverseSnapshot


FIXED_20_TICKERS = (
    "AAPL", "ABBV", "AMZN", "COST", "CVX", "GOOGL", "HD", "JPM", "KO", "MA",
    "META", "MSFT", "NVDA", "PEP", "PG", "TSLA", "UNH", "V", "WMT", "XOM",
)


class FixedUniverseProvider(UniverseProvider):
    name = "fixed20"
    index_name = "LEGACY_FIXED_20"

    def __init__(self, tickers=FIXED_20_TICKERS):
        self._tickers = tuple(str(ticker).upper() for ticker in tickers)

    def get_membership(self, as_of_date, *, require_point_in_time=False):
        return tuple(
            UniverseMember(
                ticker=ticker,
                company_name=ticker,
                source="legacy_fixed_configuration",
                source_date=pd.Timestamp("2015-01-01"),
                point_in_time_verified=False,
            )
            for ticker in self._tickers
        )

    def get_snapshot(self, as_of_date=None, *, require_point_in_time=False):
        requested = self.normalize_date(as_of_date or pd.Timestamp.today())
        return UniverseSnapshot(
            index_name=self.index_name,
            requested_date=requested,
            resolved_date=requested,
            members=self.get_membership(requested),
            provider=self.name,
            point_in_time_verified=False,
            source="legacy_fixed_configuration",
            warnings=("Fixed ex-post reference universe; survivorship bias remains.",),
            retrieved_at=datetime.now(timezone.utc).isoformat(),
        )
