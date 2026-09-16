"""Common market-data provider contract."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd


PANEL_COLUMNS = (
    "date", "canonical_ticker", "provider_ticker", "adjusted_close",
    "raw_close", "volume", "dividends", "stock_splits",
)


@dataclass(frozen=True)
class MarketDataResult:
    panel: pd.DataFrame
    requested_tickers: tuple[str, ...]
    successful_tickers: tuple[str, ...]
    failed_tickers: tuple[str, ...]
    failure_reasons: dict[str, str] = field(default_factory=dict)
    provider_parameters: dict = field(default_factory=dict)


class MarketDataProvider(ABC):
    name: str

    @abstractmethod
    def fetch_prices(self, tickers, start_date, end_date) -> MarketDataResult:
        """Fetch through inclusive ``end_date`` into the common long schema."""

    def fetch_metadata(self, tickers) -> pd.DataFrame:
        return pd.DataFrame({"canonical_ticker": list(tickers)})
