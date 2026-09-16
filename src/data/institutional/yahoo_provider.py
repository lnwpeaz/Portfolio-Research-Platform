"""Batched Yahoo research adapter. Yahoo data are not point-in-time."""

import time
import pandas as pd

from src.data.institutional.normalization import TickerNormalizer, normalize_panel
from src.data.institutional.provider import MarketDataProvider, MarketDataResult, PANEL_COLUMNS


class YahooMarketDataProvider(MarketDataProvider):
    name = "yahoo"

    def __init__(self, batch_size=75, retries=3, retry_backoff_seconds=1.0, normalizer=None):
        if batch_size < 1 or retries < 1:
            raise ValueError("batch_size and retries must be positive.")
        self.batch_size = batch_size
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.normalizer = normalizer or TickerNormalizer("yahoo")

    @staticmethod
    def _field(downloaded, field, symbols):
        if downloaded.empty:
            return pd.DataFrame(index=downloaded.index)
        if isinstance(downloaded.columns, pd.MultiIndex):
            if field not in downloaded.columns.get_level_values(0):
                return pd.DataFrame(index=downloaded.index, columns=symbols, dtype=float)
            value = downloaded[field]
            return value.to_frame(symbols[0]) if isinstance(value, pd.Series) else value
        if field not in downloaded.columns:
            return pd.DataFrame(index=downloaded.index, columns=symbols, dtype=float)
        return downloaded[[field]].rename(columns={field: symbols[0]})

    def _download(self, symbols, start, end):
        import yfinance as yf
        return yf.download(
            list(symbols), start=str(pd.Timestamp(start).date()),
            # yfinance's end is exclusive; provider API is inclusive.
            end=str((pd.Timestamp(end) + pd.Timedelta(days=1)).date()),
            auto_adjust=False, actions=True, repair=False, progress=False,
            group_by="column", threads=True, timeout=30,
        )

    def fetch_prices(self, tickers, start_date, end_date):
        mapping = self.normalizer.mapping_frame(tickers)
        provider_to_canonical = dict(zip(mapping.provider_ticker, mapping.canonical_ticker))
        requested = tuple(mapping.canonical_ticker)
        panels, failures = [], {}
        for offset in range(0, len(mapping), self.batch_size):
            batch = mapping.iloc[offset:offset + self.batch_size]
            symbols = tuple(batch.provider_ticker)
            downloaded = None
            error = None
            for attempt in range(self.retries):
                try:
                    downloaded = self._download(symbols, start_date, end_date)
                    error = None
                    break
                except Exception as caught:
                    error = caught
                    if attempt + 1 < self.retries:
                        time.sleep(self.retry_backoff_seconds * (2 ** attempt))
            if downloaded is None:
                for row in batch.itertuples():
                    failures[row.canonical_ticker] = f"download_error:{error}"
                continue
            fields = {
                "adjusted_close": self._field(downloaded, "Adj Close", symbols),
                "raw_close": self._field(downloaded, "Close", symbols),
                "volume": self._field(downloaded, "Volume", symbols),
                "dividends": self._field(downloaded, "Dividends", symbols),
                "stock_splits": self._field(downloaded, "Stock Splits", symbols),
            }
            for provider_ticker in symbols:
                canonical = provider_to_canonical[provider_ticker]
                adjusted = fields["adjusted_close"].get(provider_ticker, pd.Series(dtype=float))
                raw = fields["raw_close"].get(provider_ticker, pd.Series(dtype=float))
                if adjusted.dropna().empty and not raw.dropna().empty:
                    # Modern Yahoo may omit Adj Close when no adjustment differs.
                    adjusted = raw.copy()
                if adjusted.dropna().empty:
                    failures[canonical] = "no_price_observations"
                    continue
                ticker_panel = pd.DataFrame({
                    "date": adjusted.index,
                    "canonical_ticker": canonical,
                    "provider_ticker": provider_ticker,
                    "adjusted_close": adjusted.to_numpy(),
                    "raw_close": fields["raw_close"].get(provider_ticker, pd.Series(index=adjusted.index, dtype=float)).reindex(adjusted.index).to_numpy(),
                    "volume": fields["volume"].get(provider_ticker, pd.Series(index=adjusted.index, dtype=float)).reindex(adjusted.index).to_numpy(),
                    "dividends": fields["dividends"].get(provider_ticker, pd.Series(index=adjusted.index, dtype=float)).reindex(adjusted.index).fillna(0).to_numpy(),
                    "stock_splits": fields["stock_splits"].get(provider_ticker, pd.Series(index=adjusted.index, dtype=float)).reindex(adjusted.index).fillna(0).to_numpy(),
                }).dropna(subset=["adjusted_close"])
                panels.append(ticker_panel)
        panel = normalize_panel(pd.concat(panels, ignore_index=True)) if panels else pd.DataFrame(columns=PANEL_COLUMNS)
        successful = tuple(sorted(set(panel.canonical_ticker))) if not panel.empty else ()
        failed = tuple(sorted(set(requested).difference(successful)))
        return MarketDataResult(
            panel=panel, requested_tickers=requested,
            successful_tickers=successful, failed_tickers=failed,
            failure_reasons={ticker: failures.get(ticker, "no_price_observations") for ticker in failed},
            provider_parameters={
                "batch_size": self.batch_size, "retries": self.retries,
                "auto_adjust": False, "actions": True, "repair": False,
                "vendor_warning": "Yahoo adjusted history is revisable and is not point-in-time institutional data.",
            },
        )
