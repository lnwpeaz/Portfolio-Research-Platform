"""Batched adjusted-price downloads with explicit partial-download metadata."""

from pathlib import Path
import pandas as pd

from src.data.cache import build_cache_metadata, write_cache_metadata


def download_adjusted_prices(
    tickers, start, end, output_path, *, batch_size=100, source="yfinance"
):
    """Download in ticker batches; failed names remain visible in metadata."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    import yfinance as yf

    requested = tuple(dict.fromkeys(str(ticker) for ticker in tickers))
    panels = []
    successful = []
    for offset in range(0, len(requested), batch_size):
        batch = requested[offset:offset + batch_size]
        downloaded = yf.download(
            list(batch), start=start, end=end, auto_adjust=True,
            actions=False, progress=False, group_by="column", threads=True,
        )
        if downloaded.empty:
            continue
        close = downloaded["Close"] if isinstance(downloaded.columns, pd.MultiIndex) else downloaded[["Close"]].rename(columns={"Close": batch[0]})
        close = close.dropna(axis=1, how="all")
        successful.extend(map(str, close.columns))
        panels.append(close)
    prices = pd.concat(panels, axis=1).sort_index() if panels else pd.DataFrame()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    metadata = build_cache_metadata(source, str(start), str(end), requested, successful)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(output_path)
    write_cache_metadata(output_path.with_suffix(".metadata.json"), metadata)
    return prices, metadata
