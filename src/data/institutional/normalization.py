"""Deterministic canonical-to-vendor identifier mapping."""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from src.data.institutional.provider import PANEL_COLUMNS


@dataclass(frozen=True)
class TickerNormalizer:
    provider: str = "yahoo"
    explicit_mapping: dict[str, str] = field(default_factory=dict)

    def to_provider(self, canonical_ticker: str) -> str:
        canonical = str(canonical_ticker).strip().upper()
        if canonical in self.explicit_mapping:
            return self.explicit_mapping[canonical]
        return canonical.replace(".", "-") if self.provider == "yahoo" else canonical

    def mapping_frame(self, tickers) -> pd.DataFrame:
        canonical = tuple(dict.fromkeys(str(t).strip().upper() for t in tickers))
        provider = tuple(self.to_provider(ticker) for ticker in canonical)
        if len(set(provider)) != len(provider):
            raise ValueError("Ticker normalization creates a provider-symbol collision.")
        return pd.DataFrame({"canonical_ticker": canonical, "provider_ticker": provider})

    def to_canonical(self, provider_ticker: str, mapping: pd.DataFrame) -> str:
        matches = mapping.loc[
            mapping.provider_ticker.eq(str(provider_ticker).strip().upper()),
            "canonical_ticker",
        ]
        if len(matches) != 1:
            raise KeyError(f"Provider ticker is absent or ambiguous: {provider_ticker}")
        return str(matches.iloc[0])


def normalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Normalize types/order without changing or filling vendor observations."""
    missing = set(PANEL_COLUMNS).difference(panel.columns)
    if missing:
        raise ValueError(f"Market panel missing columns: {sorted(missing)}")
    result = panel.loc[:, PANEL_COLUMNS].copy()
    result["date"] = pd.to_datetime(result.date, errors="raise").dt.tz_localize(None).dt.normalize()
    result["canonical_ticker"] = result.canonical_ticker.astype(str).str.upper()
    result["provider_ticker"] = result.provider_ticker.astype(str).str.upper()
    for column in ("adjusted_close", "raw_close", "volume", "dividends", "stock_splits"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.sort_values(["canonical_ticker", "date"], kind="stable").reset_index(drop=True)
