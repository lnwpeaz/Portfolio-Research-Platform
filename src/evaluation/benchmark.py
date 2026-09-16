"""Benchmark identity and known return-series limitations."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkMetadata:
    benchmark_id: str
    index_name: str
    ticker: str
    return_type: str
    warning: str | None = None


SP500_PRICE_INDEX = BenchmarkMetadata(
    benchmark_id="SP500_PRICE_INDEX",
    index_name="S&P500",
    ticker="^GSPC",
    return_type="price_return",
    warning=(
        "^GSPC is a price index while adjusted security prices include dividends; "
        "benchmark-relative results have a known return-definition mismatch."
    ),
)
