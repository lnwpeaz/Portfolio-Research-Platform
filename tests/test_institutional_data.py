import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from institutional_live_main import parse_args as parse_live_args
from src.data.institutional.acquisition import acquire_market_data
from src.data.institutional.cache import InstitutionalDataCache, merge_incremental
from src.data.institutional.coverage import resolve_common_market_date
from src.data.institutional.normalization import TickerNormalizer, normalize_panel
from src.data.institutional.provider import MarketDataProvider, MarketDataResult
from src.data.institutional.validation import security_readiness, validate_market_panel
from src.features.ml_dataset import FEATURE_COLUMNS, build_ml_feature_panel
from src.live.institutional_paper import write_institutional_paper_target
from src.live.institutional_runner import _sector_exposure
from src.universe.fixed import FixedUniverseProvider
from src.universe.investability import InvestabilityConfig, screen_investability


def _panel(tickers=("AAA", "BBB"), start="2024-01-01", periods=10):
    dates = pd.bdate_range(start, periods=periods)
    rows = []
    for number, ticker in enumerate(tickers):
        for offset, date in enumerate(dates):
            price = 100 + number + offset
            rows.append({
                "date": date, "canonical_ticker": ticker,
                "provider_ticker": ticker.replace(".", "-"),
                "adjusted_close": price, "raw_close": price,
                "volume": 1000, "dividends": 0, "stock_splits": 0,
            })
    return pd.DataFrame(rows)


class FakeProvider(MarketDataProvider):
    name = "fake"

    def __init__(self):
        self.normalizer = TickerNormalizer("yahoo")
        self.calls = []

    def fetch_prices(self, tickers, start_date, end_date):
        self.calls.append((tuple(tickers), pd.Timestamp(start_date), pd.Timestamp(end_date)))
        dates = pd.bdate_range(start_date, end_date)
        panel = _panel(tickers, start=dates.min(), periods=len(dates)) if len(dates) else pd.DataFrame()
        return MarketDataResult(
            normalize_panel(panel), tuple(tickers), tuple(tickers), (), {},
            {"fake": True},
        )


def test_ticker_mapping_is_deterministic_and_reverse_is_explicit():
    normalizer = TickerNormalizer("yahoo")
    mapping = normalizer.mapping_frame(["BRK.B", "BF.B", "ABC-D"])
    assert mapping.provider_ticker.tolist() == ["BRK-B", "BF-B", "ABC-D"]
    assert normalizer.to_canonical("BRK-B", mapping) == "BRK.B"
    assert normalizer.to_canonical("ABC-D", mapping) == "ABC-D"
    with pytest.raises(KeyError):
        normalizer.to_canonical("UNKNOWN", mapping)


def test_normalization_and_incremental_merge_are_deterministic():
    original = _panel(("AAA",), periods=5)
    shuffled = original.sample(frac=1, random_state=1)
    pd.testing.assert_frame_equal(normalize_panel(original), normalize_panel(shuffled))
    extension = _panel(("AAA",), start="2024-01-08", periods=3)
    merged = merge_incremental(original, [extension])
    assert merged.date.min() == original.date.min()
    assert merged.date.max() == extension.date.max()
    assert not merged.duplicated(["canonical_ticker", "date"]).any()


def test_valid_cache_is_reused_and_incremental_extension_keeps_history(tmp_path):
    provider = FakeProvider()
    cache = InstitutionalDataCache(tmp_path / "cache")
    first = acquire_market_data(
        provider, cache, ("AAA", "BBB"), "2024-01-01", "2024-01-05", "hash",
        minimum_coverage=1.0, minimum_observations=1,
    )
    first_calls = len(provider.calls)
    second = acquire_market_data(
        provider, cache, ("AAA", "BBB"), "2024-01-01", "2024-01-05", "hash",
        minimum_coverage=1.0, minimum_observations=1,
    )
    assert len(provider.calls) == first_calls
    assert second.manifest["cache_reused_without_download"]
    third = acquire_market_data(
        provider, cache, ("AAA", "BBB"), "2024-01-01", "2024-01-10", "hash",
        minimum_coverage=1.0, minimum_observations=1,
    )
    assert third.panel.date.min() == first.panel.date.min()
    assert third.panel.date.max() > first.panel.date.max()


def test_cache_does_not_refetch_prelisting_dates_for_short_history(tmp_path):
    provider = FakeProvider()
    cache = InstitutionalDataCache(tmp_path / "cache")
    panel = normalize_panel(_panel(("OLD",), "2024-01-01", 5))
    short = normalize_panel(_panel(("NEW",), "2024-01-04", 2))
    benchmark = normalize_panel(_panel(("^GSPC",), "2024-01-01", 5))
    combined = merge_incremental(panel, [short, benchmark])
    manifest = {
        "run_id": "seed", "provider": "fake", "created_at": "2024-01-06T00:00:00+00:00",
        "requested_start": "2024-01-01", "requested_end": "2024-01-05",
        "provider_parameters": {}, "downloaded_tickers": ["OLD", "NEW", "^GSPC"],
    }
    cache.write_version("fake", "seed", [combined], combined, manifest)
    acquire_market_data(
        provider, cache, ("OLD", "NEW"), "2024-01-01", "2024-01-05", "hash",
        minimum_coverage=0.5, minimum_observations=1,
    )
    assert provider.calls == []


def test_validation_flags_invalid_prices_duplicates_nonmonotonic_and_future():
    panel = _panel(("AAA",), periods=4)
    duplicate = panel.iloc[[0]].copy()
    panel = pd.concat([panel.iloc[::-1], duplicate], ignore_index=True)
    panel.loc[0, "adjusted_close"] = 0
    panel.loc[1, "raw_close"] = -1
    panel.loc[2, "volume"] = -5
    flags = validate_market_panel(panel, requested_date="2024-01-03")
    assert {"DUPLICATE_DATE", "NON_MONOTONIC_DATES", "INVALID_ADJUSTED_CLOSE", "INVALID_RAW_CLOSE", "INVALID_VOLUME", "FUTURE_OBSERVATION"}.issubset(set(flags.flag))


def test_readiness_reports_stale_and_insufficient_history():
    panel = normalize_panel(_panel(("SHORT", "STALE"), periods=3))
    panel = panel[~((panel.canonical_ticker == "STALE") & (panel.date > panel.date.min()))]
    status = security_readiness(panel, ("SHORT", "STALE", "MISSING"), "2024-01-10", minimum_observations=5, stale_calendar_days=2).set_index("ticker")
    assert "INSUFFICIENT_HISTORY" in status.loc["SHORT", "reason"]
    assert "STALE_PRICE" in status.loc["STALE", "reason"]
    assert "MISSING_PRICE" in status.loc["MISSING", "reason"]


def test_weekend_resolution_requires_broad_coverage_not_isolated_latest_ticker():
    panel = normalize_panel(_panel(("AAA", "BBB", "CCC", "DDD"), start="2024-01-01", periods=5))
    isolated = _panel(("AAA",), start="2024-01-08", periods=1)
    panel = merge_incremental(panel, [isolated])
    resolved = resolve_common_market_date(panel, ("AAA", "BBB", "CCC", "DDD"), "2024-01-13", 0.95)
    assert resolved["resolved_market_date"] == pd.Timestamp("2024-01-05")
    with pytest.raises(ValueError, match="No date meets"):
        resolve_common_market_date(panel[panel.canonical_ticker.eq("AAA")], ("AAA", "BBB"), "2024-01-13", 0.95)


def test_valid_member_becomes_eligible_missing_member_remains_reported():
    snapshot = FixedUniverseProvider(("GOOD", "MISSING")).get_snapshot("2024-12-31")
    dates = pd.bdate_range(end="2024-12-31", periods=260)
    prices = pd.DataFrame({"GOOD": np.linspace(80, 100, len(dates))}, index=dates)
    features = pd.DataFrame({"ticker": ["GOOD"], "date": [dates[-1]], **{column: [1.0] for column in FEATURE_COLUMNS}})
    result = screen_investability(snapshot, prices, features, InvestabilityConfig(minimum_history_observations=252))
    assert result.eligible_tickers == ("GOOD",)
    assert "MISSING_PRICE" in result.report.set_index("ticker").loc["MISSING", "reason"]


def test_current_feature_panel_has_no_targets_and_excludes_future_rows():
    dates = pd.bdate_range("2023-01-01", periods=300)
    prices = pd.DataFrame({"AAA": np.linspace(50, 100, len(dates))}, index=dates)
    cutoff = dates[-20]
    features = build_ml_feature_panel(prices.loc[:cutoff])
    assert features.date.max() <= cutoff
    assert not {"target", "target_date", "future_return", "benchmark_return"}.intersection(features.columns)


def test_sp500_paper_namespace_is_explicit_and_fixed_files_unchanged(tmp_path):
    fixed = tmp_path / "paper" / "transactions.csv"
    fixed.parent.mkdir(parents=True)
    fixed.write_text("original\n")
    assert not parse_live_args([]).paper_update
    target = pd.DataFrame({"ticker": ["AAA"], "target_weight": [1.0]})
    write_institutional_paper_target("sp500", "run1", target, {"paper_update": True}, tmp_path / "paper")
    assert fixed.read_text() == "original\n"
    assert (tmp_path / "paper" / "sp500" / "runs" / "run1" / "target.csv").exists()


def test_current_sector_exposure_uses_existing_metadata_without_duplicate_columns():
    rankings = pd.DataFrame({
        "ticker": ["AAA", "BBB"], "sector": ["Tech", "Health"],
        "ml_rank": [1, 2],
    })
    membership = pd.DataFrame({"ticker": ["AAA", "BBB"], "sector": ["Tech", "Health"]})
    quantiles = pd.DataFrame({
        "ticker": ["AAA", "BBB"], "sector": ["Tech", "Health"],
        "quantile": ["Q5", "Q1"],
    })
    exposure = _sector_exposure(rankings, membership, quantiles, top_n=1)
    assert np.isclose(exposure.eligible_equal_weight.sum(), 1)
    assert np.isclose(exposure.q5_equal_weight.sum(), 1)
