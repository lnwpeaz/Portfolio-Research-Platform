import pandas as pd
import pytest

from src.universe.fixed import FIXED_20_TICKERS, FixedUniverseProvider
from src.universe.investability import InvestabilityConfig, screen_investability
from src.universe.models import PointInTimeMembershipUnavailable
from src.universe.sp500 import SP500UniverseProvider


def _history(tmp_path):
    frame = pd.DataFrame(
        [
            {"index_name": "S&P500", "ticker": "OLD", "company_name": "Old Co",
             "sector": "Industrials", "industry": "Tools", "effective_from": "2019-01-01",
             "effective_to": "2020-01-01", "source": "test_vendor", "source_date": "2024-01-01"},
            {"index_name": "S&P500", "ticker": "NEW", "company_name": "New Co",
             "sector": "Technology", "industry": "Software", "effective_from": "2020-01-01",
             "effective_to": None, "source": "test_vendor", "source_date": "2024-01-01"},
        ]
    )
    path = tmp_path / "membership.csv"
    frame.to_csv(path, index=False)
    return path


def test_fixed20_reproduces_existing_tickers_and_snapshot_is_immutable():
    provider = FixedUniverseProvider()
    snapshot = provider.get_snapshot("2025-12-31")
    assert snapshot.tickers == tuple(sorted(FIXED_20_TICKERS))
    assert snapshot.member_count == 20
    with pytest.raises((AttributeError, TypeError)):
        snapshot.metadata["member_count"] = 0
    with pytest.raises(AttributeError):
        snapshot.members = ()
    assert provider.get_snapshot("2025-12-31").snapshot_hash == snapshot.snapshot_hash


def test_membership_boundaries_are_half_open_and_no_future_constituent(tmp_path):
    provider = SP500UniverseProvider(historical_path=_history(tmp_path))
    assert {m.ticker for m in provider.get_membership("2019-12-31", require_point_in_time=True)} == {"OLD"}
    assert {m.ticker for m in provider.get_membership("2020-01-01", require_point_in_time=True)} == {"NEW"}
    assert "NEW" not in {m.ticker for m in provider.get_membership("2019-12-31")}
    assert "OLD" not in {m.ticker for m in provider.get_membership("2020-01-01")}


def test_missing_pit_history_fails_explicitly():
    provider = SP500UniverseProvider(
        current_frame=pd.DataFrame([{"ticker": "NOW", "company_name": "Now"}])
    )
    with pytest.raises(PointInTimeMembershipUnavailable, match="POINT_IN_TIME_MEMBERSHIP_UNAVAILABLE"):
        provider.get_snapshot("2020-01-01", require_point_in_time=True)


def test_investability_reports_history_staleness_and_features_without_mutation():
    snapshot = FixedUniverseProvider(("GOOD", "SHORT", "STALE", "MISSING")).get_snapshot("2024-01-10")
    dates = pd.bdate_range("2023-01-01", "2024-01-10")
    prices = pd.DataFrame(index=dates, columns=["GOOD", "SHORT", "STALE"], dtype=float)
    prices["GOOD"] = 100.0
    prices.loc[dates[-10]:, "SHORT"] = 10.0
    prices.loc[:"2023-12-01", "STALE"] = 20.0
    features = pd.DataFrame({
        "ticker": ["GOOD", "SHORT", "STALE"], "date": [dates[-1]] * 3,
        "momentum_1m": [1.0, 1.0, 1.0], "momentum_3m": [1.0, 1.0, 1.0],
        "momentum_6m": [1.0, 1.0, 1.0], "volatility_3m": [1.0, 1.0, 1.0],
        "drawdown_3m": [0.0, pd.NA, 0.0],
    })
    result = screen_investability(snapshot, prices, features, InvestabilityConfig(minimum_history_observations=20))
    report = result.report.set_index("ticker")
    assert result.eligible_tickers == ("GOOD",)
    assert "INSUFFICIENT_LOOKBACK" in report.loc["SHORT", "reason"]
    assert "FEATURES_AVAILABLE" in report.loc["SHORT", "reason"]
    assert "STALE_PRICE" in report.loc["STALE", "reason"]
    assert "MISSING_PRICE" in report.loc["MISSING", "reason"]
    assert snapshot.member_count == 4
