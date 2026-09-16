import pandas as pd

from src.config import MOMENTUM_12_1_LOOKBACK_MONTHS, MOMENTUM_12_1_SKIP_MONTHS, MOMENTUM_LOOKBACK_MONTHS
from src.data.frequency import last_observation_by_month
from src.features.momentum import calculate_monthly_momentum_signal
from src.live.data_snapshot import DataSnapshot


def generate_current_momentum_signals(snapshot: DataSnapshot) -> pd.DataFrame:
    monthly = last_observation_by_month(snapshot.prices)
    three = calculate_monthly_momentum_signal(monthly, MOMENTUM_LOOKBACK_MONTHS).iloc[-1]
    twelve = calculate_monthly_momentum_signal(monthly, MOMENTUM_12_1_LOOKBACK_MONTHS, MOMENTUM_12_1_SKIP_MONTHS).iloc[-1]
    result = pd.DataFrame({"ticker": snapshot.metadata["requested_universe"]})
    result["signal_date"] = monthly.index[-1]
    result["momentum_3m"] = result.ticker.map(three)
    result["rank_3m"] = result.momentum_3m.rank(method="first", ascending=False)
    result["momentum_12_1"] = result.ticker.map(twelve)
    result["rank_12_1"] = result.momentum_12_1.rank(method="first", ascending=False)
    eligibility = snapshot.status.set_index("ticker")
    result["signal_valid"] = result[["momentum_3m", "momentum_12_1"]].notna().all(axis=1) & result.ticker.map(eligibility.eligible).fillna(False)
    result["invalid_reason"] = result.ticker.map(eligibility.reason)
    result.loc[result.invalid_reason.isna() & ~result.signal_valid, "invalid_reason"] = "missing_momentum_history"
    return result.sort_values(["rank_3m", "ticker"])
