from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from src.config import EXECUTION_TIMING_MODE, ONE_WAY_TURNOVER_COST_BPS, TOP_N


@dataclass(frozen=True)
class LiveRunConfig:
    as_of_date: pd.Timestamp | None
    universe: tuple[str, ...]
    signal_frequency: str = "monthly"
    execution_mode: str = EXECUTION_TIMING_MODE
    transaction_cost_bps: float = ONE_WAY_TURNOVER_COST_BPS
    top_n: int = TOP_N
    data_source: str = "bundled_yahoo_adjusted_close"
    output_directory: Path = Path("reports/live")
    random_seed: int = 42

    def __post_init__(self):
        if self.as_of_date is not None:
            object.__setattr__(self, "as_of_date", pd.Timestamp(self.as_of_date).normalize())
        if not self.universe or len(self.universe) != len(set(self.universe)):
            raise ValueError("Live universe must be non-empty and unique.")
