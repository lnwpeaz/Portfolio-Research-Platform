import numpy as np
import pandas as pd

from src.data.frequency import last_observation_by_month


def test_month_end_selection_preserves_missing_value_on_final_session():
    data = pd.DataFrame(
        {"A": [100.0, np.nan], "B": [100.0, 101.0]},
        index=pd.to_datetime(["2020-01-30", "2020-01-31"]),
    )
    monthly = last_observation_by_month(data)

    assert monthly.index.tolist() == [pd.Timestamp("2020-01-31")]
    assert np.isnan(monthly.loc[pd.Timestamp("2020-01-31"), "A"])
