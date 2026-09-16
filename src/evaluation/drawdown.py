"""Drawdown episode analytics using the platform's initial-capital convention."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import drawdown_series


def drawdown_summary(returns: pd.Series) -> pd.Series:
    """Describe the maximum drawdown and longest underwater episode.

    Dates mark the last high-water mark, trough, and first recovery. For a
    drawdown beginning with the first observation, the first return date is
    used as the start because initial capital has no market timestamp.
    Durations are counted in return observations.
    """
    drawdowns = drawdown_series(returns)
    if drawdowns.empty:
        return pd.Series(
            {
                "Maximum Drawdown": np.nan,
                "Drawdown Start": pd.NaT,
                "Trough Date": pd.NaT,
                "Recovery Date": pd.NaT,
                "Drawdown Duration": np.nan,
                "Longest Drawdown Duration": np.nan,
            }
        )

    episodes = []
    in_drawdown = False
    start_position = 0
    last_peak_position = 0
    trough_position = 0
    trough_value = 0.0

    for position, value in enumerate(drawdowns.to_numpy()):
        if value >= -1e-12:
            if in_drawdown:
                episodes.append(
                    {
                        "start_position": start_position,
                        "trough_position": trough_position,
                        "recovery_position": position,
                        "depth": trough_value,
                        "duration": position - start_position,
                    }
                )
                in_drawdown = False
            last_peak_position = position
        else:
            if not in_drawdown:
                in_drawdown = True
                start_position = last_peak_position if position > 0 else 0
                trough_position = position
                trough_value = value
            elif value < trough_value:
                trough_position = position
                trough_value = value

    if in_drawdown:
        episodes.append(
            {
                "start_position": start_position,
                "trough_position": trough_position,
                "recovery_position": None,
                "depth": trough_value,
                "duration": len(drawdowns) - 1 - start_position,
            }
        )

    if not episodes:
        return pd.Series(
            {
                "Maximum Drawdown": 0.0,
                "Drawdown Start": pd.NaT,
                "Trough Date": pd.NaT,
                "Recovery Date": pd.NaT,
                "Drawdown Duration": 0,
                "Longest Drawdown Duration": 0,
            }
        )

    maximum = min(episodes, key=lambda episode: episode["depth"])
    recovery_position = maximum["recovery_position"]
    return pd.Series(
        {
            "Maximum Drawdown": maximum["depth"],
            "Drawdown Start": drawdowns.index[maximum["start_position"]],
            "Trough Date": drawdowns.index[maximum["trough_position"]],
            "Recovery Date": (
                drawdowns.index[recovery_position]
                if recovery_position is not None
                else pd.NaT
            ),
            "Drawdown Duration": maximum["duration"],
            "Longest Drawdown Duration": max(
                episode["duration"] for episode in episodes
            ),
        }
    )
