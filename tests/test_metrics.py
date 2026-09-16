import numpy as np
import pandas as pd

from src.evaluation.metrics import drawdown_series, max_drawdown, sortino_ratio


def test_max_drawdown_includes_initial_capital():
    returns = pd.Series([-0.2, 0.1])
    assert np.isclose(max_drawdown(returns), -0.2)
    assert np.isclose(drawdown_series(returns).iloc[0], -0.2)


def test_sortino_uses_full_sample_downside_deviation():
    returns = pd.Series([0.1, -0.1])
    expected = returns.mean() / np.sqrt(np.mean(np.minimum(returns, 0) ** 2)) * np.sqrt(12)
    assert np.isclose(sortino_ratio(returns), expected)
