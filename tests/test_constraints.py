import numpy as np
import pandas as pd
import pytest

from src.portfolio.constraints import cap_weights, validate_weights


def test_cap_weights_redistributes_without_rebreaking_cap():
    weights = pd.Series({"A": 0.8, "B": 0.1, "C": 0.1})
    capped = cap_weights(weights, max_weight=0.4)

    assert np.isclose(capped.sum(), 1.0)
    assert capped.max() <= 0.4 + 1e-10
    assert np.allclose(capped.sort_index(), [0.4, 0.3, 0.3])
    validate_weights(capped, max_weight=0.4)


def test_cap_weights_rejects_infeasible_constraint():
    with pytest.raises(ValueError, match="infeasible"):
        cap_weights(pd.Series({"A": 0.5, "B": 0.5}), max_weight=0.4)
