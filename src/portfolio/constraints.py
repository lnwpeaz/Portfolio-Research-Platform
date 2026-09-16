# src/portfolio/constraints.py

import numpy as np
import pandas as pd


def cap_weights(
    weights: pd.Series,
    max_weight: float = 0.40,
) -> pd.Series:
    """Project long-only weights onto a fully invested capped simplex.

    Clipping and then renormalizing is not sufficient: renormalization can
    push a clipped position above ``max_weight`` again.  The bisection below
    finds ``x = clip(w - lambda, 0, max_weight)`` with ``sum(x) == 1``.
    """
    if weights.empty:
        raise ValueError("Portfolio weights are empty.")
    if not 0 < max_weight <= 1:
        raise ValueError("max_weight must be in (0, 1].")
    if len(weights) * max_weight < 1 - 1e-12:
        raise ValueError(
            "Maximum weight is infeasible for the number of holdings: "
            f"{len(weights)} * {max_weight:.6f} < 1."
        )
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Portfolio weights must be finite.")

    raw = weights.astype(float).clip(lower=0.0).to_numpy(copy=True)
    if raw.sum() <= 0:
        raw = np.ones(len(raw), dtype=float)
    raw /= raw.sum()

    lower = float(raw.min() - max_weight)
    upper = float(raw.max())
    for _ in range(100):
        midpoint = (lower + upper) / 2
        projected = np.clip(raw - midpoint, 0.0, max_weight)
        if projected.sum() > 1:
            lower = midpoint
        else:
            upper = midpoint

    projected = np.clip(raw - upper, 0.0, max_weight)
    # Remove the final floating-point residual without breaking the cap.
    residual = 1.0 - projected.sum()
    if abs(residual) > 1e-12:
        available = max_weight - projected if residual > 0 else projected
        candidates = np.flatnonzero(available > 1e-12)
        projected[candidates[0]] += residual

    return pd.Series(projected, index=weights.index, name=weights.name)


def validate_weights(
    weights: pd.Series,
    tolerance: float = 1e-6,
    max_weight: float | None = None,
) -> None:

    if weights.empty:
        raise ValueError("Portfolio weights are empty.")

    if weights.index.has_duplicates:
        raise ValueError("Duplicate assets detected in portfolio weights.")

    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite portfolio weight detected.")

    if (weights < 0).any():
        raise ValueError(
            "Negative portfolio weight detected."
        )

    if abs(weights.sum() - 1.0) > tolerance:
        raise ValueError(
            f"Weights do not sum to 1: "
            f"{weights.sum():.6f}"
        )

    if max_weight is not None and (weights > max_weight + tolerance).any():
        raise ValueError(
            f"Portfolio weight exceeds maximum of {max_weight:.6f}."
        )
