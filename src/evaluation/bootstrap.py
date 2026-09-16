"""Paired moving-block bootstrap diagnostics for monthly strategy returns."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import max_drawdown


BOOTSTRAP_METRICS = (
    "annual_return", "annual_volatility", "sharpe", "annual_active_return",
    "tracking_error", "information_ratio", "max_drawdown",
)


def moving_block_indices(
    observations: int,
    simulations: int = 5000,
    block_size: int = 6,
    seed: int = 42,
) -> np.ndarray:
    """Return circular moving-block indices with exactly ``observations`` columns."""
    if observations < 2 or simulations < 1 or block_size < 1:
        raise ValueError("Invalid bootstrap dimensions.")
    rng = np.random.default_rng(seed)
    blocks = int(np.ceil(observations / block_size))
    starts = rng.integers(0, observations, size=(simulations, blocks))
    offsets = np.arange(block_size)
    return ((starts[..., None] + offsets) % observations).reshape(simulations, -1)[
        :, :observations
    ]


def paired_block_bootstrap(
    strategy: pd.Series,
    benchmark: pd.Series,
    simulations: int = 5000,
    block_size: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    """Resample paired observations, preserving strategy/benchmark dependence."""
    aligned = pd.concat(
        [strategy.rename("strategy"), benchmark.rename("benchmark")], axis=1,
        join="inner",
    ).dropna().sort_index()
    if len(aligned) < 2:
        raise ValueError("At least two paired observations are required.")
    indices = moving_block_indices(len(aligned), simulations, block_size, seed)
    s = aligned["strategy"].to_numpy()[indices]
    b = aligned["benchmark"].to_numpy()[indices]
    active = s - b

    def annual_return(values: np.ndarray) -> np.ndarray:
        return np.prod(1 + values, axis=1) ** (12 / values.shape[1]) - 1

    def ratio(values: np.ndarray) -> np.ndarray:
        std = values.std(axis=1, ddof=1)
        return np.divide(
            values.mean(axis=1) * np.sqrt(12), std,
            out=np.full(len(values), np.nan), where=~np.isclose(std, 0),
        )

    wealth = np.cumprod(1 + s, axis=1)
    running = np.maximum.accumulate(
        np.concatenate([np.ones((simulations, 1)), wealth], axis=1), axis=1
    )[:, 1:]
    result = pd.DataFrame(
        {
            "annual_return": annual_return(s),
            "annual_volatility": s.std(axis=1, ddof=1) * np.sqrt(12),
            "sharpe": ratio(s),
            "annual_active_return": active.mean(axis=1) * 12,
            "tracking_error": active.std(axis=1, ddof=1) * np.sqrt(12),
            "information_ratio": ratio(active),
            "max_drawdown": (wealth / running - 1).min(axis=1),
            "benchmark_annual_return": annual_return(b),
            "benchmark_sharpe": ratio(b),
        }
    )
    return result


def bootstrap_sampling_stability(
    common_returns: pd.DataFrame,
    simulations: int = 5000,
    block_size: int = 6,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return compact strategy probabilities and metric stability intervals."""
    if "benchmark" not in common_returns:
        raise ValueError("common_returns must contain benchmark.")
    summaries, intervals = [], []
    percentiles = [2.5, 5, 25, 75, 95, 97.5]
    for offset, strategy in enumerate(c for c in common_returns if c != "benchmark"):
        samples = paired_block_bootstrap(
            common_returns[strategy], common_returns["benchmark"], simulations,
            block_size, seed + offset,
        )
        observed_s = common_returns[strategy]
        observed_b = common_returns["benchmark"]
        observed_active = observed_s - observed_b
        observed = {
            "annual_return": (1 + observed_s).prod() ** (12 / len(observed_s)) - 1,
            "annual_volatility": observed_s.std(ddof=1) * np.sqrt(12),
            "sharpe": observed_s.mean() / observed_s.std(ddof=1) * np.sqrt(12),
            "annual_active_return": observed_active.mean() * 12,
            "tracking_error": observed_active.std(ddof=1) * np.sqrt(12),
            "information_ratio": observed_active.mean() / observed_active.std(ddof=1) * np.sqrt(12),
            "max_drawdown": max_drawdown(observed_s),
        }
        for metric in BOOTSTRAP_METRICS:
            values = samples[metric]
            qs = np.percentile(values.dropna(), percentiles)
            intervals.append({
                "strategy": strategy, "metric": metric, "observed": observed[metric],
                "mean": values.mean(), "median": values.median(), "std": values.std(ddof=1),
                "p2_5": qs[0], "p5": qs[1], "p25": qs[2], "p75": qs[3],
                "p95": qs[4], "p97_5": qs[5],
            })
        summaries.append({
            "strategy": strategy, "simulations": simulations, "block_size": block_size,
            "observed_sharpe": observed["sharpe"],
            "median_sharpe": samples["sharpe"].median(),
            "sharpe_p5": samples["sharpe"].quantile(.05),
            "sharpe_p95": samples["sharpe"].quantile(.95),
            "p_annual_return_above_benchmark": float((samples.annual_return > samples.benchmark_annual_return).mean()),
            "p_active_return_positive": float((samples.annual_active_return > 0).mean()),
            "p_sharpe_above_benchmark": float((samples.sharpe > samples.benchmark_sharpe).mean()),
            "p_information_ratio_positive": float((samples.information_ratio > 0).mean()),
            "p_sharpe_above_1": float((samples.sharpe > 1).mean()),
        })
    return pd.DataFrame(summaries), pd.DataFrame(intervals)


def bootstrap_block_sensitivity(
    common_returns: pd.DataFrame,
    simulations: int = 5000,
    block_sizes: tuple[int, ...] = (3, 6, 12),
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for block_size in block_sizes:
        summary, _ = bootstrap_sampling_stability(
            common_returns, simulations=simulations, block_size=block_size, seed=seed
        )
        for row in summary.itertuples(index=False):
            rows.append({
                "strategy": row.strategy, "block_size": block_size,
                "median_sharpe": row.median_sharpe, "sharpe_p5": row.sharpe_p5,
                "sharpe_p95": row.sharpe_p95,
                "p_active_return_positive": row.p_active_return_positive,
                "p_sharpe_above_benchmark": row.p_sharpe_above_benchmark,
            })
    return pd.DataFrame(rows)
