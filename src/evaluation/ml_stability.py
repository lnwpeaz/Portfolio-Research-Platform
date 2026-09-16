"""Stability, decay, and score-dispersion diagnostics for fixed ML scores."""

import numpy as np
import pandas as pd

from src.config import TOP_N
from src.data.frequency import last_observation_by_month
from src.evaluation.ranking_metrics import cross_sectional_rank_correlation


def _monthly_ranking_metrics(
    frame: pd.DataFrame, return_column: str = "future_return", top_n: int = TOP_N
) -> pd.DataFrame:
    rank_ic = cross_sectional_rank_correlation(frame, return_column=return_column)
    rows = []
    for date, group in frame.groupby("date", sort=True):
        valid = group.dropna(subset=["ml_score", return_column]).sort_values(
            ["ml_score", "ticker"], ascending=[False, True]
        )
        if valid.empty:
            continue
        top = valid.head(top_n)
        benchmark_column = "horizon_benchmark_return" if "horizon_benchmark_return" in valid else "benchmark_return"
        rows.append({
            "date": pd.Timestamp(date), "rank_ic": rank_ic.get(date, np.nan),
            "top_n_hit_rate": (top[return_column] > top[benchmark_column]).mean(),
            "top_n_spread": top[return_column].mean() - valid[return_column].mean(),
        })
    return pd.DataFrame(rows)


def ml_ranking_stability(
    scores: pd.DataFrame, rolling_window: int = 12, top_n: int = TOP_N
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = _monthly_ranking_metrics(scores, top_n=top_n)
    yearly = monthly.assign(year=monthly.date.dt.year).groupby("year").agg(
        observations=("rank_ic", "count"), mean_rank_ic=("rank_ic", "mean"),
        median_rank_ic=("rank_ic", "median"), rank_ic_std=("rank_ic", "std"),
        positive_rank_ic_fraction=("rank_ic", lambda x: (x > 0).mean()),
        top_n_hit_rate=("top_n_hit_rate", "mean"),
        average_top_n_return_spread=("top_n_spread", "mean"),
    ).reset_index()
    rolling = monthly.copy()
    rolling["rolling_12m_mean_rank_ic"] = rolling.rank_ic.rolling(rolling_window, min_periods=rolling_window).mean()
    rolling["rolling_12m_positive_ic_fraction"] = rolling.rank_ic.rolling(rolling_window, min_periods=rolling_window).apply(lambda x: (x > 0).mean())
    rolling["rolling_12m_top_n_hit_rate"] = rolling.top_n_hit_rate.rolling(rolling_window, min_periods=rolling_window).mean()
    rolling["rolling_12m_top_n_spread"] = rolling.top_n_spread.rolling(rolling_window, min_periods=rolling_window).mean()
    return yearly, rolling


def ml_signal_decay(
    scores: pd.DataFrame, prices: pd.DataFrame, benchmark_prices: pd.Series,
    horizons: tuple[int, ...] = (1, 2, 3), top_n: int = TOP_N,
) -> pd.DataFrame:
    """Evaluate original time-t scores against cumulative t-to-t+h returns."""
    monthly_prices = last_observation_by_month(prices)
    monthly_benchmark = last_observation_by_month(benchmark_prices)
    rows = []
    for horizon in horizons:
        future = monthly_prices.shift(-horizon) / monthly_prices - 1
        benchmark_future = monthly_benchmark.shift(-horizon) / monthly_benchmark - 1
        stacked = future.rename_axis(index="date", columns="ticker").stack(future_stack=True).rename("horizon_return")
        frame = scores[["date", "ticker", "ml_score"]].copy().set_index(["date", "ticker"])
        frame = frame.join(stacked).reset_index()
        frame["horizon_benchmark_return"] = frame.date.map(benchmark_future)
        monthly = _monthly_ranking_metrics(frame, "horizon_return", top_n)
        std = monthly.rank_ic.std(ddof=1)
        rows.append({
            "horizon_months": horizon, "observations": monthly.rank_ic.count(),
            "mean_rank_ic": monthly.rank_ic.mean(), "median_rank_ic": monthly.rank_ic.median(),
            "rank_ic_std": std, "positive_rank_ic_fraction": (monthly.rank_ic > 0).mean(),
            "rank_ic_ir": monthly.rank_ic.mean() / std if np.isfinite(std) and not np.isclose(std, 0) else np.nan,
            "average_top_n_spread": monthly.top_n_spread.mean(),
            "top_n_hit_rate": monthly.top_n_hit_rate.mean(),
        })
    return pd.DataFrame(rows)


def ml_score_dispersion(
    scores: pd.DataFrame, top_n: int = TOP_N
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking = _monthly_ranking_metrics(scores, top_n=top_n).set_index("date")
    rows = []
    for date, group in scores.groupby("date", sort=True):
        valid = group.dropna(subset=["ml_score"]).sort_values(["ml_score", "ticker"], ascending=[False, True])
        values = valid.ml_score
        rows.append({
            "date": pd.Timestamp(date), "score_mean": values.mean(), "score_std": values.std(ddof=1),
            "score_min": values.min(), "score_max": values.max(), "score_range": values.max() - values.min(),
            "top_1_minus_median_score": values.iloc[0] - values.median(),
            "top_5_average_minus_universe_average": values.head(5).mean() - values.mean(),
            "selected_score_average": values.head(top_n).mean(), "full_universe_score_average": values.mean(),
        })
    details = pd.DataFrame(rows).set_index("date").join(ranking).reset_index()
    summaries = []
    for dispersion in ("score_std", "score_range", "top_1_minus_median_score", "top_5_average_minus_universe_average"):
        summaries.append({
            "dispersion_measure": dispersion,
            "correlation_with_top_n_spread": details[dispersion].corr(details.top_n_spread),
            "correlation_with_rank_ic": details[dispersion].corr(details.rank_ic),
        })
    return details, pd.DataFrame(summaries)
