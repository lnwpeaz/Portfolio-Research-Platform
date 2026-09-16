"""Cross-sectional diagnostics for ranking models."""

import numpy as np
import pandas as pd


def cross_sectional_rank_correlation(
    scores: pd.DataFrame,
    score_column: str = "ml_score",
    return_column: str = "future_return",
) -> pd.Series:
    """Monthly Spearman correlation of score with next-period return."""
    required = {"date", score_column, return_column}
    missing = required.difference(scores.columns)
    if missing:
        raise ValueError(f"Scores are missing columns: {sorted(missing)}")

    def correlation(group: pd.DataFrame) -> float:
        valid = group[[score_column, return_column]].dropna()
        if len(valid) < 2:
            return np.nan
        return valid[score_column].corr(valid[return_column], method="spearman")

    return scores.groupby("date", sort=True).apply(
        correlation, include_groups=False
    ).rename("rank_ic")


def ranking_summary(
    scores: pd.DataFrame,
    top_n: int = 5,
    score_column: str = "ml_score",
) -> pd.Series:
    """Summarize fixed ranking diagnostics.

    Rank IC is the monthly Spearman correlation between model score and the
    next-period return. Rank IC IR is ``mean(IC) / std(IC)`` and is not
    annualized. Top-N Return Spread is the Top-N mean return minus the mean
    return of the full scored universe.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    information_coefficients = cross_sectional_rank_correlation(
        scores, score_column=score_column
    )
    hit_rates = []
    spreads = []
    for _, group in scores.groupby("date", sort=True):
        ranked = group.dropna(subset=[score_column]).sort_values(
            [score_column, "ticker"], ascending=[False, True]
        )
        if ranked.empty or ranked["future_return"].isna().all():
            continue
        if ranked[["future_return", "target"]].isna().any().any():
            raise ValueError(
                "Partially missing realized outcomes make ranking diagnostics "
                "conditional on future availability."
            )
        top = ranked.head(top_n)
        hit_rates.append(top["target"].astype(float).mean())
        spreads.append(top["future_return"].mean() - ranked["future_return"].mean())

    ic_std = information_coefficients.std(ddof=1)
    return pd.Series(
        {
            "Mean Rank IC": information_coefficients.mean(),
            "Rank IC Information Ratio": (
                information_coefficients.mean() / ic_std
                if np.isfinite(ic_std) and not np.isclose(ic_std, 0)
                else np.nan
            ),
            "Top-N Hit Rate": np.mean(hit_rates) if hit_rates else np.nan,
            "Top-N Return Spread": np.mean(spreads) if spreads else np.nan,
            "Evaluated Periods": len(hit_rates),
            "Positive Rank IC Fraction": (
                float((information_coefficients.dropna() > 0).mean())
                if information_coefficients.notna().any()
                else np.nan
            ),
        }
    )


def ranking_diagnostics_timeseries(
    scores: pd.DataFrame,
    top_n: int = 5,
    rolling_window: int = 12,
    score_column: str = "ml_score",
) -> pd.DataFrame:
    """Return monthly Rank IC, rolling mean IC, and Top-N hit rate.

    Rank IC remains the monthly Spearman correlation between score and
    next-period return. The rolling series is a simple 12-month mean by default.
    Top-N hit rate is the fraction of selected names with ``target == 1``.
    """
    if top_n < 1 or rolling_window < 1:
        raise ValueError("top_n and rolling_window must be positive.")
    rank_ic = cross_sectional_rank_correlation(scores, score_column=score_column)
    hit_rate = {}
    for date, group in scores.groupby("date", sort=True):
        ranked = group.dropna(subset=[score_column]).sort_values(
            [score_column, "ticker"], ascending=[False, True]
        )
        if ranked.empty or ranked["target"].isna().all():
            hit_rate[date] = np.nan
            continue
        top = ranked.head(top_n)
        if top["target"].isna().any():
            raise ValueError("Top-N outcomes are partially missing.")
        hit_rate[date] = top["target"].astype(float).mean()

    result = pd.DataFrame(
        {
            "rank_ic": rank_ic,
            "rolling_mean_rank_ic": rank_ic.rolling(
                rolling_window, min_periods=rolling_window
            ).mean(),
            "top_n_hit_rate": pd.Series(hit_rate, dtype=float),
        }
    )
    result.index.name = "date"
    return result.reset_index()
