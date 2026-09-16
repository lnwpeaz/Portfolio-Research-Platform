"""Large-universe ranking, quantile, sector, and Top-N diagnostics."""

import numpy as np
import pandas as pd
from scipy import stats

from src.backtest.engine import run_score_backtest
from src.evaluation.metrics import performance_summary


REQUIRED_SCORE_COLUMNS = {"date", "ticker", "ml_score", "future_return"}


def _validate_scores(scores):
    missing = REQUIRED_SCORE_COLUMNS.difference(scores.columns)
    if missing:
        raise ValueError(f"Scores are missing columns: {sorted(missing)}")
    if scores.duplicated(["date", "ticker"]).any():
        raise ValueError("Each security may appear only once per prediction date.")


def cross_sectional_diagnostics(scores, top_n=25):
    _validate_scores(scores)
    rows = []
    for date, group in scores.groupby("date", sort=True):
        valid_scores = group.dropna(subset=["ml_score"])
        realized = valid_scores.dropna(subset=["future_return"])
        rows.append(
            {
                "date": date,
                "spearman_rank_ic": realized.ml_score.corr(realized.future_return, method="spearman") if len(realized) >= 2 else np.nan,
                "pearson_ic": realized.ml_score.corr(realized.future_return, method="pearson") if len(realized) >= 2 else np.nan,
                "positive_rank_ic": np.nan,
                "score_dispersion": valid_scores.ml_score.std(ddof=1),
                "eligible_securities": len(valid_scores),
                "selected_securities": min(top_n, len(valid_scores)),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["positive_rank_ic"] = result.spearman_rank_ic.gt(0).where(result.spearman_rank_ic.notna())
    return result


def assign_score_quantiles(scores, quantiles=5):
    """Deterministic equal-count assignment from time-t scores only."""
    _validate_scores(scores)
    parts = []
    for _, group in scores.groupby("date", sort=True):
        valid = group.dropna(subset=["ml_score"]).copy()
        valid = valid.sort_values(["ml_score", "ticker"], ascending=[True, True])
        n = len(valid)
        if n == 0:
            continue
        # Integer ranks avoid qcut failures when model scores are tied.
        valid["quantile_number"] = np.floor(np.arange(n) * quantiles / n).astype(int) + 1
        valid["quantile"] = "Q" + valid.quantile_number.astype(str)
        parts.append(valid)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def quantile_returns(scores):
    assigned = assign_score_quantiles(scores)
    realized = assigned.dropna(subset=["future_return"])
    result = realized.pivot_table(
        index="date", columns="quantile", values="future_return", aggfunc="mean"
    ).reindex(columns=[f"Q{i}" for i in range(1, 6)])
    result["Q5_minus_Q1"] = result.Q5 - result.Q1
    return result.reset_index()


def quantile_summary(returns, periods_per_year=12):
    rows = []
    for column in ["Q1", "Q2", "Q3", "Q4", "Q5", "Q5_minus_Q1"]:
        summary = performance_summary(
            returns.set_index("date")[column], periods_per_year=periods_per_year
        )
        rows.append({"portfolio": column, **summary.to_dict()})
    return pd.DataFrame(rows)


def quantile_monotonicity(returns):
    columns = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    complete = returns.dropna(subset=columns)
    spread = complete.Q5 - complete.Q1
    record = {
        "Q5_gt_Q1_fraction": float((spread > 0).mean()) if len(spread) else np.nan,
        "average_Q5_minus_Q1": spread.mean(),
        "median_Q5_minus_Q1": spread.median(),
        "monotonic_period_fraction": float(
            complete[columns].apply(lambda row: row.is_monotonic_increasing, axis=1).mean()
        ) if len(complete) else np.nan,
    }
    record.update({f"average_{column}": complete[column].mean() for column in columns})
    return pd.DataFrame([record])


def long_short_diagnostic(returns, periods_per_year=12):
    spread = returns["Q5_minus_Q1"].dropna()
    summary = performance_summary(spread, periods_per_year=periods_per_year)
    sem = stats.sem(spread, nan_policy="omit") if len(spread) >= 2 else np.nan
    return pd.DataFrame([{
        "series": "research_long_short_spread",
        "mean_monthly_spread": spread.mean(),
        "annualized_return": summary["Annual Return"],
        "annualized_volatility": summary["Annual Volatility"],
        "sharpe": summary["Sharpe Ratio"],
        "positive_month_fraction": float((spread > 0).mean()) if len(spread) else np.nan,
        "naive_iid_t_stat_descriptive_only": spread.mean() / sem if np.isfinite(sem) and sem > 0 else np.nan,
        "warning": "IID t-statistic is descriptive only and does not adjust for autocorrelation.",
    }])


def sector_exposure(scores, membership, top_n=25):
    _validate_scores(scores)
    metadata_columns = ["ticker", "sector"]
    dated = {"effective_from", "effective_to"}.issubset(membership.columns)
    if dated:
        metadata_columns.extend(["effective_from", "effective_to"])
    metadata = membership[metadata_columns].copy()
    metadata["sector"] = metadata.sector.fillna("Unknown").replace("", "Unknown")
    merged = scores.merge(metadata, on="ticker", how="left")
    if dated:
        merged["effective_from"] = pd.to_datetime(merged.effective_from)
        merged["effective_to"] = pd.to_datetime(merged.effective_to)
        active = merged.effective_from.le(merged.date) & (
            merged.effective_to.isna() | merged.effective_to.gt(merged.date)
        )
        merged = merged.loc[active, scores.columns.tolist() + ["sector"]]
        if merged.duplicated(["date", "ticker"]).any():
            raise ValueError("Overlapping membership intervals create duplicate sector rows.")
    merged["sector"] = merged.sector.fillna("Unknown")
    quantiled = assign_score_quantiles(merged)
    rows = []
    for date, group in quantiled.groupby("date", sort=True):
        ranked = group.sort_values(["ml_score", "ticker"], ascending=[False, True])
        top = ranked.head(top_n)
        q5 = ranked[ranked["quantile"].eq("Q5")]
        sectors = sorted(set(ranked.sector) | set(top.sector) | set(q5.sector))
        for sector in sectors:
            universe_weight = float(ranked.sector.eq(sector).mean())
            top_weight = float(top.sector.eq(sector).mean()) if len(top) else 0.0
            q5_weight = float(q5.sector.eq(sector).mean()) if len(q5) else 0.0
            rows.append({
                "date": date, "sector": sector,
                "universe_weight": universe_weight, "topn_weight": top_weight,
                "q5_weight": q5_weight,
                "active_sector_weight": top_weight - universe_weight,
                "weighting": "equal_weight",
            })
    return pd.DataFrame(rows)


def sector_attribution_summary(exposure, scores, top_n=25):
    if exposure.empty:
        return pd.DataFrame()
    mean_active = exposure.groupby("sector").active_sector_weight.mean().sort_values()
    per_date = exposure.groupby("date").active_sector_weight.apply(lambda x: x.abs().sum() / 2)
    spreads, concentrations = [], []
    for date, group in scores.groupby("date", sort=True):
        ranked = group.dropna(subset=["ml_score", "future_return"]).sort_values(
            ["ml_score", "ticker"], ascending=[False, True]
        )
        if ranked.empty:
            continue
        spreads.append((date, ranked.head(top_n).future_return.mean() - ranked.future_return.mean()))
        concentrations.append((date, per_date.get(date, np.nan)))
    joined = pd.DataFrame(spreads, columns=["date", "topn_return_spread"]).merge(
        pd.DataFrame(concentrations, columns=["date", "sector_concentration"]), on="date"
    )
    return pd.DataFrame([{
        "average_absolute_sector_active_exposure": exposure.active_sector_weight.abs().mean(),
        "average_sector_concentration_half_l1": per_date.mean(),
        "most_persistently_overweight_sector": mean_active.index[-1],
        "most_persistently_underweight_sector": mean_active.index[0],
        "concentration_topn_spread_correlation": joined.sector_concentration.corr(joined.topn_return_spread),
        "interpretation_warning": "Diagnostic association only; it does not establish sector causality.",
    }])


def topn_sensitivity(scores, values=(10, 20, 25, 50), cost_bps=10):
    rows = []
    for top_n in values:
        backtest = run_score_backtest(scores, top_n=top_n, one_way_turnover_cost_bps=cost_bps)
        if backtest.empty:
            continue
        summary = performance_summary(backtest.set_index("date").portfolio_return)
        holdings_count = backtest.holdings.map(len)
        hhi = backtest.weights.map(lambda weights: sum(weight * weight for weight in weights.values()))
        rows.append({
            "top_n": top_n,
            "annual_return": summary["Annual Return"],
            "annual_volatility": summary["Annual Volatility"],
            "sharpe": summary["Sharpe Ratio"],
            "average_turnover": backtest.turnover.mean(),
            "average_effective_holdings": (1 / hhi).mean(),
            "average_holdings": holdings_count.mean(),
            "average_hhi_concentration": hhi.mean(),
            "purpose": "diagnostic_not_model_selection",
        })
    return pd.DataFrame(rows)
