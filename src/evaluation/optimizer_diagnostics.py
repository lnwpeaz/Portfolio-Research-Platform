"""Estimator diagnostics reconstructed without changing optimizer estimates."""

import numpy as np
import pandas as pd
from pypfopt import expected_returns

from src.config import OPTIMIZATION_LOOKBACK_DAYS
from src.evaluation.optimizer_stability import OPTIMIZER_STRATEGIES


def optimizer_estimator_diagnostics(
    backtests: dict[str, pd.DataFrame], prices: pd.DataFrame,
    lookback_days: int = OPTIMIZATION_LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for strategy in OPTIMIZER_STRATEGIES:
        for record in backtests[strategy].itertuples(index=False):
            tickers = list(record.weights)
            history = prices.loc[:pd.Timestamp(record.signal_date), tickers].tail(lookback_days).copy()
            returns = history.pct_change(fill_method=None).dropna(how="any")
            valid = len(history) >= lookback_days and not history.isna().any().any() and len(returns) >= 2
            values = {
                "observations": len(returns), "assets": len(tickers),
                "covariance_condition_number": np.nan, "minimum_eigenvalue": np.nan,
                "maximum_eigenvalue": np.nan, "covariance_rank": np.nan,
                "average_pairwise_correlation": np.nan, "maximum_pairwise_correlation": np.nan,
                "minimum_pairwise_correlation": np.nan,
                "minimum_expected_return": np.nan, "maximum_expected_return": np.nan,
                "expected_return_dispersion": np.nan,
            }
            if valid:
                covariance = returns.cov().to_numpy() * 252
                eigenvalues = np.linalg.eigvalsh(covariance)
                corr = returns.corr().to_numpy()
                pairs = corr[np.triu_indices_from(corr, k=1)]
                values.update({
                    "covariance_condition_number": np.linalg.cond(covariance),
                    "minimum_eigenvalue": eigenvalues.min(), "maximum_eigenvalue": eigenvalues.max(),
                    "covariance_rank": np.linalg.matrix_rank(covariance),
                    "average_pairwise_correlation": pairs.mean(),
                    "maximum_pairwise_correlation": pairs.max(), "minimum_pairwise_correlation": pairs.min(),
                })
                if strategy == "momentum_max_sharpe":
                    mu = expected_returns.mean_historical_return(history, frequency=252)
                    values.update({
                        "minimum_expected_return": mu.min(), "maximum_expected_return": mu.max(),
                        "expected_return_dispersion": mu.std(ddof=1),
                    })
            status = str(record.optimization_status)
            rows.append({
                "strategy": strategy, "date": pd.Timestamp(record.date),
                "signal_date": pd.Timestamp(record.signal_date), **values,
                "optimization_status": status, "objective_result": np.nan,
                "fallback_indicator": status.startswith("fallback_equal:"),
            })
    details = pd.DataFrame(rows)
    summaries = []
    for strategy, group in details.groupby("strategy", sort=False):
        cond = group.covariance_condition_number.dropna()
        summaries.append({
            "strategy": strategy, "valid_estimator_periods": len(cond),
            "median_covariance_condition_number": cond.median(),
            "p95_covariance_condition_number": cond.quantile(.95),
            "fraction_condition_number_above_1e3": (cond > 1e3).mean(),
            "fraction_condition_number_above_1e4": (cond > 1e4).mean(),
            "fraction_condition_number_above_1e5": (cond > 1e5).mean(),
            "fallback_rate": group.fallback_indicator.mean(),
        })
    return details, pd.DataFrame(summaries)
