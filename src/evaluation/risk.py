"""Reusable rolling risk analytics for monthly strategy returns."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import max_drawdown


def rolling_annualized_return(
    returns: pd.Series,
    window: int = 12,
    periods_per_year: int = 12,
) -> pd.Series:
    """Geometrically annualized return over each complete rolling window."""
    if window < 1:
        raise ValueError("window must be positive.")

    def annualize(values: np.ndarray) -> float:
        if np.any(values <= -1):
            return -1.0
        return float(np.prod(1 + values) ** (periods_per_year / len(values)) - 1)

    return returns.rolling(window, min_periods=window).apply(annualize, raw=True)


def rolling_annualized_volatility(
    returns: pd.Series,
    window: int = 12,
    periods_per_year: int = 12,
) -> pd.Series:
    """Annualized sample volatility over complete rolling windows."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(
        periods_per_year
    )


def rolling_sharpe_ratio(
    returns: pd.Series,
    window: int = 12,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> pd.Series:
    """Annualized rolling Sharpe ratio using a constant annual risk-free rate."""
    excess = returns - risk_free_rate / periods_per_year
    mean = excess.rolling(window, min_periods=window).mean()
    volatility = excess.rolling(window, min_periods=window).std(ddof=1)
    result = mean / volatility * np.sqrt(periods_per_year)
    return result.mask(np.isclose(volatility, 0))


def rolling_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 12,
) -> pd.Series:
    """Rolling covariance beta versus a benchmark on common dates."""
    aligned = pd.concat(
        [returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    )
    covariance = aligned["strategy"].rolling(window, min_periods=window).cov(
        aligned["benchmark"]
    )
    variance = aligned["benchmark"].rolling(window, min_periods=window).var(ddof=1)
    return (covariance / variance).mask(np.isclose(variance, 0)).rename("beta")


def rolling_tracking_error(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 12,
    periods_per_year: int = 12,
) -> pd.Series:
    """Annualized volatility of active returns over common dates."""
    active = returns.subtract(benchmark_returns, fill_value=np.nan)
    return active.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(
        periods_per_year
    )


def rolling_information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    window: int = 12,
    periods_per_year: int = 12,
) -> pd.Series:
    """Annualized mean active return divided by active-return volatility."""
    active = returns.subtract(benchmark_returns, fill_value=np.nan)
    mean = active.rolling(window, min_periods=window).mean()
    volatility = active.rolling(window, min_periods=window).std(ddof=1)
    result = mean / volatility * np.sqrt(periods_per_year)
    return result.mask(np.isclose(volatility, 0))


def rolling_max_drawdown(
    returns: pd.Series,
    window: int = 12,
) -> pd.Series:
    """Maximum drawdown within each complete rolling return window."""
    return returns.rolling(window, min_periods=window).apply(
        lambda values: max_drawdown(pd.Series(values)),
        raw=True,
    )


def rolling_risk_report(
    common_returns: pd.DataFrame,
    benchmark_name: str = "benchmark",
    window: int = 12,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Return tidy rolling risk metrics for an aligned strategy panel."""
    if benchmark_name not in common_returns:
        raise ValueError(f"Missing benchmark column: {benchmark_name!r}.")
    benchmark = common_returns[benchmark_name]
    frames = []
    for strategy in common_returns.columns:
        returns = common_returns[strategy]
        frame = pd.DataFrame(
            {
                "annualized_return": rolling_annualized_return(
                    returns, window, periods_per_year
                ),
                "annualized_volatility": rolling_annualized_volatility(
                    returns, window, periods_per_year
                ),
                "sharpe_ratio": rolling_sharpe_ratio(
                    returns, window, periods_per_year=periods_per_year
                ),
                "beta": rolling_beta(returns, benchmark, window),
                "tracking_error": rolling_tracking_error(
                    returns, benchmark, window, periods_per_year
                ),
                "information_ratio": rolling_information_ratio(
                    returns, benchmark, window, periods_per_year
                ),
                "maximum_drawdown": rolling_max_drawdown(returns, window),
            }
        )
        frame.insert(0, "strategy", strategy)
        frames.append(frame.rename_axis("date").reset_index())
    return pd.concat(frames, ignore_index=True)
