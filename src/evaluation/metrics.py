import numpy as np
import pandas as pd


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:

    returns = returns.dropna()

    if returns.empty:
        return np.nan

    if (returns <= -1).any():
        return -1.0

    growth = (1 + returns).prod()

    years = (
        len(returns)
        / periods_per_year
    )

    return (
        growth ** (1 / years)
        - 1
    )


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:

    returns = returns.dropna()
    if len(returns) < 2:
        return np.nan
    return (
        returns.std(ddof=1)
        * np.sqrt(periods_per_year)
    )


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:

    returns = returns.dropna()
    excess = (
        returns
        - risk_free_rate
        / periods_per_year
    )

    volatility = excess.std(ddof=1)
    if len(excess) < 2 or not np.isfinite(volatility) or np.isclose(volatility, 0):
        return np.nan

    return (
        excess.mean()
        / volatility
        * np.sqrt(periods_per_year)
    )


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> float:

    returns = returns.dropna()
    excess = (
        returns
        - risk_free_rate
        / periods_per_year
    )

    downside_deviation = np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2))
    if len(excess) == 0 or np.isclose(downside_deviation, 0):
        return np.nan

    return (
        excess.mean()
        / downside_deviation
        * np.sqrt(periods_per_year)
    )


def max_drawdown(
    returns: pd.Series,
) -> float:

    drawdown = drawdown_series(returns)
    return float(drawdown.min()) if not drawdown.empty else np.nan


def drawdown_series(
    returns: pd.Series,
) -> pd.Series:
    """Return drawdowns using initial wealth of 1.0 as the first high-water mark."""

    returns = returns.dropna()
    if returns.empty:
        return pd.Series(dtype=float, name="drawdown")
    wealth = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:]
    return pd.Series(
        wealth.to_numpy() / running_max - 1.0,
        index=returns.index,
        name="drawdown",
    )


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 12,
) -> float:

    ann_return = annualized_return(
        returns,
        periods_per_year=periods_per_year,
    )

    mdd = max_drawdown(
        returns
    )

    if mdd == 0:
        return np.nan

    return (
        ann_return
        / abs(mdd)
    )


def performance_summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 12,
) -> pd.Series:

    return pd.Series({
        "Annual Return":
            annualized_return(
                returns,
                periods_per_year=periods_per_year,
            ),

        "Annual Volatility":
            annualized_volatility(
                returns,
                periods_per_year=periods_per_year,
            ),

        "Sharpe Ratio":
            sharpe_ratio(
                returns,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            ),

        "Sortino Ratio":
            sortino_ratio(
                returns,
                risk_free_rate=risk_free_rate,
                periods_per_year=periods_per_year,
            ),

        "Max Drawdown":
            max_drawdown(
                returns
            ),

        "Calmar Ratio":
            calmar_ratio(
                returns,
                periods_per_year=periods_per_year,
            ),
    })


def active_performance_summary(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = 12,
) -> pd.Series:
    """Return benchmark-relative analytics on their common dates."""
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        return pd.Series(
            {"Annual Active Return": np.nan, "Tracking Error": np.nan,
             "Information Ratio": np.nan, "Beta": np.nan,
             "Annual Alpha": np.nan}
        )

    active = aligned["strategy"] - aligned["benchmark"]
    tracking_error = active.std(ddof=1) * np.sqrt(periods_per_year)
    benchmark_variance = aligned["benchmark"].var(ddof=1)
    beta = (
        aligned["strategy"].cov(aligned["benchmark"]) / benchmark_variance
        if not np.isclose(benchmark_variance, 0)
        else np.nan
    )
    annual_alpha = (
        (
            aligned["strategy"].mean()
            - beta * aligned["benchmark"].mean()
        )
        * periods_per_year
        if np.isfinite(beta)
        else np.nan
    )
    return pd.Series(
        {
            "Annual Active Return": active.mean() * periods_per_year,
            "Tracking Error": tracking_error,
            "Information Ratio": (
                active.mean() / active.std(ddof=1) * np.sqrt(periods_per_year)
                if not np.isclose(active.std(ddof=1), 0)
                else np.nan
            ),
            "Beta": beta,
            "Annual Alpha": annual_alpha,
        }
    )
