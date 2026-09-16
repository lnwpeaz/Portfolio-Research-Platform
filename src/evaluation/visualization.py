import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.evaluation.metrics import drawdown_series
from src.evaluation.risk import rolling_beta, rolling_sharpe_ratio


DISPLAY_NAMES = {
    "momentum_3m_equal": "3M momentum · equal weight",
    "momentum_12_1_equal": "12–1 momentum · equal weight",
    "momentum_max_sharpe": "3M momentum · max Sharpe",
    "momentum_min_vol": "3M momentum · min volatility",
    "ml": "ML · equal weight",
    "benchmark": "S&P 500 price index (^GSPC)",
    "S&P 500": "S&P 500 price index (^GSPC)",
}


def _finish_figure(
    output_path: str | Path | None,
    show: bool,
) -> None:
    figure = plt.gcf()
    figure.subplots_adjust(bottom=0.22)
    figure.text(0.5, 0.02,
                "Illustrative ex-post 20-stock universe • survivorship bias • same-close execution\n"
                "Benchmark: S&P 500 price index (^GSPC); stocks use dividend-adjusted prices",
                ha="center", fontsize=9, color="#555555")
    for axis in figure.axes:
        legend = axis.get_legend()
        if legend:
            for label in legend.get_texts():
                label.set_text(DISPLAY_NAMES.get(label.get_text(), label.get_text()))
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()


def plot_strategy_comparison(
    strategies: dict[str, pd.Series],
    benchmark_returns: pd.Series | None = None,
) -> None:

    plt.figure(figsize=(12, 6))

    for name, returns in strategies.items():

        returns = returns.dropna()

        growth = (1 + returns).cumprod()

        plt.plot(
            growth.index,
            growth.values,
            label=name,
        )

    if benchmark_returns is not None:

        benchmark_returns = benchmark_returns.dropna()

        benchmark_growth = (
            1 + benchmark_returns
        ).cumprod()

        plt.plot(
            benchmark_growth.index,
            benchmark_growth.values,
            label="S&P 500",
            linewidth=2,
        )

    plt.title(
        "Portfolio Strategy Comparison"
    )

    plt.xlabel("Date")
    plt.ylabel("Growth of $1")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    _finish_figure(None, True)

def plot_drawdown(
    strategies: dict[str, pd.Series],
) -> None:

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(12, 5)
    )

    for name, returns in strategies.items():

        drawdown = drawdown_series(returns)

        plt.plot(
            drawdown.index,
            drawdown.values,
            label=name,
        )

    plt.title(
        "Portfolio Drawdown"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Drawdown"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


def plot_rolling_sharpe(
    strategies: dict[str, pd.Series],
    window: int = 12,
) -> None:

    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(12, 5)
    )

    for name, returns in strategies.items():

        rolling_mean = (
            returns
            .rolling(window)
            .mean()
        )

        rolling_std = (
            returns
            .rolling(window)
            .std()
        )

        rolling_sharpe = (
            rolling_mean
            / rolling_std
            * np.sqrt(12)
        )

        plt.plot(
            rolling_sharpe.index,
            rolling_sharpe.values,
            label=name,
        )

    plt.title(
        f"Rolling {window}-Month Sharpe Ratio"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Sharpe Ratio"
    )

    plt.legend()

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    plt.show()


def plot_common_period_wealth(
    common_returns: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot cumulative wealth for an already aligned common return panel."""
    (1 + common_returns).cumprod().plot(figsize=(12, 6))
    plt.title("Common-Period Cumulative Wealth")
    plt.xlabel("Realization Date")
    plt.ylabel("Growth of $1")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_common_period_turnover(turnover, output_path=None, show=True):
    """Display recorded turnover without recomputing portfolio accounting."""
    turnover.plot(figsize=(12, 6))
    plt.title("Common-Period One-Way Turnover")
    plt.xlabel("Realization Date (Trade at Prior Signal Close)")
    plt.ylabel("Turnover (Fraction of Portfolio)")
    plt.ylim(bottom=0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_common_period_drawdowns(
    common_returns: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot drawdowns using initial wealth of one for every strategy."""
    plt.figure(figsize=(12, 6))
    for name in common_returns:
        drawdown_series(common_returns[name]).plot(label=name)
    plt.title("Common-Period Drawdowns")
    plt.xlabel("Realization Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_common_period_rolling_sharpe(
    common_returns: pd.DataFrame,
    window: int = 12,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(12, 6))
    for name in common_returns:
        rolling_sharpe_ratio(common_returns[name], window=window).plot(label=name)
    plt.title(f"Rolling {window}-Month Sharpe Ratio")
    plt.xlabel("Realization Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_common_period_rolling_beta(
    common_returns: pd.DataFrame,
    benchmark_name: str = "benchmark",
    window: int = 12,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(12, 6))
    benchmark = common_returns[benchmark_name]
    for name in common_returns:
        if name != benchmark_name:
            rolling_beta(common_returns[name], benchmark, window=window).plot(
                label=name
            )
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--")
    plt.title(f"Rolling {window}-Month Beta vs Benchmark")
    plt.xlabel("Realization Date")
    plt.ylabel("Beta")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_annual_return_comparison(
    annual_returns: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    annual_returns.plot(kind="bar", figsize=(13, 6))
    plt.title("Calendar-Year Returns (First Year May Be Partial)")
    plt.xlabel("Year")
    plt.ylabel("Return")
    plt.axhline(0, color="black", linewidth=1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_rank_ic_timeseries(
    ranking_timeseries: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    frame = ranking_timeseries.set_index("date")
    plt.figure(figsize=(12, 5))
    plt.bar(frame.index, frame["rank_ic"], width=20, alpha=0.5, label="Monthly IC")
    plt.plot(
        frame.index,
        frame["rolling_mean_rank_ic"],
        linewidth=2,
        label="Rolling Mean IC",
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.title("ML Rank IC Through Time")
    plt.xlabel("Signal Date")
    plt.ylabel("Spearman Rank IC")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_concentration_timeseries(
    concentration: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(12, 5))
    for strategy, group in concentration.groupby("strategy"):
        plt.plot(group["date"], group["effective_holdings"], label=strategy)
    plt.title("Effective Number of Holdings Through Time")
    plt.xlabel("Realization Date")
    plt.ylabel("Effective Holdings (1 / HHI)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_contribution_extremes(
    contributions: pd.DataFrame,
    strategy: str,
    n: int = 10,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    securities = contributions[
        contributions["strategy"].eq(strategy)
        & contributions["attribution_type"].eq("security")
    ].sort_values("contribution")
    extremes = pd.concat([securities.head(n), securities.tail(n)]).drop_duplicates(
        "ticker"
    )
    colors = ["#b03a2e" if value < 0 else "#1e8449" for value in extremes["contribution"]]
    plt.figure(figsize=(10, 6))
    plt.barh(extremes["ticker"], extremes["contribution"], color=colors)
    plt.title(f"Top and Bottom Security Contributions: {strategy}")
    plt.xlabel("Sum of Period Return Contributions")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_leave_one_out_sharpe(
    details: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """Plot leave-one-out Sharpe values with strategy baseline references."""
    plt.figure(figsize=(14, 6))
    for strategy, group in details.groupby("strategy", sort=False):
        group = group.sort_values("removed_ticker")
        plt.plot(
            group["removed_ticker"],
            group["sharpe"],
            marker="o",
            label=strategy,
        )
        plt.axhline(
            group["baseline_sharpe"].iloc[0],
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )
    plt.title("Leave-One-Out Universe Sharpe Sensitivity")
    plt.xlabel("Removed Ticker")
    plt.ylabel("Sharpe Ratio")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_subperiod_sharpe(
    performance: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    pivot = performance.pivot(
        index="subperiod", columns="strategy", values="sharpe"
    )
    pivot.plot(kind="bar", figsize=(13, 6))
    plt.title("Sharpe Ratio Across Fixed Calendar Subperiods")
    plt.xlabel("Subperiod")
    plt.ylabel("Sharpe Ratio")
    plt.axhline(0, color="black", linewidth=1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_cost_sensitivity_sharpe(
    sensitivity: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    plt.figure(figsize=(12, 6))
    for strategy, group in sensitivity.groupby("strategy", sort=False):
        group = group.sort_values("cost_bps")
        plt.plot(group["cost_bps"], group["sharpe"], marker="o", label=strategy)
    plt.title("Sharpe Sensitivity to One-Way Turnover Cost")
    plt.xlabel("Cost (bps)")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_execution_sensitivity_sharpe(
    sensitivity: pd.DataFrame,
    output_path: str | Path | None = None,
    show: bool = True,
) -> None:
    pivot = sensitivity.pivot(
        index="strategy", columns="execution_scenario", values="sharpe"
    )
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.title("Same-Close vs One-Period-Delay Sharpe")
    plt.xlabel("Strategy")
    plt.ylabel("Sharpe Ratio")
    plt.axhline(0, color="black", linewidth=1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _finish_figure(output_path, show)


def plot_bootstrap_sharpe_intervals(data, output_path=None, show=True):
    frame = data[data.metric.eq("sharpe")].reset_index(drop=True)
    x = range(len(frame))
    plt.figure(figsize=(11, 6))
    plt.errorbar(x, frame["median"], yerr=[frame["median"]-frame["p5"], frame["p95"]-frame["median"]], fmt="o", capsize=5, label="5th–95th")
    plt.errorbar(x, frame["median"], yerr=[frame["median"]-frame["p25"], frame["p75"]-frame["median"]], fmt="o", linewidth=4, label="25th–75th")
    plt.scatter(x, frame["observed"], marker="x", s=70, color="black", label="Observed")
    plt.xticks(list(x), frame.strategy, rotation=25); plt.ylabel("Sharpe Ratio")
    plt.title("Moving-Block Bootstrap Sharpe Sampling Stability Intervals"); plt.legend(); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    _finish_figure(output_path, show)


def plot_contribution_concentration(data, output_path=None, show=True):
    data.set_index("strategy")[["top_1_share", "top_3_share", "top_5_share"]].plot(kind="bar", figsize=(12, 6))
    plt.title("Share of Positive Contribution by Leading Securities"); plt.ylabel("Share of Positive Contributions"); plt.xticks(rotation=25); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    _finish_figure(output_path, show)


def plot_outlier_dependence(counterfactuals, output_path=None, show=True):
    wanted = counterfactuals[counterfactuals.counterfactual.isin(["baseline", "remove_best_1_months", "remove_best_3_months"])]
    wanted.pivot(index="strategy", columns="counterfactual", values="sharpe").plot(kind="bar", figsize=(12, 6))
    plt.title("Sharpe Dependence on Best Realized Months"); plt.ylabel("Sharpe Ratio"); plt.xticks(rotation=25); plt.grid(axis="y", alpha=.3); plt.tight_layout()
    _finish_figure(output_path, show)


def plot_optimizer_effective_holdings(data, output_path=None, show=True):
    plt.figure(figsize=(12, 5))
    for strategy, group in data.groupby("strategy", sort=False): plt.plot(group.date, group.effective_holdings, label=strategy)
    plt.title("Optimizer Effective Holdings Through Time"); plt.xlabel("Realization Date"); plt.ylabel("1 / HHI"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); _finish_figure(output_path, show)


def plot_optimizer_max_weight(data, output_path=None, show=True):
    plt.figure(figsize=(12, 5))
    for strategy, group in data.groupby("strategy", sort=False): plt.plot(group.date, group.maximum_weight, label=strategy)
    plt.title("Optimizer Maximum Single-Security Weight"); plt.xlabel("Realization Date"); plt.ylabel("Weight"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); _finish_figure(output_path, show)


def plot_ml_rolling_rank_ic(data, output_path=None, show=True):
    plt.figure(figsize=(12, 5)); plt.plot(data.date, data.rolling_12m_mean_rank_ic)
    plt.axhline(0, color="black", linewidth=1); plt.title("ML Rolling 12-Month Mean Rank IC"); plt.xlabel("Signal Date"); plt.ylabel("Rank IC"); plt.grid(alpha=.3); plt.tight_layout(); _finish_figure(output_path, show)


def plot_ml_signal_decay(data, output_path=None, show=True):
    plt.figure(figsize=(8, 5)); plt.plot(data.horizon_months, data.mean_rank_ic, marker="o")
    plt.axhline(0, color="black", linewidth=1); plt.xticks(data.horizon_months); plt.title("ML Signal Decay"); plt.xlabel("Forward Horizon (Months)"); plt.ylabel("Mean Rank IC"); plt.grid(alpha=.3); plt.tight_layout(); _finish_figure(output_path, show)


def plot_rolling_24m_sharpe(data, output_path=None, show=True):
    plt.figure(figsize=(12, 6))
    for strategy, group in data.groupby("strategy", sort=False): plt.plot(group.date, group.sharpe, label=strategy)
    plt.axhline(0, color="black", linewidth=1); plt.title("Rolling 24-Month Sharpe"); plt.xlabel("Realization Date"); plt.ylabel("Sharpe Ratio"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); _finish_figure(output_path, show)
