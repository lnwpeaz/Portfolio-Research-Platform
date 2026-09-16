"""Generate institutional-style research diagnostics for the strategy suite."""

import pandas as pd

from src.config import ROLLING_RISK_WINDOW_MONTHS
from src.data.loader import load_benchmark, load_prices
from src.evaluation.attribution import contribution_by_ticker
from src.evaluation.concentration import (
    concentration_summary,
    concentration_timeseries,
)
from src.evaluation.drawdown import drawdown_summary
from src.evaluation.ranking_metrics import (
    ranking_diagnostics_timeseries,
    ranking_summary,
)
from src.evaluation.report import (
    REPORT_DIR,
    common_period_performance_report,
    common_period_returns,
    common_period_yearly_returns,
    save_performance_report,
)
from src.evaluation.research_pipeline import (
    run_strategy_suite,
    strategy_return_series,
)
from src.evaluation.risk import rolling_risk_report
from src.evaluation.visualization import (
    plot_annual_return_comparison,
    plot_common_period_drawdowns,
    plot_common_period_rolling_beta,
    plot_common_period_rolling_sharpe,
    plot_common_period_wealth,
    plot_common_period_turnover,
    plot_concentration_timeseries,
    plot_contribution_extremes,
    plot_rank_ic_timeseries,
)
from src.evaluation.yearly_analysis import common_period_yearly_analysis


from src.evaluation.provenance import record_research_run


@record_research_run
def main() -> None:
    prices = load_prices()
    benchmark = load_benchmark()
    backtests, scores, benchmark_returns = run_strategy_suite(prices, benchmark)
    strategies = {
        name: strategy_return_series(result)
        for name, result in backtests.items()
    }
    common = common_period_returns(strategies, benchmark_returns)
    save_performance_report(common, "common_period_returns.csv")

    performance = common_period_performance_report(common)
    yearly_returns = common_period_yearly_returns(common)
    yearly_analysis = common_period_yearly_analysis(common)
    rolling = rolling_risk_report(
        common,
        window=ROLLING_RISK_WINDOW_MONTHS,
    )

    drawdowns = pd.DataFrame(
        {name: drawdown_summary(common[name]) for name in common.columns}
    ).T
    drawdowns.index.name = "strategy"

    concentration_frames = []
    concentration_summaries = {}
    for name, backtest in backtests.items():
        selected = backtest[backtest["date"].isin(common.index)]
        timeseries = concentration_timeseries(selected)
        if not timeseries.empty:
            timeseries.insert(0, "strategy", name)
            concentration_frames.append(timeseries)
        concentration_summaries[name] = concentration_summary(selected)
    concentration = pd.concat(concentration_frames, ignore_index=True)
    concentration_report = pd.DataFrame(concentration_summaries).T
    concentration_report.index.name = "strategy"

    attribution = pd.concat(
        [
            contribution_by_ticker(backtest, name, dates=common.index)
            for name, backtest in backtests.items()
        ],
        ignore_index=True,
    )
    turnover = pd.DataFrame({
        name: result.set_index("date")["turnover"].reindex(common.index)
        for name, result in backtests.items()
    })
    save_performance_report(turnover, "common_period_turnover.csv")
    plot_common_period_turnover(turnover, REPORT_DIR / "common_period_turnover.png", show=False)
    ranking_timeseries = ranking_diagnostics_timeseries(scores)
    ranking = ranking_summary(scores)

    print(
        f"\nCommon research period: {common.index.min().date()} to "
        f"{common.index.max().date()} ({len(common)} months)\n"
    )
    print("Common-Period Performance\n")
    print(performance.round(4))
    print("\nDrawdown Summary\n")
    print(drawdowns)
    print("\nConcentration Summary\n")
    print(concentration_report.round(4))
    print("\nML Ranking Summary\n")
    print(ranking.round(4))
    print("\nTop and Bottom Contributors\n")
    for name in backtests:
        securities = attribution[
            attribution["strategy"].eq(name)
            & attribution["attribution_type"].eq("security")
        ].sort_values("contribution")
        print(f"\n{name} — bottom 10")
        print(securities.head(10)[["ticker", "contribution"]].to_string(index=False))
        print(f"{name} — top 10")
        print(securities.tail(10).sort_values("contribution", ascending=False)[
            ["ticker", "contribution"]
        ].to_string(index=False))

    save_performance_report(performance, "common_period_performance.csv")
    save_performance_report(yearly_returns, "common_period_yearly_returns.csv")
    save_performance_report(yearly_analysis, "common_period_yearly_analysis.csv")
    save_performance_report(rolling, "rolling_risk_metrics.csv", index=False)
    save_performance_report(
        attribution, "contribution_by_ticker.csv", index=False
    )
    save_performance_report(drawdowns, "drawdown_summary.csv")
    save_performance_report(
        ranking_timeseries,
        "ranking_diagnostics_timeseries.csv",
        index=False,
    )
    save_performance_report(concentration_report, "concentration_summary.csv")
    save_performance_report(
        concentration, "concentration_timeseries.csv", index=False
    )

    plot_common_period_wealth(
        common, REPORT_DIR / "common_period_wealth.png", show=False
    )
    plot_common_period_drawdowns(
        common, REPORT_DIR / "common_period_drawdowns.png", show=False
    )
    plot_common_period_rolling_sharpe(
        common,
        window=ROLLING_RISK_WINDOW_MONTHS,
        output_path=REPORT_DIR / "rolling_sharpe.png",
        show=False,
    )
    plot_common_period_rolling_beta(
        common,
        window=ROLLING_RISK_WINDOW_MONTHS,
        output_path=REPORT_DIR / "rolling_beta.png",
        show=False,
    )
    plot_annual_return_comparison(
        yearly_returns,
        REPORT_DIR / "annual_return_comparison.png",
        show=False,
    )
    plot_rank_ic_timeseries(
        ranking_timeseries,
        REPORT_DIR / "ranking_ic_timeseries.png",
        show=False,
    )
    plot_concentration_timeseries(
        concentration,
        REPORT_DIR / "concentration_timeseries.png",
        show=False,
    )
    for name in backtests:
        plot_contribution_extremes(
            attribution,
            name,
            output_path=REPORT_DIR / f"contribution_{name}.png",
            show=False,
        )


if __name__ == "__main__":
    main()
