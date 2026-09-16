import pandas as pd

from src.data.loader import (
    load_prices,
    load_benchmark,
)
from src.data.frequency import last_observation_by_month

from src.features.ml_dataset import (
    build_ml_dataset,
)

from src.models.walk_forward import (
    generate_walk_forward_scores,
)

from src.backtest.engine import (
    run_score_backtest,
)

from src.evaluation.metrics import (
    active_performance_summary,
    performance_summary,
)

from src.evaluation.diagnostics import (
    backtest_diagnostics,
)
from src.evaluation.ranking_metrics import (
    ranking_summary,
)
from src.evaluation.report import format_backtest_diagnostics
from src.config import (
    EXECUTION_TIMING_MODE,
    ML_MINIMUM_TRAIN_MONTHS,
    ONE_WAY_TURNOVER_COST_BPS,
    TOP_N,
)


from src.evaluation.provenance import record_research_run


@record_research_run
def main():

    print(
        "\n"
        "===================================="
    )

    print(
        " ML Portfolio Research"
    )

    print(
        "===================================="
    )

    prices = (
        load_prices()
    )

    benchmark = (
        load_benchmark()
    )

    print(
        f"\nStocks loaded: "
        f"{prices.shape[1]}"
    )

    # -----------------------------------
    # Dataset
    # -----------------------------------

    print(
        "\nBuilding ML dataset..."
    )

    dataset = (
        build_ml_dataset(
            prices,
            benchmark,
        )
    )

    print(
        f"ML rows: "
        f"{len(dataset):,}"
    )

    print(
        f"Period: "
        f"{dataset['date'].min().date()} "
        f"to "
        f"{dataset['date'].max().date()}"
    )

    print(
        "\nTarget Distribution:"
    )

    print(
        dataset[
            "target"
        ]
        .value_counts(
            normalize=True
        )
        .round(3)
    )

    # -----------------------------------
    # Walk-forward ML
    # -----------------------------------

    print(
        "\nRunning walk-forward ML..."
    )

    scores = (
        generate_walk_forward_scores(
            dataset,
            minimum_train_months=ML_MINIMUM_TRAIN_MONTHS,
        )
    )

    print(
        f"Predictions: "
        f"{len(scores):,}"
    )

    # -----------------------------------
    # Portfolio
    # -----------------------------------

    print(
        "\nRunning ML portfolio..."
    )

    backtest = (
        run_score_backtest(
            scores,
            top_n=TOP_N,
            one_way_turnover_cost_bps=ONE_WAY_TURNOVER_COST_BPS,
            execution_timing=EXECUTION_TIMING_MODE,
        )
    )

    returns = (
        backtest
        .set_index(
            "date"
        )[
            "portfolio_return"
        ]
    )

    # -----------------------------------
    # Performance
    # -----------------------------------

    performance = (
        performance_summary(
            returns
        )
    )

    benchmark_returns = (
        last_observation_by_month(benchmark).pct_change(fill_method=None)
        .reindex(returns.index)
    )
    benchmark_performance = performance_summary(benchmark_returns)
    active_performance = active_performance_summary(
        returns,
        benchmark_returns,
    )

    diagnostics = (
        backtest_diagnostics(
            backtest
        )
    )
    display_diagnostics = format_backtest_diagnostics(
        diagnostics,
        uses_optimizer=False,
    )

    ranking_diagnostics = ranking_summary(
        scores,
        top_n=TOP_N,
    )

    print(
        "\nML Portfolio Performance"
    )

    print(
        performance.round(4)
    )

    print(
        "\nBenchmark Performance"
    )

    print(
        benchmark_performance.round(4)
    )

    print(
        "\nBenchmark-Relative Performance"
    )

    print(
        active_performance.round(4)
    )

    print(
        "\nBacktest Diagnostics"
    )

    print(
        display_diagnostics
    )

    print(
        "\nRanking Diagnostics"
    )

    print(
        "Rank IC: monthly Spearman(score, next-period return); "
        "Rank IC IR: mean(IC) / std(IC), unannualized; "
        "Top-N spread: Top-N minus full scored universe."
    )

    print(
        ranking_diagnostics.round(4)
    )


if __name__ == "__main__":
    main()
