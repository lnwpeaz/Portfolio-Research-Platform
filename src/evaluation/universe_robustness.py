"""Leave-one-out robustness for the fixed research universe."""

from collections.abc import Callable

import pandas as pd

from src.backtest.engine import run_momentum_backtest, run_score_backtest
from src.config import (
    EXECUTION_TIMING_MODE,
    MAX_POSITION_WEIGHT,
    ML_MINIMUM_TRAIN_MONTHS,
    MOMENTUM_12_1_LOOKBACK_MONTHS,
    MOMENTUM_12_1_SKIP_MONTHS,
    MOMENTUM_LOOKBACK_MONTHS,
    ONE_WAY_TURNOVER_COST_BPS,
    OPTIMIZATION_LOOKBACK_DAYS,
    RISK_FREE_RATE,
    TOP_N,
)
from src.evaluation.research_pipeline import strategy_return_series
from src.evaluation.robustness_common import evaluate_return_series
from src.features.ml_dataset import build_ml_dataset
from src.models.walk_forward import generate_walk_forward_scores


MINIMUM_LOO_STRATEGIES = (
    "momentum_3m_equal",
    "momentum_12_1_equal",
    "ml",
)


def leave_one_out_universes(tickers: list[str]) -> dict[str, list[str]]:
    """Return independent universes, each excluding exactly one ticker."""
    if len(tickers) != len(set(tickers)):
        raise ValueError("Ticker universe contains duplicates.")
    return {
        removed: [ticker for ticker in tickers if ticker != removed]
        for removed in tickers
    }


def run_leave_one_out_strategies(
    prices: pd.DataFrame,
    benchmark: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Run the minimum required strategy set on an experimental price copy."""
    common_momentum = {
        "prices": prices,
        "top_n": TOP_N,
        "weighting": "equal",
        "optimization_lookback_days": OPTIMIZATION_LOOKBACK_DAYS,
        "one_way_turnover_cost_bps": ONE_WAY_TURNOVER_COST_BPS,
        "max_position_weight": MAX_POSITION_WEIGHT,
        "risk_free_rate": RISK_FREE_RATE,
        "execution_timing": EXECUTION_TIMING_MODE,
    }
    results = {
        "momentum_3m_equal": run_momentum_backtest(
            lookback_months=MOMENTUM_LOOKBACK_MONTHS,
            skip_recent_months=0,
            **common_momentum,
        ),
        "momentum_12_1_equal": run_momentum_backtest(
            lookback_months=MOMENTUM_12_1_LOOKBACK_MONTHS,
            skip_recent_months=MOMENTUM_12_1_SKIP_MONTHS,
            **common_momentum,
        ),
    }
    dataset = build_ml_dataset(prices, benchmark)
    scores = generate_walk_forward_scores(
        dataset, minimum_train_months=ML_MINIMUM_TRAIN_MONTHS
    )
    results["ml"] = run_score_backtest(
        scores,
        top_n=TOP_N,
        one_way_turnover_cost_bps=ONE_WAY_TURNOVER_COST_BPS,
        execution_timing=EXECUTION_TIMING_MODE,
    )
    return results


def leave_one_out_analysis(
    prices: pd.DataFrame,
    benchmark_returns: pd.Series,
    benchmark_prices: pd.Series,
    baseline_backtests: dict[str, pd.DataFrame],
    evaluation_dates: pd.Index | None = None,
    strategy_runner: Callable[
        [pd.DataFrame, pd.Series], dict[str, pd.DataFrame]
    ] = run_leave_one_out_strategies,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run and summarize every one-ticker exclusion without mutating inputs."""
    original_tickers = prices.columns.tolist()
    universes = leave_one_out_universes(original_tickers)
    rows = []

    for removed_ticker, experimental_tickers in universes.items():
        experimental_prices = prices.loc[:, experimental_tickers].copy()
        results = strategy_runner(experimental_prices, benchmark_prices.copy())
        unexpected = set(results).difference(MINIMUM_LOO_STRATEGIES)
        if unexpected:
            raise ValueError(f"Unexpected leave-one-out strategies: {unexpected}")

        for strategy in MINIMUM_LOO_STRATEGIES:
            if strategy not in results or strategy not in baseline_backtests:
                raise ValueError(f"Missing leave-one-out strategy: {strategy}")
            experimental_returns = strategy_return_series(results[strategy])
            baseline_returns = strategy_return_series(baseline_backtests[strategy])
            aligned = pd.concat(
                [
                    experimental_returns.rename("experimental"),
                    baseline_returns.rename("baseline"),
                    benchmark_returns.rename("benchmark"),
                ],
                axis=1,
                join="inner",
            ).dropna().sort_index()
            if evaluation_dates is not None:
                aligned = aligned.reindex(evaluation_dates).dropna()
            if aligned.empty or not aligned.index.is_monotonic_increasing:
                raise ValueError("Invalid leave-one-out realization-date alignment.")

            experiment = evaluate_return_series(
                aligned["experimental"], aligned["benchmark"]
            )
            baseline = evaluate_return_series(
                aligned["baseline"], aligned["benchmark"]
            )
            rows.append(
                {
                    "strategy": strategy,
                    "removed_ticker": removed_ticker,
                    **experiment,
                    "baseline_annual_return": baseline["annual_return"],
                    "baseline_sharpe": baseline["sharpe"],
                    "baseline_max_drawdown": baseline["max_drawdown"],
                    "baseline_information_ratio": baseline["information_ratio"],
                    "annual_return_delta": (
                        experiment["annual_return"] - baseline["annual_return"]
                    ),
                    "sharpe_delta": experiment["sharpe"] - baseline["sharpe"],
                    "max_drawdown_delta": (
                        experiment["max_drawdown"] - baseline["max_drawdown"]
                    ),
                    "information_ratio_delta": (
                        experiment["information_ratio"]
                        - baseline["information_ratio"]
                    ),
                }
            )

    details = pd.DataFrame(rows)
    summaries = []
    for strategy, group in details.groupby("strategy", sort=False):
        worst_sharpe = group.loc[group["sharpe"].idxmin()]
        best_sharpe = group.loc[group["sharpe"].idxmax()]
        largest_return_drop = group.loc[group["annual_return_delta"].idxmin()]
        largest_ir_drop = group.loc[group["information_ratio_delta"].idxmin()]
        summaries.append(
            {
                "strategy": strategy,
                "baseline_sharpe": group["baseline_sharpe"].iloc[0],
                "mean_leave_one_out_sharpe": group["sharpe"].mean(),
                "std_leave_one_out_sharpe": group["sharpe"].std(ddof=1),
                "minimum_leave_one_out_sharpe": worst_sharpe["sharpe"],
                "maximum_leave_one_out_sharpe": best_sharpe["sharpe"],
                "worst_removed_ticker_by_sharpe": worst_sharpe["removed_ticker"],
                "best_removed_ticker_by_sharpe": best_sharpe["removed_ticker"],
                "largest_annual_return_drop_ticker": largest_return_drop[
                    "removed_ticker"
                ],
                "largest_information_ratio_drop_ticker": largest_ir_drop[
                    "removed_ticker"
                ],
            }
        )
    return details, pd.DataFrame(summaries)
