"""Portfolio backtest engines with explicit signal and realization dates."""

import numpy as np
import pandas as pd

from pypfopt.exceptions import InstantiationError, OptimizationError

from src.backtest.costs import (
    calculate_one_way_transaction_cost,
    calculate_turnover,
)
from src.data.frequency import last_observation_by_month
from src.features.momentum import calculate_monthly_momentum_signal
from src.portfolio.constraints import cap_weights, validate_weights
from src.portfolio.optimizer import (
    equal_weight_weights,
    max_sharpe_weights,
    min_volatility_weights,
)
from src.portfolio.selector import select_top_n


SUPPORTED_EXECUTION_TIMING_MODES = {"same_close"}


def _validate_execution_timing(execution_timing: str) -> None:
    if execution_timing not in SUPPORTED_EXECUTION_TIMING_MODES:
        raise ValueError(
            f"Unsupported execution_timing={execution_timing!r}. "
            "Only 'same_close' is currently implemented consistently across "
            "momentum and ML backtests."
        )


def _validate_prices(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("Price data is empty.")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError("Price data must have a DatetimeIndex.")
    if prices.index.has_duplicates or prices.columns.has_duplicates:
        raise ValueError("Price dates and asset names must be unique.")
    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive where observed.")
    return prices.sort_index().astype(float)


def _drift_weights(
    starting_weights: pd.Series,
    asset_returns: pd.Series,
) -> pd.Series:
    """Return pre-trade weights after assets realize one holding-period return."""
    aligned_returns = asset_returns.reindex(starting_weights.index)
    if aligned_returns.isna().any():
        missing = aligned_returns[aligned_returns.isna()].index.tolist()
        raise ValueError(f"Missing realized returns for selected assets: {missing}")

    ending_values = starting_weights * (1.0 + aligned_returns)
    ending_total = float(ending_values.sum())
    if not np.isfinite(ending_total) or ending_total <= 0:
        raise ValueError("Portfolio value is non-positive after realized returns.")
    return ending_values / ending_total


def _construct_weights(
    weighting: str,
    selected: list[str],
    historical_prices: pd.DataFrame,
    max_position_weight: float,
    minimum_observations: int,
    risk_free_rate: float,
) -> tuple[pd.Series, str]:
    status = "optimized"
    if weighting == "equal":
        weights = equal_weight_weights(selected)
        status = "equal"
    elif weighting == "max_sharpe":
        if len(historical_prices) < minimum_observations:
            weights = equal_weight_weights(selected)
            status = "fallback_equal:insufficient_history"
        elif historical_prices.isna().any().any():
            weights = equal_weight_weights(selected)
            status = "fallback_equal:missing_data"
        else:
            try:
                weights = max_sharpe_weights(
                    historical_prices,
                    risk_free_rate=risk_free_rate,
                    max_weight=max_position_weight,
                )
            except (
                OptimizationError,
                InstantiationError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                weights = equal_weight_weights(selected)
                status = "fallback_equal:optimizer_failure"
    elif weighting == "min_vol":
        if len(historical_prices) < minimum_observations:
            weights = equal_weight_weights(selected)
            status = "fallback_equal:insufficient_history"
        elif historical_prices.isna().any().any():
            weights = equal_weight_weights(selected)
            status = "fallback_equal:missing_data"
        else:
            try:
                weights = min_volatility_weights(
                    historical_prices,
                    max_weight=max_position_weight,
                )
            except (
                OptimizationError,
                InstantiationError,
                ValueError,
                np.linalg.LinAlgError,
            ):
                weights = equal_weight_weights(selected)
                status = "fallback_equal:optimizer_failure"
    else:
        raise ValueError(f"Unknown weighting method: {weighting}")

    # An optimizer may exclude a column with an incomplete history.  Never
    # turn that historical-data decision into an implicit concentrated bet.
    if set(weights.index) != set(selected):
        weights = equal_weight_weights(selected)
        status = "fallback_equal:missing_data"

    weights = weights.reindex(selected)
    weights = cap_weights(weights, max_weight=max_position_weight)
    validate_weights(weights, max_weight=max_position_weight)
    return weights, status


def run_momentum_backtest(
    prices: pd.DataFrame,
    lookback_months: int = 3,
    top_n: int = 5,
    weighting: str = "equal",
    optimization_lookback_days: int = 252,
    one_way_turnover_cost_bps: float = 10,
    max_position_weight: float = 0.40,
    risk_free_rate: float = 0.0,
    minimum_optimization_observations: int | None = None,
    execution_timing: str = "same_close",
    skip_recent_months: int = 0,
) -> pd.DataFrame:
    """Run a monthly close-to-close momentum backtest.

    ``signal_date`` is the month-end close used to form the portfolio. ``date``
    is the following month end when its return is realized. This convention
    assumes execution at the signal close; ``execution_timing`` records and
    validates that assumption. Live use should form the signal before the
    market-on-close cutoff or introduce an execution lag with suitable prices.
    """
    prices = _validate_prices(prices)
    _validate_execution_timing(execution_timing)
    if lookback_months < 1 or top_n < 1 or optimization_lookback_days < 2:
        raise ValueError("Lookbacks and top_n must be positive.")

    if minimum_optimization_observations is None:
        minimum_optimization_observations = optimization_lookback_days
    if minimum_optimization_observations < 2:
        raise ValueError("minimum_optimization_observations must be at least 2.")

    monthly_prices = last_observation_by_month(prices)
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    momentum = calculate_monthly_momentum_signal(
        monthly_prices,
        lookback_months=lookback_months,
        skip_recent_months=skip_recent_months,
    )

    results: list[dict] = []
    pretrade_weights: pd.Series | None = None

    for position in range(len(momentum.index) - 1):
        signal_date = momentum.index[position]
        realization_date = momentum.index[position + 1]
        selected = select_top_n(momentum.iloc[position], n=top_n)
        if not selected:
            continue
        if len(selected) * max_position_weight < 1 - 1e-12:
            raise ValueError(
                "Position cap is infeasible for the selected portfolio: "
                f"{len(selected)} holdings at max {max_position_weight:.4f}."
            )

        # Selection and weights use only information through signal_date.
        historical_prices = prices.loc[:signal_date, selected].tail(
            optimization_lookback_days
        )
        weights, optimization_status = _construct_weights(
            weighting,
            selected,
            historical_prices,
            max_position_weight,
            minimum_optimization_observations,
            risk_free_rate,
        )

        # Check outcomes only after the portfolio has been fixed. Dropping a
        # name here would use future data to alter the ex-ante portfolio.
        realized_returns = monthly_returns.loc[realization_date, selected]
        if realized_returns.isna().any():
            missing = realized_returns[realized_returns.isna()].index.tolist()
            raise ValueError(
                "Missing next-period returns for selected assets; provide "
                f"delisting-aware data or an explicit return policy: {missing}"
            )

        turnover = calculate_turnover(pretrade_weights, weights)
        transaction_cost = calculate_one_way_transaction_cost(
            turnover,
            one_way_turnover_cost_bps=one_way_turnover_cost_bps,
        )
        gross_return = float((realized_returns * weights).sum())
        net_return = gross_return - transaction_cost

        results.append(
            {
                "signal_date": signal_date,
                "date": realization_date,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "turnover": turnover,
                "portfolio_return": net_return,
                "holdings": selected,
                "weights": weights.to_dict(),
                "asset_returns": realized_returns.to_dict(),
                "weighting": weighting,
                "momentum_lookback_months": lookback_months,
                "momentum_skip_recent_months": skip_recent_months,
                "optimization_status": optimization_status,
                "optimization_fallback_reason": (
                    optimization_status.partition(":")[2]
                    if optimization_status.startswith("fallback_equal:")
                    else None
                ),
                "execution_timing": execution_timing,
            }
        )
        pretrade_weights = _drift_weights(weights, realized_returns)

    return pd.DataFrame(results)


def run_score_backtest(
    scores: pd.DataFrame,
    top_n: int = 5,
    one_way_turnover_cost_bps: float = 10,
    execution_timing: str = "same_close",
) -> pd.DataFrame:
    """Backtest equal-weight scores without conditioning on future availability.

    Required columns are ``date`` (signal date), ``ticker``, ``ml_score``,
    ``future_return`` and ``target_date`` (return realization date).
    Entirely unlabeled terminal dates are skipped. A partially missing return
    among selected names raises because silently replacing the name is leakage.
    ``execution_timing='same_close'`` makes the current research assumption
    explicit; no lagged mode is implemented without suitable execution prices.
    """
    required = {"date", "target_date", "ticker", "ml_score", "future_return"}
    missing_columns = required.difference(scores.columns)
    if missing_columns:
        raise ValueError(f"Score data is missing columns: {sorted(missing_columns)}")
    if top_n < 1:
        raise ValueError("top_n must be positive.")
    _validate_execution_timing(execution_timing)
    if scores.duplicated(["date", "ticker"]).any():
        raise ValueError("Score data contains duplicate date/ticker rows.")

    results: list[dict] = []
    pretrade_weights: pd.Series | None = None

    for signal_date, cross_section in scores.groupby("date", sort=True):
        ranked = cross_section.dropna(subset=["ml_score"]).sort_values(
            ["ml_score", "ticker"], ascending=[False, True]
        )
        selected_rows = ranked.head(top_n).copy()
        if selected_rows.empty:
            continue
        if cross_section["future_return"].isna().all():
            # The final signal can be scored live but cannot yet be evaluated.
            continue
        if selected_rows["future_return"].isna().any():
            missing = selected_rows.loc[
                selected_rows["future_return"].isna(), "ticker"
            ].tolist()
            raise ValueError(
                "Missing next-period returns for selected assets; provide "
                f"delisting-aware data or an explicit return policy: {missing}"
            )
        target_dates = selected_rows["target_date"].dropna().unique()
        if len(target_dates) != 1:
            raise ValueError("Selected rows must share one non-null target_date.")

        holdings = selected_rows["ticker"].tolist()
        weights = equal_weight_weights(holdings)
        realized_returns = selected_rows.set_index("ticker")["future_return"]
        turnover = calculate_turnover(pretrade_weights, weights)
        transaction_cost = calculate_one_way_transaction_cost(
            turnover,
            one_way_turnover_cost_bps=one_way_turnover_cost_bps,
        )
        gross_return = float((realized_returns * weights).sum())

        results.append(
            {
                "signal_date": signal_date,
                "date": pd.Timestamp(target_dates[0]),
                "portfolio_return": gross_return - transaction_cost,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "turnover": turnover,
                "holdings": holdings,
                "weights": weights.to_dict(),
                "asset_returns": realized_returns.to_dict(),
                "execution_timing": execution_timing,
            }
        )
        pretrade_weights = _drift_weights(weights, realized_returns)

    return pd.DataFrame(results)
