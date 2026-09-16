"""Fixed-target, constant-AUM sensitivity scenarios; no strategy fitting here."""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from src.evaluation.metrics import performance_summary
from src.execution.data import next_session

TRADE_WEIGHT_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ExecutionConfig:
    aums: tuple = (100_000., 1_000_000., 10_000_000., 50_000_000., 100_000_000.)
    adv_window: int = 20
    spread_slippage_bps: float = 5.
    impact_bps_at_reference: float = 10.
    reference_participation: float = .01
    participation_limits: tuple = (.01, .05, .10)

    def __post_init__(self):
        if self.adv_window not in (20, 60):
            raise ValueError("Choose ADV20 or ADV60")
        for values in (self.aums, self.participation_limits):
            if not values or any(not np.isfinite(x) or x <= 0 for x in values) or len(set(values)) != len(values):
                raise ValueError("Scenario values must be positive, finite and unique")
        if any(x > 1 for x in self.participation_limits):
            raise ValueError("Participation limits must be at most one")
        for value in (self.spread_slippage_bps, self.impact_bps_at_reference):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Cost assumptions must be finite and nonnegative")
        if not np.isfinite(self.reference_participation) or self.reference_participation <= 0:
            raise ValueError("Reference participation must be positive")


def trade_diagnostics(target, pretrade, adv, aum, config):
    names = target.index.union(pretrade.index).sort_values()
    target = target.reindex(names, fill_value=0.).astype(float)
    previous = pretrade.reindex(names, fill_value=0.).astype(float)
    if not np.isfinite(aum) or aum <= 0:
        raise ValueError("AUM must be positive and finite")
    if not np.isfinite(target).all() or not np.isfinite(previous).all() or (target < 0).any() or (previous < 0).any():
        raise ValueError("Weights must be finite and long-only")
    if not np.isclose(target.sum(), 1) or not (np.isclose(previous.sum(), 0) or np.isclose(previous.sum(), 1)):
        raise ValueError("Targets must sum to one; pretrade must be invested or initial cash")
    delta = target - previous
    delta = delta.mask(delta.abs() <= TRADE_WEIGHT_TOLERANCE, 0.)
    trading = delta.abs() > 0
    liquidity = adv.reindex(names).astype(float)
    if (trading & (~np.isfinite(liquidity) | liquidity.le(0))).any():
        raise ValueError("INVALID_ADV: missing/zero/nonfinite liquidity for a trade; no silent deletion")
    participation = (aum * delta.abs() / liquidity).where(trading, 0.)
    unit_bps = config.spread_slippage_bps + config.impact_bps_at_reference * np.sqrt(participation / config.reference_participation)
    return pd.DataFrame({
        "ticker": names, "target_weight": target.to_numpy(), "pretrade_weight": previous.to_numpy(),
        "trade_weight": delta.to_numpy(), "position_dollars": (aum * target).to_numpy(),
        "trade_dollars": (aum * delta).to_numpy(), "ADV_dollars": liquidity.to_numpy(),
        "trade_as_pct_ADV": participation.to_numpy(), "assumed_cost_bps_per_traded_dollar": unit_bps.to_numpy(),
        "estimated_cost_dollars": (aum * delta.abs() * unit_bps / 10_000).to_numpy(),
    })


def paired_schedule(backtests, close, adv):
    """Use shared signals; restrict only data boundaries, never selected outcomes."""
    first = next(iter(backtests.values())).set_index("signal_date")["date"]
    shared = first.index
    for frame in backtests.values():
        shared = shared.intersection(pd.DatetimeIndex(frame.signal_date))
    ready = adv.notna().all(axis=1)
    if not ready.any():
        raise ValueError("INSUFFICIENT_LIQUIDITY_HISTORY: no complete ADV window")
    start = ready[ready].index[0]
    rows, excluded = [], []
    for signal in shared.sort_values():
        end = pd.Timestamp(first.loc[signal])
        for frame in backtests.values():
            if pd.Timestamp(frame.set_index("signal_date").loc[signal, "date"]) != end:
                raise ValueError("Strategy realization dates disagree")
        entry, exit_date = next_session(close.index, signal), next_session(close.index, end)
        if signal < start:
            excluded.append({"signal_date": signal, "reason": "before_complete_liquidity_history"})
        elif entry is None or exit_date is None:
            excluded.append({"signal_date": signal, "reason": "terminal_next_session_unavailable"})
        else:
            rows.append({"signal_date": signal, "realization_date": end, "next_entry": entry, "next_exit": exit_date})
    if len(rows) < 2:
        raise ValueError("INSUFFICIENT_PAIRED_EXECUTION_PERIODS")
    if any(previous["realization_date"] != following["signal_date"] for previous, following in zip(rows, rows[1:])):
        raise ValueError("NONCONTIGUOUS_PAIRED_SCHEDULE: cannot skip holding periods")
    return pd.DataFrame(rows), pd.DataFrame(excluded)


def evaluate_capacity(backtests, close, opening, adv, config=ExecutionConfig()):
    schedule, excluded = paired_schedule(backtests, close, adv)
    trade_rows, period_rows = [], []
    for strategy, backtest in backtests.items():
        targets = backtest.set_index("signal_date").weights
        for timing in ("same_close_diagnostic", "next_session_open"):
            pretrade = pd.Series(dtype=float)
            prices = close if timing == "same_close_diagnostic" else opening
            for period in schedule.itertuples(index=False):
                signal, realization = period.signal_date, period.realization_date
                entry, exit_date = (signal, realization) if timing == "same_close_diagnostic" else (period.next_entry, period.next_exit)
                target = pd.Series(targets.loc[signal], dtype=float).sort_index()
                # Form trades before reading the holding-period outcomes.
                initial = pretrade.empty
                turnover = 1. if initial else .5 * (target.subtract(pretrade, fill_value=0)).abs().sum()
                formed = []
                for aum in config.aums:
                    trades = trade_diagnostics(target, pretrade, adv.loc[signal], aum, config)
                    trades = trades.assign(strategy=strategy, timing=timing, aum=aum, signal_date=signal, execution_date=entry)
                    formed.append((aum, trades))
                active = target[target > 0]
                start_prices, end_prices = prices.loc[entry, active.index], prices.loc[exit_date, active.index]
                if (~np.isfinite(start_prices) | ~np.isfinite(end_prices) | start_prices.le(0) | end_prices.le(0)).any():
                    raise ValueError("MISSING_EXECUTION_PRICE: selected portfolio cannot be repriced or replaced")
                returns = end_prices / start_prices - 1
                gross = float(active @ returns)
                if not np.isfinite(gross) or gross <= -1:
                    raise ValueError("Invalid holding-period gross return")
                for aum, trades in formed:
                    cost = trades.estimated_cost_dollars.sum() / aum
                    if cost >= 1 + gross:
                        raise ValueError("Diagnostic costs exhaust capital; scenario is not executable")
                    trade_rows.append(trades)
                    period_rows.append({"strategy": strategy, "timing": timing, "aum": aum,
                        "signal_date": signal, "execution_date": entry, "valuation_date": exit_date,
                        "gross_return": gross, "turnover": turnover,
                        "constant_10bps_return": gross - turnover * .001,
                        "estimated_cost_bps": cost * 10_000, "diagnostic_net_return": gross - cost,
                        "initial_deployment": initial})
                pretrade = active * (1 + returns) / (1 + gross)
    trades, periods = pd.concat(trade_rows, ignore_index=True), pd.DataFrame(period_rows)
    summaries, capacities = [], []
    for (strategy, timing, aum), group in periods.groupby(["strategy", "timing", "aum"], sort=True):
        selected = trades[(trades.strategy == strategy) & (trades.timing == timing) & (trades.aum == aum) & trades.trade_weight.ne(0)]
        participation = selected.trade_as_pct_ADV
        metrics = performance_summary(group.diagnostic_net_return)
        reference = performance_summary(group.constant_10bps_return)
        summaries.append({"strategy": strategy, "timing": timing, "aum": aum, "observations": len(group),
            "first_signal_date": group.signal_date.min(), "last_valuation_date": group.valuation_date.max(),
            "median_trade_pct_adv": participation.median(), "p95_trade_pct_adv": participation.quantile(.95),
            "max_trade_pct_adv": participation.max(),
            **{f"pct_trades_gt_{n}pct_adv": float((participation > n / 100).mean()) for n in (1, 5, 10)},
            "estimated_cost_bps": group.estimated_cost_bps.mean(),
            "annualized_turnover": group.turnover.mean() * 12,
            "diagnostic_sharpe": metrics['Sharpe Ratio'], "diagnostic_cagr": metrics['Annual Return'],
            "constant_10bps_sharpe": reference['Sharpe Ratio'],
            "capacity_flag": "above_largest_tested_limit" if participation.max() > max(config.participation_limits) else "within_largest_tested_limit_not_execution_assurance"})
        if aum == config.aums[0]:
            for limit in config.participation_limits:
                bounds = limit * selected.ADV_dollars / selected.trade_weight.abs()
                binding = selected.loc[bounds.idxmin()]
                capacities.append({"strategy": strategy, "timing": timing, "max_participation": limit,
                    "diagnostic_aum_ceiling": bounds.min(), "binding_ticker": binding.ticker,
                    "binding_signal_date": binding.signal_date})
    return trades, periods, pd.DataFrame(summaries), pd.DataFrame(capacities), excluded
