"""Separate v1.1 execution/capacity diagnostics using unchanged v1.0 targets."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import pandas as pd

from src.data.loader import load_benchmark, load_prices
from src.evaluation.research_pipeline import run_strategy_suite
from src.execution.capacity import ExecutionConfig, TRADE_WEIGHT_TOLERANCE, evaluate_capacity
from src.execution.data import calculate_adv, liquidity_panels, load_execution_prices, resolve_liquidity_path


WARNING = "Execution/capacity sensitivity — not official v1.0 results."


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def figures(summary, output):
    for field, ylabel, filename in (
        ("estimated_cost_bps", "Mean rebalance cost (bps of portfolio AUM)", "aum_cost.png"),
        ("max_trade_pct_adv", "Maximum trade / prior ADV (fraction)", "aum_participation.png"),
        ("diagnostic_sharpe", "Diagnostic monthly Sharpe (annualized)", "aum_sharpe.png"),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        for axis, timing in zip(axes, ("same_close_diagnostic", "next_session_open")):
            for strategy, group in summary[summary.timing.eq(timing)].groupby("strategy", sort=True):
                group = group.sort_values("aum")
                axis.plot(group.aum, group[field], marker="o", label=strategy)
            axis.set(xscale="log", xlabel="Hypothetical AUM (USD, log scale)", ylabel=ylabel, title=timing.replace("_", " "))
            axis.grid(alpha=.25)
        axes[0].legend(fontsize=7)
        fig.suptitle(WARNING, fontsize=12)
        fig.text(.5, .01, "Ex-post survivors; hypothetical spread/impact; daily volume is not executable auction liquidity", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .06, 1, .92))
        fig.savefig(output / filename, dpi=140)
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--liquidity-panel", type=Path, help="Existing normalized provider Parquet with raw_close and volume")
    parser.add_argument("--adv-window", type=int, choices=(20, 60), default=20)
    parser.add_argument("--aum", type=float, nargs="+", default=list(ExecutionConfig().aums))
    parser.add_argument("--spread-slippage-bps", type=float, default=5.)
    parser.add_argument("--impact-bps", type=float, default=10., help="Hypothetical impact bps per traded dollar at reference participation")
    parser.add_argument("--reference-participation", type=float, default=.01)
    parser.add_argument("--participation-limits", type=float, nargs="+", default=[.01, .05, .10])
    args = parser.parse_args(argv)
    config = ExecutionConfig(tuple(args.aum), args.adv_window, args.spread_slippage_bps,
        args.impact_bps, args.reference_participation, tuple(args.participation_limits))
    liquidity_path = resolve_liquidity_path(args.liquidity_panel)
    stock_path, benchmark_path = Path("data/raw/us_stock.parquet"), Path("data/raw/benchmark.parquet")
    close, opening = load_execution_prices(stock_path)
    raw_close, volume = liquidity_panels(liquidity_path, close.index, close.columns)
    adv = calculate_adv(raw_close, volume, config.adv_window)
    if not adv.notna().all(axis=1).any():
        raise ValueError("INSUFFICIENT_LIQUIDITY_HISTORY: cannot run any complete cross-section")
    backtests, _, _ = run_strategy_suite(load_prices(), load_benchmark())
    trades, periods, summary, capacity, exclusions = evaluate_capacity(backtests, close, opening, adv, config)
    timing = summary.pivot(index=["strategy", "aum"], columns="timing", values=["diagnostic_sharpe", "diagnostic_cagr", "constant_10bps_sharpe"])
    timing.columns = [f"{metric}_{mode}" for metric, mode in timing.columns]
    for metric in ("diagnostic_sharpe", "diagnostic_cagr", "constant_10bps_sharpe"):
        timing[f"{metric}_next_minus_same"] = timing[f"{metric}_next_session_open"] - timing[f"{metric}_same_close_diagnostic"]
    output = Path("reports/execution_capacity")
    output.mkdir(parents=True, exist_ok=True)
    frames = {"trades.csv": trades, "periods.csv": periods, "execution_capacity_summary.csv": summary,
        "capacity_thresholds.csv": capacity, "execution_timing_comparison.csv": timing.reset_index(), "excluded_periods.csv": exclusions}
    for name, frame in frames.items():
        frame.to_csv(output / name, index=False)
    figures(summary, output)
    inputs = {}
    for path in (stock_path, benchmark_path, liquidity_path):
        label = str(path.relative_to(Path.cwd())) if path.is_absolute() and path.is_relative_to(Path.cwd()) else path.name if path.is_absolute() else str(path)
        inputs[label] = hash_file(path)
    code = sorted(Path('src').rglob('*.py')) + [Path('execution_capacity_main.py')]
    import importlib.metadata
    import platform
    commit = None
    try:
        root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True, stderr=subprocess.DEVNULL).strip()
        if Path(root).resolve() == Path.cwd().resolve():
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    manifest = {
        "warning": WARNING, "config": asdict(config), "input_sha256": inputs,
        "trade_weight_zero_tolerance": TRADE_WEIGHT_TOLERANCE,
        "source_sha256": {str(p): hash_file(p) for p in code},
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scikit-learn', 'PyPortfolioOpt', 'matplotlib')},
        "git_commit": commit,
        "first_signal_date": str(summary.first_signal_date.min()),
        "last_signal_date": str(periods.signal_date.max()),
        "observations_per_scenario": sorted(summary.observations.unique().tolist()),
        "liquidity_proxy": "vendor raw_close * volume; full trailing window ending strictly before signal",
        "price_proxy": "frozen dividend-adjusted open, not an executable fill or a quote",
        "capital_convention": "constant hypothetical AUM at each rebalance; initial cash reset at paired sample start; no compounding of scenario capital",
        "cost_convention": "per-traded-dollar costs on buys and sells; constant 10bps turnover reference separate; no double charging",
        "warnings": ["Ex-post survivorship remains unresolved.", "Mixed vendor vintages: frozen adjusted OHLC vs later downloaded raw-close/volume cache.",
            "Yahoo-style data are revisable, not institutional execution data.", "No empirical bid/ask data; spread and impact parameters are hypothetical.",
            "Daily volume does not guarantee liquidity at the open or close; no true fund capacity estimate.",
            "Net returns deduct estimated costs without a cash/fee financing simulation.",
            "Paired same-close diagnostic resets holdings at sample start and is not the official full-period v1.0 backtest."],
        "output_sha256": {p.name: hash_file(p) for p in sorted(output.glob('*')) if p.suffix in ('.csv', '.png')},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(WARNING)
    print(f"Paired signals: {periods.signal_date.min().date()} to {periods.signal_date.max().date()}; {summary.observations.iloc[0]} observations per scenario")
    print(summary[['strategy', 'timing', 'aum', 'max_trade_pct_adv', 'estimated_cost_bps', 'diagnostic_sharpe']].to_string(index=False))
    print(f"Outputs: {output}")


if __name__ == '__main__':
    main()
