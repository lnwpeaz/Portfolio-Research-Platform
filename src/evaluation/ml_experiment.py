"""Frozen, chronological comparison of the existing ML ranker settings."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from src.backtest.engine import run_score_backtest
from src.config import (
    EXECUTION_TIMING_MODE, ML_MINIMUM_TRAIN_MONTHS,
    ONE_WAY_TURNOVER_COST_BPS, TOP_N,
)
from src.data.frequency import last_observation_by_month
from src.evaluation.metrics import drawdown_series, sharpe_ratio
from src.evaluation.robustness_common import evaluate_return_series
from src.features.ml_dataset import FEATURE_COLUMNS, build_ml_dataset
from src.models.ml_ranker import MLStockRanker
from src.models.walk_forward import generate_walk_forward_scores


# Insertion order is the predeclared exact-tie rule. Do not tune this grid using
# the retrospective evaluation results.
CANDIDATES = {
    "baseline": {"max_depth": 3, "l2_regularization": 0.0},
    "depth2_l2_1": {"max_depth": 2, "l2_regularization": 1.0},
    "depth2_l2_10": {"max_depth": 2, "l2_regularization": 10.0},
    "depth3_l2_1": {"max_depth": 3, "l2_regularization": 1.0},
    "depth3_l2_10": {"max_depth": 3, "l2_regularization": 10.0},
}

VALIDATION_START = "2021-01-01"
VALIDATION_END = "2022-12-31"
EVALUATION_START = "2023-01-01"
EVALUATION_END = "2025-12-31"
ROOT = Path(__file__).resolve().parents[2]


def select_candidate(validation_returns: pd.DataFrame) -> str:
    """Select solely by net validation Sharpe; never consult later returns."""
    if not isinstance(validation_returns.index, pd.DatetimeIndex):
        raise ValueError("Validation returns require a DatetimeIndex.")
    validation = validation_returns.sort_index().loc[VALIDATION_START:VALIDATION_END]
    expected = pd.period_range("2021-01", "2022-12", freq="M")
    if not validation.index.to_period("M").equals(expected):
        raise ValueError("Validation requires exactly 24 distinct calendar months.")
    missing = set(CANDIDATES).difference(validation.columns)
    if missing:
        raise ValueError(f"Validation is missing candidates: {sorted(missing)}")
    values = validation[list(CANDIDATES)]
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Validation requires complete finite candidate returns.")
    sharpes = {name: sharpe_ratio(values[name]) for name in CANDIDATES}
    finite = {name: value for name, value in sharpes.items() if np.isfinite(value)}
    if not finite:
        raise ValueError("All validation Sharpe ratios are undefined; no finite winner.")
    return max(finite, key=finite.get)


def compare_segment(
    backtests: dict[str, pd.DataFrame], benchmark_returns: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare every portfolio on exactly the benchmark's dates, without filling."""
    benchmark = benchmark_returns.sort_index().rename("benchmark")
    if (
        len(benchmark) < 2 or benchmark.index.has_duplicates
        or not np.isfinite(benchmark.to_numpy(dtype=float)).all()
    ):
        raise ValueError("Benchmark requires at least two complete unique months.")
    returns = benchmark.to_frame()
    rows = []
    if not backtests:
        raise ValueError("At least one candidate backtest is required.")
    for name, backtest in backtests.items():
        history = backtest.sort_values("date").set_index("date")
        series = history.portfolio_return
        if not series.index.equals(benchmark.index):
            raise ValueError(f"Candidate {name} must use identical benchmark months.")
        numeric = history[["portfolio_return", "turnover", "transaction_cost"]]
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"Candidate {name} has missing or non-finite observations.")
        returns[name] = series
        rows.append({
            "candidate": name,
            **evaluate_return_series(series, benchmark),
            "mean_monthly_turnover": history.turnover.mean(),
            "total_transaction_cost": history.transaction_cost.sum(),
        })
    return pd.DataFrame(rows), returns


def _write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _frame_hash(frame: pd.DataFrame | pd.Series) -> str:
    data = frame.to_frame() if isinstance(frame, pd.Series) else frame
    digest = hashlib.sha256()
    digest.update(str(list(data.columns)).encode())
    digest.update(str(data.dtypes.tolist()).encode())
    digest.update(pd.util.hash_pandas_object(data, index=True).values.tobytes())
    return digest.hexdigest()


def _segment_backtests(dataset, start, end, stage, on_progress):
    first_signal = (pd.Timestamp(start).to_period("M") - 1).start_time
    last_signal = (pd.Timestamp(end).to_period("M") - 1).end_time
    backtests, score_frames = {}, []
    for name, parameters in CANDIDATES.items():
        if on_progress is not None:
            on_progress(stage, name)
        scores = generate_walk_forward_scores(
            dataset, minimum_train_months=ML_MINIMUM_TRAIN_MONTHS,
            model_parameters=parameters,
            prediction_start=first_signal, prediction_end=last_signal,
        )
        if scores.empty:
            raise ValueError(f"No scores for {stage}/{name}; insufficient history.")
        backtests[name] = run_score_backtest(
            scores, top_n=TOP_N,
            one_way_turnover_cost_bps=ONE_WAY_TURNOVER_COST_BPS,
            execution_timing=EXECUTION_TIMING_MODE,
        )
        score_frames.append(scores.assign(candidate=name))
    return backtests, pd.concat(score_frames, ignore_index=True)


def _save_segment(output, stage, summary, returns, backtests, scores):
    summary.to_csv(output / f"{stage}_summary.csv", index=False)
    returns.rename_axis("date").to_csv(output / f"{stage}_returns.csv")
    scores.to_csv(output / f"{stage}_scores.csv", index=False)
    histories = []
    for name, backtest in backtests.items():
        frame = backtest.assign(candidate=name).copy()
        for column in ("holdings", "weights", "asset_returns"):
            frame[column] = frame[column].map(lambda value: json.dumps(value, sort_keys=True))
        histories.append(frame)
    pd.concat(histories, ignore_index=True).to_csv(
        output / f"{stage}_backtests.csv", index=False,
    )


def _plot_evaluation(output, returns, summary, selected):
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter, YearLocator

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), layout="constrained")
    colors = {"baseline": "#555555", "benchmark": "#d18d17", selected: "#1865ad"}
    if selected == "baseline":
        colors["baseline"] = "#1865ad"
    for name in dict.fromkeys(["baseline", selected, "benchmark"]):
        series = returns[name]
        initial_date = series.index[0] - pd.offsets.MonthEnd(1)
        wealth = pd.concat([
            pd.Series([1.0], index=[initial_date]), (1 + series).cumprod(),
        ])
        label = name + (" (selected)" if name == selected else "")
        axes[0, 0].plot(wealth.index, wealth, label=label, color=colors[name])
        axes[0, 1].plot(
            series.index, 100 * drawdown_series(series),
            label=label, color=colors[name],
        )
    axes[0, 0].set(title="Net wealth: initial capital = 1", ylabel="Wealth")
    axes[0, 1].set(title="Drawdown from initial capital / running peak", ylabel="Drawdown (%)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    for axis in axes[0]:
        axis.xaxis.set_major_locator(YearLocator())
        axis.xaxis.set_major_formatter(DateFormatter("%Y"))
    labels = summary.candidate.tolist()
    position = np.arange(len(labels))
    bar_colors = ["#1865ad" if name == selected else "#aeb9c6" for name in labels]
    axes[1, 0].bar(position, summary.sharpe, color=bar_colors)
    axes[1, 0].set(title="Evaluation net Sharpe: all frozen candidates", ylabel="Sharpe")
    axes[1, 1].bar(position - 0.18, 100 * summary.annual_volatility, width=0.36, label="Annual volatility")
    axes[1, 1].bar(position + 0.18, -100 * summary.max_drawdown, width=0.36, label="Maximum drawdown magnitude")
    axes[1, 1].set(title="Evaluation risk: all frozen candidates", ylabel="Percent")
    axes[1, 1].legend(fontsize=8)
    for axis in axes[1]:
        axis.set_xticks(position, labels, rotation=20, ha="right", fontsize=8)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle(
        "ML regularization | 2023–2025 retrospective evaluation\n"
        "Selected on 2021–2022 net Sharpe; ex-post 20-stock universe in bundled research",
        fontsize=12,
    )
    fig.savefig(output / "comparison.png", dpi=160)
    plt.close(fig)


def _write_report(output, validation, evaluation, selected):
    lines = [
        "# Frozen ML regularization study", "",
        f"Selected candidate: **{selected}**, using only 2021–2022 net Sharpe.", "",
        "## Baseline versus selected candidate", "",
        "| Segment | Candidate | CAGR | Annual alpha | Sharpe | Annual volatility | Max drawdown |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for segment, summary in (("Validation 2021–2022", validation), ("Evaluation 2023–2025", evaluation)):
        for name in dict.fromkeys(["baseline", selected]):
            row = summary.set_index("candidate").loc[name]
            lines.append(
                f"| {segment} | {name} | {row.annual_return:.2%} | "
                f"{row.annual_alpha:.2%} | {row.sharpe:.3f} | "
                f"{row.annual_volatility:.2%} | {row.max_drawdown:.2%} |"
            )
    lines.extend([
        "", "## All evaluation candidates (no reselection)", "",
        "| Candidate | Selected | Annual alpha | Sharpe | Annual volatility | Max drawdown |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in evaluation.itertuples():
        lines.append(
            f"| {row.candidate} | {'yes' if row.is_selected else ''} | "
            f"{row.annual_alpha:.2%} | {row.sharpe:.3f} | "
            f"{row.annual_volatility:.2%} | {row.max_drawdown:.2%} |"
        )
    base = evaluation.set_index("candidate").loc["baseline"]
    chosen = evaluation.set_index("candidate").loc[selected]
    lines.extend([
        "", "## Evaluation changes versus baseline", "",
        f"- Net Sharpe change: {chosen.sharpe - base.sharpe:+.3f}.",
        f"- Annual alpha change: {100 * (chosen.annual_alpha - base.annual_alpha):+.2f} percentage points.",
        f"- Annual volatility change: {100 * (chosen.annual_volatility - base.annual_volatility):+.2f} percentage points.",
        f"- Maximum drawdown change: {100 * (chosen.max_drawdown - base.max_drawdown):+.2f} percentage points (positive means less severe).",
        "", "![Chronological evaluation comparison](comparison.png)", "",
        "## Interpretation and reproducibility", "",
        "The 2023–2025 results were already inspected in prior research. This is a "
        "retrospective chronological evaluation, not an unseen holdout. No test-period "
        "winner is substituted for the validation-selected candidate.", "",
        "Monthly expanding refits use only labels ending strictly before prediction. "
        "Hyperparameters are fixed for evaluation. Each segment restarts in cash; "
        "the initial deployment cost is included. These segment results must not be "
        "compared directly to the official continuous 2018–2025 table.", "",
        "Annual alpha is the existing annualized single-benchmark intercept at zero "
        "risk-free rate, not excess CAGR or factor-adjusted alpha. The ex-post "
        "survivor universe, missing delistings, stock-adjusted versus price-index "
        "benchmark mismatch and same-close execution remain limitations.", "",
        "See `protocol.md`, `config.json`, `selection.json` and `manifest.json`. "
        "CSV summaries, net returns, scores and target/return ledgers retain all "
        "candidates, including worse results. No official v1.0 reports are overwritten.", "",
    ])
    (output / "report.md").write_text("\n".join(lines))


def run_ml_experiment(prices, benchmark, output_dir, on_progress=None):
    """Run validation first, freeze its winner, then evaluate all fixed candidates.

    The output directory must be new. A failed run retains a failure manifest,
    never an apparent successful result with silently shortened samples.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "unseen_holdout": False,
    }
    try:
        from importlib.metadata import version

        manifest["input_frames_sha256"] = {
            "prices": _frame_hash(prices), "benchmark": _frame_hash(benchmark),
        }
        source_paths = sorted((ROOT / "src").rglob("*.py")) + [
            ROOT / "ml_experiment_main.py", ROOT / "docs/ML_REGULARIZATION.md",
            ROOT / "requirements.txt", ROOT / "requirements-lock.txt",
        ]
        manifest["source_files_sha256"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths if path.is_file()
        }
        manifest["packages"] = {
            name: version(name) for name in ("pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "threadpoolctl")
        }
        manifest["input_summary"] = {
            "tickers": list(prices.columns), "price_rows": len(prices),
            "first_date": str(prices.index.min()), "last_date": str(prices.index.max()),
        }
        config = {
            "candidates": {name: MLStockRanker(**params).model.get_params() for name, params in CANDIDATES.items()},
            "feature_columns": FEATURE_COLUMNS, "top_n": TOP_N,
            "minimum_train_months": ML_MINIMUM_TRAIN_MONTHS,
            "one_way_turnover_cost_bps": ONE_WAY_TURNOVER_COST_BPS,
            "execution_timing": EXECUTION_TIMING_MODE,
            "validation_start": VALIDATION_START, "validation_end": VALIDATION_END,
            "evaluation_start": EVALUATION_START, "evaluation_end": EVALUATION_END,
            "selection_objective": "maximum finite validation net Sharpe; baseline-first exact ties",
            "portfolio_start": "cash reset for each segment",
            "unseen_holdout": False,
        }
        _write_json(output / "config.json", config)
        (output / "protocol.md").write_text((ROOT / "docs/ML_REGULARIZATION.md").read_text())

        monthly_benchmark = last_observation_by_month(benchmark.sort_index())
        benchmark_returns = monthly_benchmark.pct_change(fill_method=None)
        for start, end in ((VALIDATION_START, VALIDATION_END), (EVALUATION_START, EVALUATION_END)):
            observed = benchmark_returns.loc[start:end]
            expected = pd.period_range(start, end, freq="M")
            if not observed.index.to_period("M").equals(expected) or observed.isna().any():
                raise ValueError(f"Incomplete benchmark calendar months for {start} through {end}.")
        validation_data = build_ml_dataset(
            prices.loc[:VALIDATION_END], benchmark.loc[:VALIDATION_END],
        )
        validation_bt, validation_scores = _segment_backtests(
            validation_data, VALIDATION_START, VALIDATION_END, "validation", on_progress,
        )
        validation, validation_returns = compare_segment(
            validation_bt, benchmark_returns.loc[VALIDATION_START:VALIDATION_END],
        )
        selected = select_candidate(validation_returns)
        validation["is_selected"] = validation.candidate.eq(selected)
        _save_segment(output, "validation", validation, validation_returns, validation_bt, validation_scores)
        selection = {
            "selected_candidate": selected,
            "selected_parameters": config["candidates"][selected],
            "validation_start": VALIDATION_START, "validation_end": VALIDATION_END,
            "validation_net_sharpe": {
                row.candidate: float(row.sharpe) if np.isfinite(row.sharpe) else None
                for row in validation.itertuples()
            },
            "validation_returns_sha256": hashlib.sha256((output / "validation_returns.csv").read_bytes()).hexdigest(),
            "config_sha256": hashlib.sha256((output / "config.json").read_bytes()).hexdigest(),
            "tie_break_order": list(CANDIDATES), "unseen_holdout": False,
        }
        # Persist before constructing the evaluation dataset or producing its predictions.
        _write_json(output / "selection.json", selection)
        evaluation_data = build_ml_dataset(
            prices.loc[:EVALUATION_END], benchmark.loc[:EVALUATION_END],
        )
        evaluation_bt, evaluation_scores = _segment_backtests(
            evaluation_data, EVALUATION_START, EVALUATION_END, "evaluation", on_progress,
        )
        evaluation, evaluation_returns = compare_segment(
            evaluation_bt, benchmark_returns.loc[EVALUATION_START:EVALUATION_END],
        )
        evaluation["is_selected"] = evaluation.candidate.eq(selected)
        _save_segment(output, "evaluation", evaluation, evaluation_returns, evaluation_bt, evaluation_scores)
        _plot_evaluation(output, evaluation_returns, evaluation, selected)
        _write_report(output, validation, evaluation, selected)
        manifest["status"] = "completed"
        manifest["selected_candidate"] = selected
        return {"selection": selection, "validation": validation, "evaluation": evaluation}
    except BaseException as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        manifest["outputs_sha256"] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.iterdir()) if path.is_file()
        }
        _write_json(output / "manifest.json", manifest)
