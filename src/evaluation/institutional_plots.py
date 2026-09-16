"""Compact static figures for institutional diagnostics."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def _save(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.25)
    plt.gcf().text(
        0.5, 0.02,
        "Illustrative research; fixed20 uses an ex-post universe with survivorship bias\n"
        "Interpret with run_manifest.json; ^GSPC is a price-return benchmark",
        ha="center", fontsize=8, color="#555555",
    )
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_quantile_performance(returns, path):
    frame = returns.set_index("date")[[f"Q{i}" for i in range(1, 6)]]
    (1 + frame.fillna(0)).cumprod().plot(figsize=(9, 5), linewidth=1.4)
    plt.title("ML score quantiles: cumulative equal-weight returns")
    plt.ylabel("Growth of 1")
    _save(path)


def plot_long_short(returns, path):
    spread = returns.set_index("date")["Q5_minus_Q1"]
    (1 + spread.fillna(0)).cumprod().plot(figsize=(9, 4), color="black")
    plt.title("Research long-short spread: Q5 minus Q1")
    plt.ylabel("Growth of 1")
    _save(path)


def plot_rank_ic(ic, path):
    frame = ic.set_index("date")
    frame.spearman_rank_ic.plot(figsize=(9, 4), alpha=0.65, label="Monthly Rank IC")
    frame.spearman_rank_ic.rolling(12, min_periods=6).mean().plot(label="12M mean")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.legend()
    plt.title("Cross-sectional Spearman Rank IC")
    _save(path)


def plot_universe_coverage(coverage, path):
    coverage.set_index("date")[["universe_members", "eligible_members", "prediction_eligible"]].plot(figsize=(9, 4))
    plt.title("Universe and feature coverage")
    plt.ylabel("Securities")
    _save(path)


def plot_sector_exposure(exposure, path):
    average = exposure.groupby("sector").active_sector_weight.mean().sort_values()
    average.plot.barh(figsize=(8, max(4, len(average) * 0.3)))
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Average Top-N active sector exposure (equal weight)")
    plt.xlabel("Top-N weight minus universe weight")
    _save(path)
