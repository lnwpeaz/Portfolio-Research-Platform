import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path


RESEARCH_WARNINGS = [
    "Historical performance is affected by a fixed ex-post universe.",
    "Historical tests do not include delisted securities.",
    "Yahoo data are not point-in-time institutional data.",
    "The benchmark uses ^GSPC price returns while securities use adjusted prices.",
    "Same-close historical execution remains a research assumption.",
    "Live recommendations are research outputs, not trade execution instructions.",
]


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def build_manifest(run_id, created_at, config, snapshot, signal_date, model_metadata, outputs):
    packages = {}
    for name in ("pandas", "numpy", "scikit-learn", "PyPortfolioOpt"):
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name] = "unavailable"
    return {
        "run_id": run_id, "created_at": created_at, "as_of_date": snapshot.metadata["as_of_date"],
        "signal_date": str(signal_date), "data_source": config.data_source,
        "universe": list(config.universe), "eligible_universe": snapshot.metadata["eligible_universe"],
        "config": {"signal_frequency": config.signal_frequency, "execution_mode": config.execution_mode, "transaction_cost_bps": config.transaction_cost_bps, "top_n": config.top_n, "random_seed": config.random_seed},
        "input_hash": snapshot.metadata["input_hash"], "feature_columns": model_metadata["feature_columns"],
        "model_hyperparameters": model_metadata["hyperparameters"],
        "strategy_names": ["momentum_3m_equal", "momentum_12_1_equal", "momentum_max_sharpe", "momentum_min_vol", "ml"],
        "execution_assumption": config.execution_mode, "transaction_cost_assumption": f"{config.transaction_cost_bps} bps one-way turnover",
        "git_commit_if_available": _git_commit(), "python_version": platform.python_version(),
        "package_versions": packages, "outputs": outputs, "warnings": RESEARCH_WARNINGS,
    }


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, default=str))
