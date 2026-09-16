"""Record historical CLI provenance without changing research calculations."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import wraps
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
from uuid import uuid4

from src import config
from src.data.loader import load_benchmark, load_prices
from src.features.ml_dataset import FEATURE_COLUMNS
from src.models.ml_ranker import MLStockRanker


ROOT = Path(__file__).resolve().parents[2]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class ResearchRunConfig:
    provider: str = "fixed20_ex_post"
    top_n: int = config.TOP_N
    optimizer_window: int = config.OPTIMIZATION_LOOKBACK_DAYS
    max_weight: float = config.MAX_POSITION_WEIGHT
    cost_bps: float = config.ONE_WAY_TURNOVER_COST_BPS
    execution_mode: str = config.EXECUTION_TIMING_MODE
    momentum_months: int = config.MOMENTUM_LOOKBACK_MONTHS
    momentum_long_months: int = config.MOMENTUM_12_1_LOOKBACK_MONTHS
    momentum_skip_months: int = config.MOMENTUM_12_1_SKIP_MONTHS
    minimum_train_months: int = config.ML_MINIMUM_TRAIN_MONTHS
    risk_free_rate: float = config.RISK_FREE_RATE
    features: tuple = tuple(FEATURE_COLUMNS)


def capture_provenance(entrypoint, root=ROOT):
    """Hash actual files, including uncommitted code; never borrow a parent repo SHA."""
    root = Path(root).resolve()
    sources = sorted(root.glob("*.py")) + sorted((root / "src").rglob("*.py"))
    sources += [p for p in (root / "requirements.txt", root / "requirements-lock.txt") if p.exists()]
    source_hashes = {str(p.relative_to(root)): file_hash(p) for p in sources}
    inputs = {str(p.relative_to(root)): file_hash(p) for p in sorted((root / "data/raw").glob("*.parquet"))}
    data_summary = {}
    if {"data/raw/us_stock.parquet", "data/raw/benchmark.parquet"}.issubset(inputs):
        prices = load_prices(root / "data/raw/us_stock.parquet")
        benchmark = load_benchmark(root / "data/raw/benchmark.parquet")
        data_summary = {
            "universe": list(prices.columns), "first_price_date": str(prices.index.min()),
            "last_price_date": str(prices.index.max()), "price_rows": len(prices),
            "last_benchmark_date": str(benchmark.index.max()),
            "as_of_date": str(min(prices.index.max(), benchmark.index.max())),
        }
    commit = None
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip()
        if Path(top).resolve() == root:
            commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    packages = {}
    for name in ("pandas", "numpy", "scipy", "scikit-learn", "PyPortfolioOpt", "cvxpy", "matplotlib", "pyarrow"):
        packages[name] = importlib.metadata.version(name)
    return {
        "entrypoint": entrypoint, "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(ResearchRunConfig()), "model_parameters": MLStockRanker().model.get_params(),
        "input_files_sha256": inputs, "source_files_sha256": source_hashes,
        "data_summary": data_summary,
        "source_hash": hashlib.sha256(json.dumps(source_hashes, sort_keys=True).encode()).hexdigest(),
        "git_commit": commit, "python_version": platform.python_version(), "packages": packages,
        "warning": "Illustrative ex-post universe; missing delistings; ^GSPC price-index mismatch; same-close execution.",
    }


def record_research_run(function):
    """Write an exclusive success/failure manifest; existing report paths stay stable."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        entrypoint = Path(function.__code__.co_filename).name
        manifest = capture_provenance(entrypoint)
        reports = ROOT / "reports"
        before = {p.name: p.stat().st_mtime_ns for p in reports.glob("*") if p.is_file()}
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        else:
            manifest["status"] = "completed"
            return result
        finally:
            manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
            manifest["outputs_sha256"] = {
                str(p.relative_to(ROOT)): file_hash(p) for p in sorted(reports.glob("*"))
                if p.is_file() and before.get(p.name) != p.stat().st_mtime_ns
            }
            directory = reports / "provenance"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{Path(entrypoint).stem}_{uuid4().hex}.json"
            with path.open("x") as stream:
                json.dump(manifest, stream, indent=2)
                stream.write("\n")
    return wrapped
