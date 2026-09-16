from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


def load_prices(path=None) -> pd.DataFrame:
    path = Path(path) if path is not None else RAW_DIR / "us_stock.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Price data not found: {path}"
        )

    df = pd.read_parquet(path)

    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]

    return df.sort_index()


def load_benchmark(path=None) -> pd.Series:
    path = Path(path) if path is not None else RAW_DIR / "benchmark.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark data not found: {path}"
        )

    df = pd.read_parquet(path)

    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]

    if isinstance(df, pd.DataFrame):
        df = df.iloc[:, 0]

    df.name = "benchmark"

    return df.sort_index()


def calculate_daily_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    return prices.pct_change(
        fill_method=None
    )


def save_processed(
    df: pd.DataFrame,
    filename: str,
) -> None:
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    path = PROCESSED_DIR / filename

    df.to_parquet(path)
