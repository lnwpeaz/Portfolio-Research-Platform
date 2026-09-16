import pandas as pd

from src.models.ml_ranker import (
    MLStockRanker,
)


def generate_walk_forward_scores(
    dataset: pd.DataFrame,
    minimum_train_months: int = 36,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward ML.

    At prediction cutoff t, train only on non-null labels with target_date < t.
    A label ending exactly at t is embargoed even though its feature date is
    earlier. Count the minimum window using available labeled feature months.
    """

    required = {
        "date", "target_date", "ticker", "target", "future_return",
        "benchmark_return",
    }
    missing_columns = required.difference(dataset.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing columns: {sorted(missing_columns)}"
        )
    if minimum_train_months < 1:
        raise ValueError("minimum_train_months must be positive.")

    dates = (
        dataset["date"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    results = []

    for prediction_date in dates:

        train_data = (
            dataset[
                dataset["target_date"].notna()
                & (dataset["target_date"] < prediction_date)
                & dataset["target"].notna()
            ]
        )

        if train_data["date"].nunique() < minimum_train_months:
            continue

        prediction_data = (
            dataset[
                dataset["date"]
                == prediction_date
            ]
            .copy()
        )

        if (
            train_data.empty
            or prediction_data.empty
        ):
            continue

        # Need both classes
        if (
            train_data["target"]
            .nunique()
            < 2
        ):
            continue

        model = MLStockRanker()

        model.fit(
            train_data
        )

        prediction_data[
            "ml_score"
        ] = model.predict_scores(
            prediction_data
        )

        results.append(
            prediction_data[
                [
                    "date",
                    "target_date",
                    "ticker",
                    "ml_score",
                    "future_return",
                    "benchmark_return",
                    "target",
                ]
            ]
        )

    if not results:
        return pd.DataFrame()

    scores = pd.concat(
        results,
        ignore_index=True,
    )

    return (
        scores
        .sort_values(
            ["date", "ml_score"],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(drop=True)
    )
