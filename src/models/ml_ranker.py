import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
)

from src.features.ml_dataset import (
    FEATURE_COLUMNS,
)


class MLStockRanker:
    """
    Baseline classifier used to estimate
    probability of benchmark outperformance.
    """

    def __init__(
        self,
        max_iter: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        random_state: int = 42,
    ):
        self.model = (
            HistGradientBoostingClassifier(
                max_iter=max_iter,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
            )
        )

        self.is_fitted = False

    def fit(
        self,
        data: pd.DataFrame,
    ) -> None:

        if data.empty:
            raise ValueError(
                "Training dataset is empty."
            )

        training_data = data.dropna(
            subset=FEATURE_COLUMNS + ["target"]
        )
        if training_data.empty:
            raise ValueError("Training dataset has no labeled complete rows.")

        X = training_data[
            FEATURE_COLUMNS
        ]

        y = training_data[
            "target"
        ].astype(int)

        self.model.fit(
            X,
            y,
        )

        self.is_fitted = True

    def predict_scores(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:

        if not self.is_fitted:
            raise RuntimeError(
                "Model must be fitted "
                "before prediction."
            )

        X = data[
            FEATURE_COLUMNS
        ]

        probabilities = (
            self.model
            .predict_proba(X)[:, 1]
        )

        return pd.Series(
            probabilities,
            index=data.index,
            name="ml_score",
        )
