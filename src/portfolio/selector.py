import pandas as pd


def select_top_n(
    scores: pd.Series,
    n: int = 5,
) -> list[str]:
    """
    Select stocks with highest factor/model score.
    """

    return (
        scores
        .dropna()
        .sort_values(
            ascending=False
        )
        .head(n)
        .index
        .tolist()
    )


def equal_weights(
    tickers: list[str],
) -> pd.Series:

    if not tickers:
        raise ValueError(
            "Ticker list is empty."
        )

    weight = 1 / len(tickers)

    return pd.Series(
        weight,
        index=tickers,
        name="weight",
    )