"""Corporate-action and adjusted/raw-price relationship diagnostics."""

import numpy as np
import pandas as pd


def corporate_action_flags(panel, ratio_jump_threshold=0.05):
    rows = []
    for ticker, group in panel.groupby("canonical_ticker", sort=True):
        group = group.sort_values("date")
        for record in group[group.stock_splits.fillna(0).ne(0)].itertuples():
            rows.append({"ticker": ticker, "date": record.date, "flag": "REPORTED_STOCK_SPLIT", "value": record.stock_splits})
        for record in group[group.dividends.fillna(0).gt(0)].itertuples():
            rows.append({"ticker": ticker, "date": record.date, "flag": "REPORTED_DIVIDEND", "value": record.dividends})
        ratio = group.adjusted_close / group.raw_close
        jumps = ratio.pct_change(fill_method=None).abs().gt(ratio_jump_threshold)
        reported = group.stock_splits.fillna(0).ne(0) | group.dividends.fillna(0).gt(0)
        for index in group.index[jumps & ~reported]:
            rows.append({"ticker": ticker, "date": group.loc[index, "date"], "flag": "UNEXPLAINED_ADJUSTMENT_RATIO_CHANGE", "value": ratio.loc[index]})
    return pd.DataFrame(rows, columns=["ticker", "date", "flag", "value"])
