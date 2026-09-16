import numpy as np
import pandas as pd


def build_cross_strategy_consensus(portfolios, signals, predictions):
    tickers = sorted(set(signals.ticker) | set(predictions.ticker))
    result = pd.DataFrame({"ticker": tickers}).set_index("ticker")
    holdings = {(s, t): w for s, t, w in portfolios[["strategy", "ticker", "target_weight"]].itertuples(index=False)}
    result["selected_by_3m"] = [(("momentum_3m_equal", t) in holdings) for t in tickers]
    result["selected_by_12_1"] = [(("momentum_12_1_equal", t) in holdings) for t in tickers]
    result["selected_by_ml"] = [(("ml", t) in holdings) for t in tickers]
    result["weight_in_max_sharpe"] = [holdings.get(("momentum_max_sharpe", t), 0) for t in tickers]
    result["weight_in_min_vol"] = [holdings.get(("momentum_min_vol", t), 0) for t in tickers]
    result["number_strategies_selecting_or_holding"] = [sum((s, t) in holdings for s in portfolios.strategy.unique()) for t in tickers]
    ranks = signals.set_index("ticker")[["rank_3m", "rank_12_1"]].join(predictions.set_index("ticker")[["ml_rank"]], how="outer")
    normalized = ranks.div(ranks.max())
    result["average_normalized_rank"] = normalized.mean(axis=1).reindex(tickers)
    return result.reset_index().sort_values(["number_strategies_selecting_or_holding", "average_normalized_rank", "ticker"], ascending=[False, True, True])
