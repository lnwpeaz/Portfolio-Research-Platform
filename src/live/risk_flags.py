from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class RiskFlagThresholds:
    high_single_name_weight: float = .35
    low_effective_holdings: float = 3.5
    high_turnover: float = .75
    model_score_range_low: float = .10
    historical_top1_contribution_high: float = .25


def generate_risk_flags(summary, snapshot_status, signals, predictions,
                        historical_contribution=None, thresholds=RiskFlagThresholds()):
    rows = []
    def add(scope, code, value, threshold, message): rows.append({"scope": scope, "flag": code, "value": value, "threshold": threshold, "message": message})
    for row in summary.itertuples(index=False):
        if row.max_weight >= thresholds.high_single_name_weight: add(row.strategy, "HIGH_SINGLE_NAME_WEIGHT", row.max_weight, thresholds.high_single_name_weight, "Target portfolio has a high single-security weight.")
        if row.effective_holdings < thresholds.low_effective_holdings: add(row.strategy, "LOW_EFFECTIVE_HOLDINGS", row.effective_holdings, thresholds.low_effective_holdings, "Target portfolio is concentrated by HHI.")
        if row.estimated_turnover_vs_previous > thresholds.high_turnover: add(row.strategy, "HIGH_TURNOVER", row.estimated_turnover_vs_previous, thresholds.high_turnover, "Estimated one-way turnover is high.")
        if str(row.optimization_status).startswith("fallback_equal:"): add(row.strategy, "OPTIMIZER_FALLBACK", row.optimization_status, "optimized", "Optimizer used its recorded equal-weight fallback.")
    for row in snapshot_status[~snapshot_status.eligible].itertuples(index=False): add(row.ticker, "STALE_MARKET_DATA" if row.reason == "stale_price" else "MISSING_SIGNAL", row.staleness_days, 7, row.reason)
    for row in signals[~signals.signal_valid].itertuples(index=False): add(row.ticker, "MISSING_SIGNAL", row.invalid_reason, "valid", "Momentum signal unavailable.")
    score_range = predictions.ml_score.max() - predictions.ml_score.min()
    if score_range < thresholds.model_score_range_low: add("ml", "MODEL_SCORE_CONCENTRATION", score_range, thresholds.model_score_range_low, "ML cross-sectional score range is narrow.")
    if historical_contribution is not None:
        for row in historical_contribution.itertuples(index=False):
            if row.top_1_share > thresholds.historical_top1_contribution_high: add(row.strategy, "HIGH_HISTORICAL_CONTRIBUTION_DEPENDENCE", row.top_1_share, thresholds.historical_top1_contribution_high, "Historical positive attribution is concentrated in one security.")
    return pd.DataFrame(rows, columns=["scope", "flag", "value", "threshold", "message"])
