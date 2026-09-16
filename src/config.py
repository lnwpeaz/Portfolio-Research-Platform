# src/config.py

TOP_N = 5

MOMENTUM_LOOKBACK_MONTHS = 3

MOMENTUM_12_1_LOOKBACK_MONTHS = 12
MOMENTUM_12_1_SKIP_MONTHS = 1

OPTIMIZATION_LOOKBACK_DAYS = 252

ONE_WAY_TURNOVER_COST_BPS = 10

MAX_POSITION_WEIGHT = 0.40

RISK_FREE_RATE = 0.0

ML_MINIMUM_TRAIN_MONTHS = 36

ROLLING_RISK_WINDOW_MONTHS = 12

# Signals use month-end closing data and are assumed to execute at that same
# close. This optimistic research convention is deliberately explicit. No
# alternative execution mode is implemented until suitable execution prices
# are incorporated consistently in both the momentum and ML pipelines.
EXECUTION_TIMING_MODE = "same_close"

PERIODS_PER_YEAR_MONTHLY = 12
PERIODS_PER_YEAR_DAILY = 252
