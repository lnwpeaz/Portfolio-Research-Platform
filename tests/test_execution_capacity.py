from dataclasses import replace
import hashlib
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.execution.capacity import ExecutionConfig, evaluate_capacity, trade_diagnostics
from src.execution.data import calculate_adv, next_session, resolve_liquidity_path


def fixture():
    dates = pd.bdate_range('2024-01-01', '2024-05-02')
    close = pd.DataFrame({'A': np.linspace(100, 120, len(dates)), 'B': np.linspace(100, 95, len(dates))}, index=dates)
    opening = close * pd.DataFrame({'A': np.linspace(1, 1.04, len(dates)), 'B': np.linspace(1, .98, len(dates))}, index=dates)
    signals = pd.to_datetime(['2024-01-31','2024-02-29','2024-03-29'])
    ends = pd.to_datetime(['2024-02-29','2024-03-29','2024-04-30'])
    backtest = pd.DataFrame({'signal_date': signals, 'date': ends, 'weights': [{'A': .5, 'B': .5}, {'A': .8, 'B': .2}, {'A': 0., 'B': 1.}]})
    adv = pd.DataFrame(1_000_000., index=dates, columns=close.columns)
    return {'s': backtest}, close, opening, adv


def test_adv_is_raw_dollars_full_window_strictly_prior():
    dates = pd.bdate_range('2024-01-01', periods=23)
    price = pd.DataFrame({'A': np.arange(23) + 10.}, index=dates)
    volume = price * 0 + 100
    actual = calculate_adv(price, volume)
    assert actual.iloc[:20].isna().all().all()
    assert actual.iloc[20, 0] == price.iloc[:20,0].mean() * 100
    price.iloc[20:] = 1e9
    assert calculate_adv(price, volume).iloc[20,0] == actual.iloc[20,0]


@pytest.mark.parametrize('bad', [0., -1., np.nan, np.inf])
def test_invalid_volume_never_filled_or_treated_as_liquid(bad):
    dates = pd.bdate_range('2024-01-01', periods=22)
    price = pd.DataFrame({'A': 10.}, index=dates)
    volume = pd.DataFrame({'A': 100.}, index=dates)
    volume.iloc[10,0] = bad
    assert pd.isna(calculate_adv(price, volume).iloc[-1,0])
    with pytest.raises(ValueError, match='INVALID_ADV'):
        trade_diagnostics(pd.Series({'A': 1.}), pd.Series(dtype=float), pd.Series({'A': bad}), 100_000, ExecutionConfig())


def test_trade_dollars_include_entries_exits_and_participation():
    config = ExecutionConfig()
    trades = trade_diagnostics(pd.Series({'A': .5,'C': .5}), pd.Series({'A': .8,'B': .2}), pd.Series({'A': 1e6,'B': 2e6,'C': 1e6}), 100_000, config).set_index('ticker')
    assert np.allclose(trades.trade_dollars, [-30_000, -20_000, 50_000])
    assert np.allclose(trades.position_dollars, [50_000,0,50_000])
    assert np.allclose(trades.trade_as_pct_ADV, [.03,.01,.05])
    expected = 30000* (5+10*np.sqrt(3))/10000
    assert trades.loc['A','estimated_cost_dollars'] == pytest.approx(expected)


def test_scaling_monotonic_cost_and_initial_turnover():
    inputs = fixture()
    config = ExecutionConfig(aums=(100_000.,1_000_000.))
    trades, periods, summary, capacity, _ = evaluate_capacity(*inputs, config)
    small, large = [trades[trades.aum.eq(a)].reset_index(drop=True) for a in config.aums]
    assert np.allclose(large.trade_dollars, small.trade_dollars*10)
    assert np.allclose(large.trade_as_pct_ADV, small.trade_as_pct_ADV*10)
    for _, group in summary.groupby(['strategy','timing']):
        assert group.sort_values('aum').estimated_cost_bps.is_monotonic_increasing
    assert periods[periods.initial_deployment].turnover.eq(1).all()
    assert not capacity.empty


def test_next_session_is_market_calendar_not_next_valid_ticker_price():
    inputs = fixture()
    assert next_session(inputs[1].index, pd.Timestamp('2024-03-29')) == pd.Timestamp('2024-04-01')
    trades, periods, *_ = evaluate_capacity(*inputs, ExecutionConfig(aums=(100_000.,)))
    delayed = periods[periods.timing.eq('next_session_open')]
    assert (delayed.execution_date > delayed.signal_date).all()
    assert delayed.iloc[0].gross_return == pytest.approx(float(pd.Series({'A':.5,'B':.5}) @ (inputs[2].loc['2024-03-01']/inputs[2].loc['2024-02-01']-1)))
    inputs[2].loc['2024-02-01','A'] = np.nan
    with pytest.raises(ValueError, match='MISSING_EXECUTION_PRICE'):
        evaluate_capacity(*inputs)


def test_missing_adv_inside_sample_fails_instead_of_skipping_rebalance():
    inputs = fixture()
    inputs[3].loc['2024-02-29','A'] = np.nan
    with pytest.raises(ValueError, match='INVALID_ADV'):
        evaluate_capacity(*inputs)


def test_future_outcomes_cannot_change_already_formed_trades():
    inputs = fixture()
    first = evaluate_capacity(*inputs)[0]
    altered_open = inputs[2].copy()
    altered_open.loc['2024-03-01':] *= 2
    second = evaluate_capacity(inputs[0], inputs[1], altered_open, inputs[3])[0]
    assert_frame_equal(first[first.signal_date.eq('2024-01-31')], second[second.signal_date.eq('2024-01-31')])


def test_deterministic_and_does_not_mutate_targets_or_prices():
    inputs = fixture()
    snapshots = [inputs[0]['s'].copy(deep=True), *(p.copy(deep=True) for p in inputs[1:])]
    first, second = evaluate_capacity(*inputs), evaluate_capacity(*inputs)
    for a,b in zip(first,second):assert_frame_equal(a,b)
    for a,b in zip([inputs[0]['s'], *inputs[1:]],snapshots):assert_frame_equal(a,b)


def test_capacity_bound_matches_worst_trade_and_zero_cost_reference():
    inputs = fixture()
    config = replace(ExecutionConfig(), aums=(100_000.,), spread_slippage_bps=0., impact_bps_at_reference=0.)
    trades, periods, _, capacities, _ = evaluate_capacity(*inputs, config)
    assert periods.estimated_cost_bps.eq(0).all()
    assert np.allclose(periods.diagnostic_net_return, periods.gross_return)
    for row in capacities.itertuples():
        group = trades[trades.timing.eq(row.timing) & trades.trade_weight.ne(0)]
        assert (row.diagnostic_aum_ceiling*group.trade_weight.abs()/group.ADV_dollars).max() == pytest.approx(row.max_participation)


def test_unavailable_liquidity_is_explicit(tmp_path):
    with pytest.raises(ValueError, match='LIQUIDITY_DATA_UNAVAILABLE'):
        resolve_liquidity_path(tmp_path/'absent.parquet')


def test_v1_frozen_official_csvs_match_release():
    # Existing release artifacts are the reference, not values recomputed by this extension.
    hashes = json.loads((Path(__file__).parent / 'fixtures/v1_0_csv_hashes.json').read_text())
    assert len(hashes) == 39
    for name, digest in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name


def test_trade_weights_use_drift_at_actual_execution_date():
    inputs = fixture()
    trades, periods, *_ = evaluate_capacity(*inputs, ExecutionConfig(aums=(100_000.,)))
    for mode, price, start, end in (
        ('same_close_diagnostic', inputs[1], '2024-01-31', '2024-02-29'),
        ('next_session_open', inputs[2], '2024-02-01', '2024-03-01'),
    ):
        values = pd.Series({'A':.5,'B':.5}) * price.loc[end] / price.loc[start]
        expected = values / values.sum()
        actual = trades[trades.timing.eq(mode) & trades.signal_date.eq('2024-02-29')].set_index('ticker')
        assert np.allclose(actual.pretrade_weight, expected)
        turnover = .5 * (pd.Series({'A':.8,'B':.2})-expected).abs().sum()
        row = periods[periods.timing.eq(mode) & periods.signal_date.eq('2024-02-29')].iloc[0]
        assert row.turnover == pytest.approx(turnover)


def test_adv60_and_terminal_session_exclusion():
    dates = pd.bdate_range('2024-01-01',periods=65)
    price = pd.DataFrame({'A':10.},index=dates)
    adv = calculate_adv(price, price*10, 60)
    assert adv.iloc[:60].isna().all().all()
    assert adv.iloc[60,0] == 1000
    inputs = fixture()
    truncated_close = inputs[1].loc[:'2024-04-30']
    _, periods, _, _, exclusions = evaluate_capacity(inputs[0],truncated_close,inputs[2].loc[truncated_close.index],inputs[3].loc[truncated_close.index])
    assert periods.signal_date.max() == pd.Timestamp('2024-02-29')
    assert exclusions.iloc[-1].reason == 'terminal_next_session_unavailable'


@pytest.mark.parametrize('parameters', [{'aums':(0.,)}, {'aums':(float('nan'),)}, {'spread_slippage_bps':-1}, {'participation_limits':(2.,)}])
def test_invalid_scenario_assumptions_fail(parameters):
    with pytest.raises(ValueError):
        ExecutionConfig(**parameters)


def test_noncontiguous_targets_cannot_silently_skip_drift():
    inputs = fixture()
    inputs[0]['s'] = inputs[0]['s'].iloc[[0,2]]
    with pytest.raises(ValueError, match='NONCONTIGUOUS_PAIRED_SCHEDULE'):
        evaluate_capacity(*inputs)


def test_numerical_dust_is_not_a_liquidity_trade():
    target = pd.Series({'A': .5 + 1e-16, 'B': .5 - 1e-16})
    trades = trade_diagnostics(target, pd.Series({'A':.5,'B':.5}), pd.Series({'A':1e6,'B':1e6}), 1e8, ExecutionConfig())
    assert trades.trade_dollars.eq(0).all()
    assert trades.estimated_cost_dollars.eq(0).all()
