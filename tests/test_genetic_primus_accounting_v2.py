import numpy as np
import pandas as pd

from exia.genetic_primus.accounting_v2 import InsolventSleeveError, simulate_cash_targets


STEP = 4 * 60 * 60 * 1000


def _input(paths, directions):
    bars, targets = [], []
    for symbol, prices in paths.items():
        for index, (opening, closing) in enumerate(zip(prices[:-1], prices[1:])):
            timestamp = index * STEP
            bars.append({"timestamp": timestamp, "symbol": symbol, "open": opening, "close": closing})
            targets.append({"execution_timestamp": timestamp, "symbol": symbol,
                            "target": directions[symbol], "signal_timestamp": timestamp,
                            "model_sha256": "manual_fixed_policy"})
    return pd.DataFrame(bars), pd.DataFrame(targets)


def _run(paths, directions, cost=0.0, reset=()):
    bars, targets = _input(paths, directions)
    return simulate_cash_targets(
        bars, targets, symbols=tuple(paths), start_timestamp=0,
        end_timestamp=(len(next(iter(paths.values()))) - 1) * STEP,
        timeframe_ms=STEP, round_trip_cost_bps=cost, force_flat_timestamps=reset,
    )


def test_fixed_quantity_short_round_trip_price_is_zero_gross_pnl():
    result = _run({"A": [100., 110., 100.]}, {"A": -1})
    assert np.isclose(result.metrics["final_equity"], 1.)
    assert np.isclose(result.trades.net_pnl.sum(), 0.)
    assert np.isclose(result.metrics["max_drawdown"], .1)


def test_last_bar_compounds_and_exit_time_is_actual_candle_close():
    result = _run({"A": [100., 110., 121.]}, {"A": 1})
    assert np.isclose(result.metrics["final_equity"], 1.21)
    assert np.allclose(result.portfolio.net_return, [.1, .1])
    assert result.trades.exit_timestamp.iloc[0] == 2 * STEP
    assert result.events.event_index.is_unique
    assert result.events.timestamp.is_monotonic_increasing


def test_first_loss_is_in_drawdown_from_initial_capital():
    result = _run({"A": [100., 90.]}, {"A": 1})
    assert np.isclose(result.metrics["max_drawdown"], .1)
    assert result.events.iloc[0].equity == 1.


def test_fees_use_actual_entry_and_exit_notional():
    cost = 20.
    rate = cost / 20_000.
    result = _run({"A": [100., 120.]}, {"A": 1}, cost)
    quantity = 1. / (100. * (1. + rate))
    expected_fees = quantity * (100. + 120.) * rate
    expected_equity = 1. + quantity * 20. - expected_fees
    assert np.isclose(result.metrics["total_fees"], expected_fees)
    assert np.isclose(result.metrics["final_equity"], expected_equity)
    assert np.isclose(result.trades.net_pnl.sum(), expected_equity - 1.)


def test_no_implicit_cross_asset_rebalancing():
    result = _run({"A": [100., 200., 200.], "B": [100., 100., 200.]}, {"A": 1, "B": 1})
    assert np.isclose(result.metrics["final_equity"], 2.)
    assert not np.isclose(result.metrics["final_equity"], 2.25)
    assert np.allclose(result.trades.quantity, [.005, .005])


def test_fixed_schedule_cost_increase_cannot_improve_flat_price_pnl():
    values = [_run({"A": [100., 100., 100.]}, {"A": 1}, cost).metrics["final_equity"]
              for cost in (0., 16., 24.)]
    assert values[0] > values[1] > values[2]


def test_retrain_boundary_carries_quantity_if_target_unchanged():
    carry = _run({"A": [100., 105., 100.]}, {"A": 1}, 16.)
    reset = _run({"A": [100., 105., 100.]}, {"A": 1}, 16., (STEP,))
    assert carry.metrics["fills"] == 2
    assert reset.metrics["fills"] == 4
    assert carry.metrics["final_equity"] > reset.metrics["final_equity"]
    assert "retrain_forced_exit" in set(reset.trades.exit_reason)


def test_no_trade_has_no_costs_and_no_position():
    result = _run({"A": [100., 110., 80.]}, {"A": 0}, 24.)
    assert result.metrics["final_equity"] == 1.
    assert result.metrics["trades"] == 0
    assert result.metrics["total_fees"] == 0.


def test_missing_symbol_candle_fails_instead_of_reweighting():
    bars, targets = _input({"A": [100., 110., 120.], "B": [100., 100., 100.]}, {"A": 1, "B": 0})
    bars = bars.drop(bars.index[-1])
    try:
        simulate_cash_targets(bars, targets, symbols=("A", "B"), start_timestamp=0,
                              end_timestamp=2 * STEP, timeframe_ms=STEP, round_trip_cost_bps=16.)
    except ValueError as exc:
        assert "missing/nonfinite" in str(exc)
    else:
        raise AssertionError("missing candle accepted")


def test_future_signal_is_rejected():
    bars, targets = _input({"A": [100., 110.]}, {"A": 1})
    targets.loc[0, "signal_timestamp"] = STEP
    try:
        simulate_cash_targets(bars, targets, symbols=("A",), start_timestamp=0,
                              end_timestamp=STEP, timeframe_ms=STEP, round_trip_cost_bps=0.)
    except ValueError as exc:
        assert "future signal" in str(exc)
    else:
        raise AssertionError("future signal accepted")


def test_every_decision_has_model_identity_and_ledger_reconciles():
    result = _run({"A": [100., 90., 120.], "B": [100., 110., 95.]}, {"A": 1, "B": -1}, 16.)
    assert set(result.decisions.model_sha256) == {"manual_fixed_policy"}
    assert result.metrics["maximum_reconciliation_error"] < 1e-12
    assert np.isclose(result.trades.net_pnl.sum(), result.metrics["net_return"])
    assert np.allclose(result.events.equity - 1.,
                       result.events.realized_pnl + result.events.unrealized_pnl)


def test_future_bars_do_not_change_earlier_marks_or_decisions():
    first = _run({"A": [100., 110., 120., 130.]}, {"A": 1})
    second = _run({"A": [100., 110., 120., 70.]}, {"A": 1})
    pd.testing.assert_frame_equal(first.portfolio.iloc[:2], second.portfolio.iloc[:2])
    pd.testing.assert_frame_equal(first.decisions, second.decisions)


def test_unbounded_short_loss_fails_closed_without_equity_clipping():
    try:
        _run({"A": [100., 250.]}, {"A": -1})
    except InsolventSleeveError:
        pass
    else:
        raise AssertionError("insolvent short accepted")
