"""Tests for live-state guards before real execution."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from panteon_v2.app.live_state import (
    filter_real_signals_against_tracker,
    prepare_v2_agents_for_live_after_warmup,
    reconcile_tracker_with_exchange,
    sync_player_agents_to_real_positions,
)
from panteon_v2.app.bootstrap import LiveExecutionConfig
from panteon_v2.attribution import EventLog, PositionClosed
from panteon_v2.domain.types import Action, Regime, Signal, Trade
from panteon_v2.execution import ExchangePosition, PositionTracker
from panteon_v2.execution.position_tracker import TrackedPosition
from panteon_v2.memory import PerformanceMemory


def _signal(sid: int, action: Action, *, price: float = 100.0) -> Signal:
    return Signal(
        id=sid,
        bar=1,
        sym="BTC",
        action=action,
        price=price,
        regime=Regime.BULLISH,
        by_player="Leader",
        by_agent="Agent",
    )


def _external_position(sym: str = "BTC", side: str = "long") -> TrackedPosition:
    return TrackedPosition(
        open_signal_id=0,
        sym=sym,
        side=side,
        entry_price=100.0,
        qty=0.1,
        fee_open=0.0,
        by_player="RecoveredExchangePosition",
        by_agent="",
        opened_at=datetime.now(timezone.utc),
    )


def _pipeline_for_tracker(tracker: PositionTracker):
    class Executor:
        _tracker = tracker

    class Pipeline:
        executor = Executor()

    return Pipeline()


class _Player:
    agents = []


class TestLiveSignalGuard(unittest.TestCase):
    def test_drops_new_open_when_position_capacity_is_full(self):
        tracker = PositionTracker()
        for sid, sym in ((1, "BTC"), (2, "ETH")):
            open_sig = _signal(sid, Action.FUT_LONG_FULL, price=100.0)
            object.__setattr__(open_sig, "sym", sym)
            tracker.on_open(
                signal=open_sig,
                trade=Trade(
                    signal_id=sid,
                    bar=1,
                    sym=sym,
                    side="long",
                    qty=0.1,
                    fill_price=100.0,
                    fee=0.01,
                ),
            )
        new_open = _signal(3, Action.FUT_LONG_FULL, price=20.0)
        object.__setattr__(new_open, "sym", "SOL")

        result = filter_real_signals_against_tracker(
            [new_open],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
            max_open_positions=2,
        )

        self.assertEqual(result.signals, [])
        self.assertEqual(result.max_position_saturated_opens, 1)
        self.assertEqual(result.filtered, 1)
        self.assertIn("max_position_saturated:SOL:Agent", result.details)

    def test_reserved_new_opens_consume_open_cap_before_later_signals(self):
        tracker = PositionTracker()
        new_open = _signal(3, Action.FUT_LONG_FULL, price=20.0)
        object.__setattr__(new_open, "sym", "SOL")

        result = filter_real_signals_against_tracker(
            [new_open],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
            reserved_new_opens=1,
        )

        self.assertEqual(result.signals, [])
        self.assertEqual(result.rate_limited_opens, 1)
        self.assertEqual(result.filtered, 1)
        self.assertIn("rate_limited_open:SOL:Agent", result.details)

    def test_opposite_open_becomes_close_instead_of_duplicate_drop(self):
        tracker = PositionTracker()
        open_sig = _signal(1, Action.FUT_LONG_FULL, price=100.0)
        open_trade = Trade(
            signal_id=1,
            bar=1,
            sym="BTC",
            side="long",
            qty=0.1,
            fill_price=100.0,
            fee=0.01,
        )
        tracker.on_open(signal=open_sig, trade=open_trade)

        class Executor:
            _tracker = tracker

        class Pipeline:
            executor = Executor()

        class Player:
            agents = []

        result = filter_real_signals_against_tracker(
            [_signal(2, Action.FUT_SHORT_FULL, price=95.0)],
            player=Player(),
            pipeline=Pipeline(),
            bar_index=2,
            max_new_opens_per_bar=1,
        )

        self.assertEqual(result.duplicate_opens, 0)
        self.assertEqual(len(result.signals), 1)
        self.assertEqual(result.signals[0].action, Action.FUT_CLOSE_ALL)
        self.assertEqual(result.signals[0].id, 2)
        self.assertIn("reverse_close:BTC:Agent", result.details)

    def test_drops_foreign_close_for_owned_position(self):
        tracker = PositionTracker()
        open_sig = _signal(1, Action.FUT_SHORT_FULL, price=100.0)
        object.__setattr__(open_sig, "by_player", "GeneticsCore")
        object.__setattr__(open_sig, "by_agent", "GeneticsCore")
        tracker.on_open(
            signal=open_sig,
            trade=Trade(
                signal_id=1,
                bar=1,
                sym="BTC",
                side="short",
                qty=0.1,
                fill_price=100.0,
                fee=0.01,
            ),
        )
        close_sig = _signal(2, Action.FUT_CLOSE_ALL, price=101.0)
        object.__setattr__(close_sig, "by_player", "NeutralRangeScalper")
        object.__setattr__(close_sig, "by_agent", "NeutralRangeScalper")

        result = filter_real_signals_against_tracker(
            [close_sig],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
        )

        self.assertEqual(result.signals, [])
        self.assertEqual(result.foreign_closes, 1)
        self.assertEqual(result.filtered, 1)
        self.assertIn(
            "foreign_close:BTC:NeutralRangeScalper:owner=GeneticsCore",
            result.details,
        )

    def test_allows_owner_close_for_owned_position(self):
        tracker = PositionTracker()
        open_sig = _signal(1, Action.FUT_SHORT_FULL, price=100.0)
        object.__setattr__(open_sig, "by_player", "GeneticsCore")
        object.__setattr__(open_sig, "by_agent", "GeneticsCore")
        tracker.on_open(
            signal=open_sig,
            trade=Trade(
                signal_id=1,
                bar=1,
                sym="BTC",
                side="short",
                qty=0.1,
                fill_price=100.0,
                fee=0.01,
            ),
        )
        close_sig = _signal(2, Action.FUT_CLOSE_ALL, price=101.0)
        object.__setattr__(close_sig, "by_player", "GeneticsCore")
        object.__setattr__(close_sig, "by_agent", "GeneticsCore")

        result = filter_real_signals_against_tracker(
            [close_sig],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
        )

        self.assertEqual(result.signals, [close_sig])
        self.assertEqual(result.foreign_closes, 0)
        self.assertEqual(result.filtered, 0)

    def test_allows_ensemble_close_when_signal_agent_matches_owner_player(self):
        tracker = PositionTracker()
        open_sig = _signal(1, Action.FUT_SHORT_FULL, price=100.0)
        object.__setattr__(open_sig, "by_player", "GeneticsCore")
        object.__setattr__(open_sig, "by_agent", "")
        tracker.on_open(
            signal=open_sig,
            trade=Trade(
                signal_id=1,
                bar=1,
                sym="BTC",
                side="short",
                qty=0.1,
                fill_price=100.0,
                fee=0.01,
            ),
        )
        close_sig = _signal(2, Action.FUT_CLOSE_ALL, price=101.0)
        object.__setattr__(close_sig, "by_player", "DefaultEnsemble")
        object.__setattr__(close_sig, "by_agent", "GeneticsCore")

        result = filter_real_signals_against_tracker(
            [close_sig],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
        )

        self.assertEqual(result.signals, [close_sig])
        self.assertEqual(result.foreign_closes, 0)
        self.assertEqual(result.filtered, 0)

    def test_allows_system_guard_close_for_owned_position(self):
        tracker = PositionTracker()
        open_sig = _signal(1, Action.FUT_SHORT_FULL, price=100.0)
        object.__setattr__(open_sig, "by_player", "GeneticsCore")
        object.__setattr__(open_sig, "by_agent", "GeneticsCore")
        tracker.on_open(
            signal=open_sig,
            trade=Trade(
                signal_id=1,
                bar=1,
                sym="BTC",
                side="short",
                qty=0.1,
                fill_price=100.0,
                fee=0.01,
            ),
        )
        close_sig = _signal(2, Action.FUT_CLOSE_ALL, price=101.0)
        object.__setattr__(close_sig, "by_player", "Panteon_Flash")
        object.__setattr__(close_sig, "by_agent", "StalePositionGuard")

        result = filter_real_signals_against_tracker(
            [close_sig],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
        )

        self.assertEqual(result.signals, [close_sig])
        self.assertEqual(result.foreign_closes, 0)
        self.assertEqual(result.filtered, 0)

    def test_drops_foreign_opposite_open_for_owned_position(self):
        tracker = PositionTracker()
        open_sig = _signal(1, Action.FUT_LONG_FULL, price=100.0)
        object.__setattr__(open_sig, "by_player", "GeneticsCore")
        object.__setattr__(open_sig, "by_agent", "GeneticsCore")
        tracker.on_open(
            signal=open_sig,
            trade=Trade(
                signal_id=1,
                bar=1,
                sym="BTC",
                side="long",
                qty=0.1,
                fill_price=100.0,
                fee=0.01,
            ),
        )
        opposite = _signal(2, Action.FUT_SHORT_FULL, price=95.0)
        object.__setattr__(opposite, "by_player", "NeutralRangeScalper")
        object.__setattr__(opposite, "by_agent", "NeutralRangeScalper")

        result = filter_real_signals_against_tracker(
            [opposite],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
        )

        self.assertEqual(result.signals, [])
        self.assertEqual(result.foreign_closes, 1)
        self.assertEqual(result.filtered, 1)
        self.assertIn(
            "foreign_reverse_close:BTC:NeutralRangeScalper:owner=GeneticsCore",
            result.details,
        )

    def test_drops_close_for_external_recovered_position(self):
        tracker = PositionTracker()
        tracker.force_set(_external_position())

        result = filter_real_signals_against_tracker(
            [_signal(2, Action.FUT_CLOSE_ALL, price=95.0)],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
        )

        self.assertEqual(result.signals, [])
        self.assertEqual(result.external_position_signals, 1)
        self.assertEqual(result.filtered, 1)
        self.assertIn("external_position_signal:BTC:Agent", result.details)

    def test_drops_opposite_open_on_external_instead_of_reverse_close(self):
        tracker = PositionTracker()
        tracker.force_set(_external_position(side="long"))

        result = filter_real_signals_against_tracker(
            [_signal(2, Action.FUT_SHORT_FULL, price=95.0)],
            player=_Player(),
            pipeline=_pipeline_for_tracker(tracker),
            bar_index=2,
            max_new_opens_per_bar=1,
        )

        self.assertEqual(result.signals, [])
        self.assertEqual(result.external_position_signals, 1)
        self.assertNotIn("reverse_close:BTC:Agent", result.details)

    def test_sync_skips_external_recovered_positions_for_agents(self):
        tracker = PositionTracker()
        tracker.force_set(_external_position())

        class Agent:
            pos = {"BTC": "short"}
            ep = {"BTC": 90.0}
            et = {"BTC": 1}

        class Player:
            agents = [Agent()]

        summary = sync_player_agents_to_real_positions(
            Player(),
            _pipeline_for_tracker(tracker),
            bar_index=2,
            market_symbols=["BTC"],
        )

        self.assertEqual(summary["real_positions"], 1)
        self.assertEqual(summary["agent_positions"], 0)
        self.assertEqual(summary["external_positions_skipped"], 1)
        self.assertIsNone(Agent.pos["BTC"])
        self.assertEqual(Agent.ep["BTC"], 0.0)
        self.assertEqual(Agent.et["BTC"], 0)

    def test_prepare_live_clears_and_injects_genetics_position_state(self):
        tracker = PositionTracker()
        open_sig = _signal(1, Action.FUT_SHORT_FULL, price=100.0)
        object.__setattr__(open_sig, "by_player", "GeneticsCore")
        object.__setattr__(open_sig, "by_agent", "GeneticsCore")
        tracker.on_open(
            signal=open_sig,
            trade=Trade(
                signal_id=1,
                bar=1,
                sym="BTC",
                side="short",
                qty=0.25,
                fill_price=100.0,
                fee=0.01,
            ),
        )

        class GeneticsRuntime:
            spot_qty = {"SOL": 2.0}
            spot_entry = {"SOL": 25.0}
            fut_qty = {"SOL": -1.2}
            fut_entry = {"SOL": 24.0}
            pos = {"SOL": "fut_s"}
            updates = []

            def update_from_exchange(self, symbol, spot_qty, spot_entry, fut_qty, fut_entry):
                self.updates.append((symbol, spot_qty, spot_entry, fut_qty, fut_entry))
                self.spot_qty[symbol] = spot_qty
                self.spot_entry[symbol] = spot_entry
                self.fut_qty[symbol] = fut_qty
                self.fut_entry[symbol] = fut_entry
                self.pos[symbol] = "fut_s" if fut_qty < 0 else "fut_l" if fut_qty > 0 else None

        class Agent:
            label = "GeneticsBest"
            v1_agent = GeneticsRuntime()

        class Registry:
            def all_agents(self):
                return [Agent()]

        class Pipeline:
            registry = Registry()
            executor = _pipeline_for_tracker(tracker).executor

        summary = prepare_v2_agents_for_live_after_warmup(
            Pipeline(),
            exchange_name="MEXC",
            bar_index=10,
        )

        runtime = Agent.v1_agent
        self.assertEqual(summary["agent_positions"], 1)
        self.assertIsNone(runtime.pos["SOL"])
        self.assertEqual(runtime.spot_qty["SOL"], 0.0)
        self.assertEqual(runtime.spot_entry["SOL"], 0.0)
        self.assertEqual(runtime.fut_qty["SOL"], 0.0)
        self.assertEqual(runtime.fut_entry["SOL"], 0.0)
        self.assertEqual(runtime.fut_qty["BTC"], -0.25)
        self.assertEqual(runtime.fut_entry["BTC"], 100.0)
        self.assertIn(("BTC", 0.0, 0.0, -0.25, 100.0), runtime.updates)

    def test_prepare_live_resets_direct_agent_check_timer_after_warmup(self):
        class Agent:
            label = "LiveVolCompress"
            CHECK_INT = 3
            _lc = 6380
            pos = {"BTC": "long"}
            ep = {"BTC": 100.0}
            et = {"BTC": 6370}

        agent = Agent()

        class Registry:
            def all_agents(self):
                return [agent]

        class Pipeline:
            registry = Registry()
            executor = _pipeline_for_tracker(PositionTracker()).executor

        prepare_v2_agents_for_live_after_warmup(
            Pipeline(),
            exchange_name="MEXC",
            bar_index=6380,
        )

        self.assertIsNone(Agent.pos["BTC"])
        self.assertEqual(Agent.ep["BTC"], 0.0)
        self.assertEqual(Agent.et["BTC"], 0)
        self.assertLessEqual(agent._lc, 6380 - Agent.CHECK_INT)

    def test_sync_preserves_tracker_opened_bar_for_time_based_exits(self):
        tracker = PositionTracker()
        open_sig = _signal(1, Action.FUT_LONG_FULL, price=100.0)
        object.__setattr__(open_sig, "bar", 17)
        tracker.on_open(
            signal=open_sig,
            trade=Trade(
                signal_id=1,
                bar=17,
                sym="BTC",
                side="long",
                qty=0.25,
                fill_price=100.0,
                fee=0.01,
            ),
        )

        class Agent:
            pos = {"BTC": None}
            ep = {"BTC": 0.0}
            et = {"BTC": 0}

        class Player:
            agents = [Agent()]

        sync_player_agents_to_real_positions(
            Player(),
            _pipeline_for_tracker(tracker),
            bar_index=250,
            market_symbols=["BTC"],
        )

        self.assertEqual(Agent.pos["BTC"], "long")
        self.assertEqual(Agent.ep["BTC"], 100.0)
        self.assertEqual(Agent.et["BTC"], 17)


class TestLiveExchangeReconcile(unittest.TestCase):
    def test_reconcile_adds_exchange_position_missing_from_tracker(self):
        tracker = PositionTracker()

        class Exchange:
            def get_all_positions(self):
                return {
                    "BTC": ExchangePosition(
                        sym="BTC",
                        side="long",
                        qty=0.25,
                        entry=101.0,
                    )
                }

        class Executor:
            _tracker = tracker
            _exchange = Exchange()

        class Pipeline:
            executor = Executor()

        summary = reconcile_tracker_with_exchange(Pipeline(), bar_index=7)

        self.assertEqual(summary["added"], 1)
        pos = tracker.get("BTC")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.by_player, "RecoveredExchangePosition")
        self.assertAlmostEqual(pos.qty, 0.25)
        self.assertAlmostEqual(pos.entry_price, 101.0)

    def test_reconcile_adopts_unknown_exchange_position_when_enabled(self):
        tracker = PositionTracker()
        perf = PerformanceMemory(trade_fraction=1.0)

        class Exchange:
            def get_all_positions(self):
                return {
                    "BTC": ExchangePosition(
                        sym="BTC",
                        side="short",
                        qty=0.25,
                        entry=101.0,
                    )
                }

        class Executor:
            _tracker = tracker
            _exchange = Exchange()

        class Pipeline:
            executor = Executor()
            live_execution = LiveExecutionConfig(
                adopt_existing_positions_enabled=True,
            )

        pipeline = Pipeline()
        pipeline.perf = perf

        summary = reconcile_tracker_with_exchange(pipeline, bar_index=7)

        self.assertEqual(summary["added"], 1)
        self.assertEqual(summary["adopted_added"], 1)
        self.assertEqual(summary["external_added"], 0)
        pos = tracker.get("BTC")
        self.assertIsNotNone(pos)
        self.assertEqual(pos.by_player, "PanteonFlashAdopted")
        self.assertEqual(pos.by_agent, "AdoptedExchangePosition")
        self.assertEqual(pos.open_action, "FUT_SHORT_FULL")
        self.assertEqual(pos.open_regime, "neutral")
        opened = perf.snapshot()["open"]
        self.assertIn("AdoptedExchangePosition|BTC", opened)
        self.assertIn("PanteonFlashAdopted|BTC", opened)

    def test_reconcile_does_not_remove_positions_on_unreliable_empty_snapshot(self):
        tracker = PositionTracker()
        tracker.force_set(TrackedPosition(
            open_signal_id=7,
            sym="BTC",
            side="long",
            entry_price=100.0,
            qty=0.1,
            fee_open=0.0,
            by_player="PlayerA",
            by_agent="AgentA",
            opened_at=datetime.now(timezone.utc),
        ))
        event_log = EventLog()

        class Exchange:
            def get_all_positions(self):
                return {}

            def positions_snapshot_reliable(self):
                return False

            def positions_snapshot_error(self):
                return "cached_futures_assets"

        class Executor:
            _tracker = tracker
            _exchange = Exchange()

        class Pipeline:
            executor = Executor()
            account_snapshot = {
                "data_health": {
                    "uses_cached_balance": True,
                    "last_data_error_reason": "cached_futures_assets",
                }
            }

        pipeline = Pipeline()
        pipeline.event_log = event_log

        summary = reconcile_tracker_with_exchange(pipeline, bar_index=8)

        self.assertEqual(summary["removed"], 0)
        self.assertEqual(summary["owned_removed"], 0)
        self.assertEqual(summary["removal_guarded"], 1)
        self.assertTrue(summary["snapshot_unreliable"])
        self.assertEqual(summary["guard_reason"], "cached_futures_assets")
        self.assertIsNotNone(tracker.get("BTC"))
        self.assertEqual(list(event_log.query(event_types=[PositionClosed])), [])

    def test_reconcile_external_close_keeps_forensic_context(self):
        tracker = PositionTracker()
        tracker.force_set(_external_position())
        event_log = EventLog()

        class Exchange:
            def get_all_positions(self):
                return {}

        class Executor:
            _tracker = tracker
            _exchange = Exchange()

        class Pipeline:
            executor = Executor()
            exchange_name = "BITGET"
            timeframe = "bridge_poll"
            mode = "live_futures"
            run_id = "run-7"
            session_id = "session-7"

        pipeline = Pipeline()
        pipeline.event_log = event_log
        summary = reconcile_tracker_with_exchange(pipeline, bar_index=7)

        self.assertEqual(summary["removed"], 1)
        closed = list(event_log.query(event_types=[PositionClosed]))
        self.assertEqual(len(closed), 1)
        ev = closed[0]
        self.assertEqual(ev.by_player, "RecoveredExchangePosition")
        self.assertEqual(ev.decision_id, "exchange-reconcile-7-BTC")
        self.assertEqual(ev.exchange, "BITGET")
        self.assertEqual(ev.symbol, "BTC")
        self.assertEqual(ev.timeframe, "bridge_poll")
        self.assertEqual(ev.mode, "live_futures")
        self.assertEqual(ev.run_id, "run-7")
        self.assertEqual(ev.session_id, "session-7")


if __name__ == "__main__":
    unittest.main(verbosity=2)
