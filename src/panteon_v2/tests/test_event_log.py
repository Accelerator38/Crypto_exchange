"""Тесты EventLog skeleton."""

from __future__ import annotations

import os
import tempfile
import unittest

from panteon_v2.attribution import (
    BarEnded,
    BarStarted,
    CandidateScored,
    EventLog,
    EventLogPersistenceError,
    ExecutionAttributed,
    LeaderSelected,
    QuarantineRecomputed,
    RegimeDetected,
    SignalEmitted,
)
from panteon_v2.domain.types import Action, Regime, Signal


class TestEventLog(unittest.TestCase):
    def test_emit_and_query_all(self):
        log = EventLog()
        log.emit(BarStarted(bar=1, trace_id="t1"))
        log.emit(BarEnded(bar=1, trace_id="t1"))
        self.assertEqual(len(log), 2)
        self.assertEqual(len(log.all()), 2)

    def test_query_by_trace(self):
        log = EventLog()
        log.emit(BarStarted(bar=1, trace_id="t1"))
        log.emit(BarStarted(bar=2, trace_id="t2"))
        log.emit(BarEnded(bar=1, trace_id="t1"))
        out = list(log.query(trace_id="t1"))
        self.assertEqual(len(out), 2)
        for ev in out:
            self.assertEqual(ev.trace_id, "t1")

    def test_query_by_event_type(self):
        log = EventLog()
        log.emit(BarStarted(bar=1, trace_id="t1"))
        log.emit(RegimeDetected(bar=1, trace_id="t1", regime=Regime.BULLISH))
        log.emit(BarEnded(bar=1, trace_id="t1"))
        bars_started = list(log.query(event_types=[BarStarted]))
        self.assertEqual(len(bars_started), 1)
        regimes = list(log.query(event_types=[RegimeDetected]))
        self.assertEqual(len(regimes), 1)
        self.assertEqual(regimes[0].regime, Regime.BULLISH)

    def test_query_by_bar_range(self):
        log = EventLog()
        for b in range(5):
            log.emit(BarStarted(bar=b, trace_id=f"t{b}"))
        out = list(log.query(after_bar=2, before_bar=4))
        self.assertEqual([e.bar for e in out], [2, 3])

    def test_emit_many(self):
        log = EventLog()
        log.emit_many([
            BarStarted(bar=1, trace_id="t1"),
            BarStarted(bar=2, trace_id="t2"),
        ])
        self.assertEqual(len(log), 2)

    def test_emit_rejects_non_event(self):
        log = EventLog()
        with self.assertRaises(TypeError):
            log.emit("not an event")  # type: ignore[arg-type]

    def test_jsonl_persistence(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "events.jsonl")
            log = EventLog(jsonl_path=path)
            log.emit(BarStarted(bar=1, trace_id="t1"))
            log.emit(LeaderSelected(
                bar=1, trace_id="t1",
                player_label="Ensemble", score=0.5, reason="test",
            ))
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("BarStarted", lines[0])
            self.assertIn("LeaderSelected", lines[1])

    def test_jsonl_write_failure_is_not_silent(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "events.jsonl")
            log = EventLog(jsonl_path=path)
            log._jsonl_path = os.path.join(td, "missing", "events.jsonl")  # type: ignore[attr-defined]

            with self.assertRaises(EventLogPersistenceError):
                log.emit(BarStarted(bar=1, trace_id="t1"))

    def test_jsonl_write_failure_can_be_non_fatal_for_live(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "events.jsonl")
            log = EventLog(
                jsonl_path=path,
                raise_on_persist_error=False,
                persist_retry_attempts=0,
            )
            log._jsonl_path = os.path.join(td, "missing", "events.jsonl")  # type: ignore[attr-defined]

            log.emit(BarStarted(bar=1, trace_id="t1"))

            self.assertEqual(len(log), 1)
            self.assertEqual(log.persistence_dropped, 1)
            self.assertIn("BarStarted", log.persistence_errors[-1])

    def test_candidate_scored_event_serializes_full_audit_fields(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "events.jsonl")
            log = EventLog(jsonl_path=path)
            log.emit(CandidateScored(
                bar=7,
                trace_id="decision-7",
                decision_id="decision-7",
                player_label="P",
                rank=1,
                score=-0.05,
                score_source="no_data",
                selected_by_pantheon=True,
                memory_keys_read=("P|bullish", "A|bullish"),
                agent_labels=("A",),
                recent_bars=12,
                recent_actionable_bars=6,
                actionable_share=0.5,
                recent_filled=4,
                recent_pnl_usd=9.5,
            ))

            with open(path, "r", encoding="utf-8") as f:
                payload = f.read()
            self.assertIn("CandidateScored", payload)
            self.assertIn("memory_keys_read", payload)
            self.assertIn("selected_by_pantheon", payload)
            self.assertIn("actionable_share", payload)
            self.assertIn("recent_filled", payload)

    def test_execution_attributed_event_serializes_outcome_bucket(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "events.jsonl")
            log = EventLog(jsonl_path=path)
            log.emit(ExecutionAttributed(
                bar=8,
                trace_id="decision-8",
                decision_id="decision-8",
                exchange="BITGET",
                symbol="BTC",
                timeframe="bridge_poll",
                mode="live_futures",
                run_id="run-8",
                session_id="session-8",
                signal_id=42,
                sym="BTC",
                action="FUT_LONG_FULL",
                status="blocked",
                attribution_bucket="BLOCKED",
                reason="risk_limits: external recovered position",
                realized_pnl=0.0,
                fees=0.0,
                slippage_pct=0.0,
                latency_ms=12.5,
                order_id="BITGET:BTC:long:decision-8:8",
                owner_scope="panteon_owned",
            ))

            with open(path, "r", encoding="utf-8") as f:
                payload = f.read()
            self.assertIn("ExecutionAttributed", payload)
            self.assertIn("attribution_bucket", payload)
            self.assertIn("BLOCKED", payload)

    def test_signal_emitted_event(self):
        log = EventLog()
        sig = Signal(id=1, bar=10, sym="BTC", action=Action.FUT_LONG_FULL,
                     price=50000.0, regime=Regime.BULLISH, by_player="Ensemble")
        log.emit(SignalEmitted(bar=10, trace_id="t10", signal=sig))
        out = list(log.query(event_types=[SignalEmitted]))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].signal.sym, "BTC")

    def test_quarantine_recomputed_event(self):
        log = EventLog()
        log.emit(QuarantineRecomputed(
            bar=5, trace_id="t5",
            added=frozenset({"X"}),
            removed=frozenset({"Y"}),
            current=frozenset({"X", "Z"}),
        ))
        events = list(log.query(event_types=[QuarantineRecomputed]))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].added, frozenset({"X"}))
        self.assertEqual(events[0].removed, frozenset({"Y"}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
