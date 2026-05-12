"""Тесты EventLog skeleton."""

from __future__ import annotations

import os
import tempfile
import unittest

from panteon_v2.attribution import (
    BarEnded,
    BarStarted,
    EventLog,
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
