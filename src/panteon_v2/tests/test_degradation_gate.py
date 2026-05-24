from __future__ import annotations

import json
import os
import tempfile
import unittest

from panteon_v2.app import OutputWriter, OutputWriterConfig
from panteon_v2.app.bootstrap import build_production_pipeline
from panteon_v2.app.main_loop import StepResult, _apply_degradation_gate
from panteon_v2.attribution import QuarantineRecomputed
from panteon_v2.domain.types import Action, Metrics, Regime, Signal, Trade
from panteon_v2.execution import FakeExchange
from panteon_v2.memory import (
    DegradationGate,
    DegradationGateConfig,
    PerformanceMemory,
    QuarantineManager,
)
from panteon_v2.selection import AgentRegistry
from panteon_v2.tests._helpers import FakeAgent


def _add_trade(
    perf: PerformanceMemory,
    label: str,
    pnl_pct: float,
    *,
    regime: Regime = Regime.NEUTRAL,
    start_id: int = 1,
) -> None:
    open_price = 100.0
    close_price = open_price * (1.0 + pnl_pct / 100.0)
    open_signal = Signal(
        id=start_id,
        bar=start_id,
        sym=f"T{start_id}",
        action=Action.FUT_LONG_FULL,
        price=open_price,
        regime=regime,
        by_player=label,
        by_agent=label,
    )
    close_signal = Signal(
        id=start_id + 1,
        bar=start_id + 1,
        sym=f"T{start_id}",
        action=Action.FUT_CLOSE_ALL,
        price=close_price,
        regime=regime,
        by_player=label,
        by_agent=label,
    )
    perf.update_from_trade(
        Trade(
            signal_id=open_signal.id,
            bar=open_signal.bar,
            sym=open_signal.sym,
            side="long",
            qty=1.0,
            fill_price=open_price,
            fee=0.0,
        ),
        open_signal,
    )
    perf.update_from_trade(
        Trade(
            signal_id=close_signal.id,
            bar=close_signal.bar,
            sym=close_signal.sym,
            side="long",
            qty=1.0,
            fill_price=close_price,
            fee=0.0,
        ),
        close_signal,
    )


class TestDegradationGate(unittest.TestCase):
    def test_metrics_delta_keeps_trade_counts_consistent_and_net_fields(self) -> None:
        from panteon_v2.memory.degradation import _metrics_delta

        current = Metrics(
            pnl_pct=3.0,
            closed_trades=8,
            wins=5,
            losses=3,
            pnl_gross_pct=5.0,
            fee_pct=1.0,
            funding_pct=0.75,
        )
        baseline = Metrics(
            pnl_pct=1.0,
            closed_trades=5,
            wins=3,
            losses=0,
            pnl_gross_pct=2.0,
            fee_pct=0.25,
            funding_pct=0.25,
        )

        delta = _metrics_delta(current, baseline)

        self.assertEqual(delta.closed_trades, 3)
        self.assertEqual(delta.wins, 2)
        self.assertEqual(delta.losses, 1)
        self.assertLessEqual(delta.wins + delta.losses, delta.closed_trades)
        self.assertAlmostEqual(delta.pnl_pct, 2.0)
        self.assertAlmostEqual(delta.pnl_gross_pct, 3.0)
        self.assertAlmostEqual(delta.fee_pct, 0.75)
        self.assertAlmostEqual(delta.funding_pct, 0.5)

    def test_metrics_delta_never_reports_negative_fee_delta(self) -> None:
        from panteon_v2.memory.degradation import _metrics_delta

        current = Metrics(
            pnl_pct=1.0,
            closed_trades=3,
            wins=2,
            losses=1,
            fee_pct=0.25,
        )
        baseline = Metrics(
            pnl_pct=0.5,
            closed_trades=1,
            wins=1,
            losses=0,
            fee_pct=0.75,
        )

        delta = _metrics_delta(current, baseline)

        self.assertEqual(delta.fee_pct, 0.0)

    def test_session_loss_quarantines_genetics_agent_after_min_activity(self) -> None:
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager()
        gate = DegradationGate(
            DegradationGateConfig(
                min_session_closed_trades=2,
                max_session_loss_pct=1.0,
                max_session_drawdown_pct=0.0,
                max_execution_failure_rate=0.0,
            )
        )
        gate.capture_baseline(perf, labels=["GeneticsNeutral"])

        _add_trade(perf, "GeneticsNeutral", -0.6, start_id=10)
        first = gate.apply(perf, qm, labels=["GeneticsNeutral"], bar=10)
        self.assertTrue(first.is_no_op)
        self.assertFalse(qm.is_quarantined("GeneticsNeutral"))

        _add_trade(perf, "GeneticsNeutral", -0.6, start_id=20)
        second = gate.apply(perf, qm, labels=["GeneticsNeutral"], bar=20)
        self.assertIn("GeneticsNeutral", second.added)
        self.assertTrue(qm.is_quarantined("GeneticsNeutral"))
        record = qm.record_for("GeneticsNeutral")
        self.assertIsNotNone(record)
        self.assertIn("degradation_gate:session_loss_pct", record.reason)

    def test_baseline_prevents_old_history_from_disabling_agent(self) -> None:
        perf = PerformanceMemory(trade_fraction=1.0)
        _add_trade(perf, "GeneticsBearish", -5.0, start_id=100)
        qm = QuarantineManager()
        gate = DegradationGate(
            DegradationGateConfig(
                min_session_closed_trades=1,
                max_session_loss_pct=1.0,
                max_session_drawdown_pct=0.0,
                max_execution_failure_rate=0.0,
            )
        )
        gate.capture_baseline(perf, labels=["GeneticsBearish"])

        result = gate.apply(perf, qm, labels=["GeneticsBearish"], bar=1)

        self.assertTrue(result.is_no_op)
        self.assertFalse(qm.is_quarantined("GeneticsBearish"))

    def test_default_gate_ignores_non_genetics_labels(self) -> None:
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager()
        gate = DegradationGate(
            DegradationGateConfig(
                min_session_closed_trades=1,
                max_session_loss_pct=0.1,
                max_session_drawdown_pct=0.0,
                max_execution_failure_rate=0.0,
            )
        )
        gate.capture_baseline(perf, labels=["LiveTrendFollow"])
        _add_trade(perf, "LiveTrendFollow", -10.0, start_id=200)

        result = gate.apply(perf, qm, labels=["LiveTrendFollow"], bar=2)

        self.assertTrue(result.is_no_op)
        self.assertFalse(qm.is_quarantined("LiveTrendFollow"))

    def test_execution_failure_rate_can_disable_without_closed_trades(self) -> None:
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager()
        gate = DegradationGate(
            DegradationGateConfig(
                min_session_signals=3,
                min_session_closed_trades=99,
                max_session_loss_pct=0.0,
                max_session_drawdown_pct=0.0,
                max_execution_failure_rate=0.50,
                max_blocked_signal_rate=0.0,
            )
        )
        gate.capture_baseline(perf, labels=["GeneticsBullish"])
        for _ in range(3):
            perf.record_actor_failure("GeneticsBullish", Regime.BULLISH)

        result = gate.apply(perf, qm, labels=["GeneticsBullish"], bar=3)

        self.assertIn("GeneticsBullish", result.added)
        record = qm.record_for("GeneticsBullish")
        self.assertIsNotNone(record)
        self.assertIn("execution_failure_rate", record.reason)

    def test_blocked_shadow_signals_do_not_trip_execution_failure_gate(self) -> None:
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager()
        gate = DegradationGate(
            DegradationGateConfig(
                min_session_signals=3,
                min_session_closed_trades=99,
                max_session_loss_pct=0.0,
                max_session_drawdown_pct=0.0,
                max_execution_failure_rate=0.50,
                max_blocked_signal_rate=0.0,
            )
        )
        label = "GeneticsNeutral"
        gate.capture_baseline(perf, labels=[label])
        for idx in range(3):
            signal = Signal(
                id=500 + idx,
                bar=idx,
                sym=f"B{idx}",
                action=Action.FUT_LONG_FULL,
                price=100.0,
                regime=Regime.BULLISH,
                by_player=label,
                by_agent=label,
            )
            perf.record_execution_outcome(
                signal,
                status="blocked",
                reason="shadow risk limit",
            )

        result = gate.apply(perf, qm, labels=[label], bar=10)

        self.assertTrue(result.is_no_op)
        self.assertFalse(qm.is_quarantined(label))
        decision = gate.last_decisions[0]
        self.assertEqual(decision.session_metrics.blocked_signals, 3)
        self.assertEqual(decision.execution_failure_rate, 0.0)

    def test_blocked_rate_can_disable_non_executable_genetics_agent(self) -> None:
        perf = PerformanceMemory(trade_fraction=1.0)
        qm = QuarantineManager()
        gate = DegradationGate(
            DegradationGateConfig(
                min_session_signals=5,
                min_session_closed_trades=99,
                max_session_loss_pct=0.0,
                max_session_drawdown_pct=0.0,
                max_execution_failure_rate=0.0,
                max_blocked_signal_rate=0.80,
            )
        )
        label = "GeneticsBullish"
        gate.capture_baseline(perf, labels=[label])
        for idx in range(5):
            signal = Signal(
                id=600 + idx,
                bar=idx,
                sym=f"X{idx}",
                action=Action.FUT_LONG_FULL,
                price=100.0,
                regime=Regime.BULLISH,
                by_player=label,
                by_agent=label,
            )
            perf.record_execution_outcome(signal, status="blocked")

        result = gate.apply(perf, qm, labels=[label], bar=11)

        self.assertIn(label, result.added)
        record = qm.record_for(label)
        self.assertIsNotNone(record)
        self.assertIn("blocked_signal_rate", record.reason)
        self.assertEqual(gate.last_decisions[0].blocked_signal_rate, 1.0)

    def test_main_loop_helper_applies_gate_and_emits_event(self) -> None:
        registry = AgentRegistry()
        registry.register(FakeAgent("GeneticsNeutral"))
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=FakeExchange(name="DRY-RUN"),
            initial_capital=100.0,
            degradation_config=DegradationGateConfig(
                min_session_closed_trades=1,
                max_session_loss_pct=0.1,
                max_session_drawdown_pct=0.0,
                max_execution_failure_rate=0.0,
            ),
        )
        pipeline.degradation_gate.capture_baseline(
            pipeline.perf,
            labels=pipeline.registry.all_labels(),
        )
        _add_trade(pipeline.perf, "GeneticsNeutral", -1.0, start_id=300)

        result = _apply_degradation_gate(pipeline, bar=5, trace_id="test-5")

        self.assertIn("GeneticsNeutral", result.added)
        self.assertTrue(pipeline.qm.is_quarantined("GeneticsNeutral"))
        events = list(pipeline.event_log.query(event_types=[QuarantineRecomputed]))
        self.assertEqual(len(events), 1)
        self.assertIn("GeneticsNeutral", events[0].added)

    def test_output_status_exposes_degradation_quarantine_reason(self) -> None:
        registry = AgentRegistry()
        registry.register(FakeAgent("GeneticsNeutral"))
        pipeline = build_production_pipeline(
            registry=registry,
            exchange=FakeExchange(name="DRY-RUN"),
            initial_capital=100.0,
            degradation_config=DegradationGateConfig(
                min_session_closed_trades=1,
                max_session_loss_pct=0.1,
                max_session_drawdown_pct=0.0,
                max_execution_failure_rate=0.0,
            ),
        )
        pipeline.degradation_gate.capture_baseline(
            pipeline.perf,
            labels=pipeline.registry.all_labels(),
        )
        _add_trade(pipeline.perf, "GeneticsNeutral", -1.0, start_id=400)
        _apply_degradation_gate(pipeline, bar=7, trace_id="test-7")

        with tempfile.TemporaryDirectory() as tmp:
            writer = OutputWriter(
                pipeline,
                OutputWriterConfig(output_dir=tmp, full_snapshot_every=1),
            )
            writer.write(
                StepResult(
                    bar=7,
                    regime=Regime.NEUTRAL,
                    leader=None,
                    leader_changed=False,
                    n_signals=0,
                    n_filled=0,
                    n_rejected=0,
                    n_blocked=0,
                )
            )
            with open(os.path.join(tmp, "status.json"), "r", encoding="utf-8") as f:
                status = json.load(f)

        record = status["quarantine_records"]["GeneticsNeutral"]
        self.assertIn("degradation_gate:session_loss_pct", record["reason"])
        gate_status = status["degradation_gate"]
        self.assertTrue(gate_status["baseline_captured"])
        self.assertEqual(
            gate_status["last_decisions"][0]["label"],
            "GeneticsNeutral",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
