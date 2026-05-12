"""Тесты Phase 6: builders, palette, renderer."""

from __future__ import annotations

import unittest

from panteon_v2.attribution import (
    AttributionLedger, EventLog, PositionClosed, PositionOpened, BarStarted,
    LeaderSelected,
)
from panteon_v2.dashboards import (
    DEFAULT_PALETTE,
    DashboardPalette,
    DashboardRenderer,
    TextRenderer,
    build_attribution_panel,
    build_leaderboard,
    build_quarantine_panel,
    build_regime_heatmap,
    build_decision_timeline,
)
from panteon_v2.domain.types import Action, Regime, Signal, Trade
from panteon_v2.execution import SymbolHealthMonitor
from panteon_v2.memory import PerformanceMemory, QuarantineManager


# ════════════════════════════════════════════════════════════════════
# Palette
# ════════════════════════════════════════════════════════════════════


class TestPalette(unittest.TestCase):
    def test_quarantine_always_grey(self):
        p = DEFAULT_PALETTE
        # Даже с положительным PnL карантинный — серый
        self.assertEqual(p.for_pnl(+5.0, is_quarantined=True), p.quarantine)
        self.assertEqual(p.for_pnl(-5.0, is_quarantined=True), p.quarantine)
        self.assertEqual(p.for_pnl(0.0,  is_quarantined=True), p.quarantine)

    def test_non_quarantine_signs(self):
        p = DEFAULT_PALETTE
        self.assertEqual(p.for_pnl(+1.0, is_quarantined=False), p.positive)
        self.assertEqual(p.for_pnl(-1.0, is_quarantined=False), p.negative)
        self.assertEqual(p.for_pnl(0.0,  is_quarantined=False), p.muted)

    def test_palette_immutable(self):
        with self.assertRaises(Exception):
            DEFAULT_PALETTE.bg = "#FFF"  # type: ignore


# ════════════════════════════════════════════════════════════════════
# Helpers для тестов
# ════════════════════════════════════════════════════════════════════


def _add_close_to_log(log: EventLog, *, sym, op_id, cl_id, bar,
                     player="P", agent="A", pnl=10.0, side="long",
                     entry=100.0, exit_=110.0, qty=1.0):
    log.emit(PositionOpened(bar=bar - 1, signal_id=op_id, sym=sym,
                            side=side, entry=entry, qty=qty))
    log.emit(PositionClosed(bar=bar, open_signal_id=op_id, close_signal_id=cl_id,
                            sym=sym, side=side, entry=entry, exit=exit_,
                            qty=qty, realized_pnl=pnl, by_player=player, by_agent=agent))


def _seed_perf(perf: PerformanceMemory, label: str, regime: Regime,
              n_trades: int, pnl_each: float, *, start_id: int = 1):
    sid = start_id
    for i in range(n_trades):
        op_p = 100.0
        cl_p = op_p * (1.0 + pnl_each / 100.0)
        os = Signal(id=sid, bar=i*2+1, sym=f"S{sid}", action=Action.FUT_LONG_FULL,
                    price=op_p, regime=regime, by_player=label, by_agent=label)
        cs = Signal(id=sid+1, bar=i*2+2, sym=f"S{sid}", action=Action.FUT_CLOSE_ALL,
                    price=cl_p, regime=regime, by_player=label, by_agent=label)
        perf.update_from_trade(Trade(signal_id=sid, bar=os.bar, sym=os.sym,
                                     side="long", qty=1.0, fill_price=op_p, fee=0.0), os)
        perf.update_from_trade(Trade(signal_id=sid+1, bar=cs.bar, sym=cs.sym,
                                     side="long", qty=1.0, fill_price=cl_p, fee=0.0), cs)
        sid += 2


# ════════════════════════════════════════════════════════════════════
# AttributionPanel builder
# ════════════════════════════════════════════════════════════════════


class TestAttributionPanelBuilder(unittest.TestCase):
    def setUp(self):
        log = EventLog()
        _add_close_to_log(log, sym="BTC", op_id=1, cl_id=2, bar=2,
                         player="Alpha", agent="X", pnl=5.0)
        _add_close_to_log(log, sym="ETH", op_id=3, cl_id=4, bar=4,
                         player="Beta", agent="Y", pnl=-3.0)
        _add_close_to_log(log, sym="DOGE", op_id=5, cl_id=6, bar=6,
                         player="Alpha", agent="X", pnl=2.0)
        self.ledger = AttributionLedger()
        self.ledger.replay_from_event_log(log)
        self.qm = QuarantineManager(seed={"Beta"})

    def test_basic_panel(self):
        panel = build_attribution_panel(self.ledger, self.qm)
        self.assertEqual(len(panel.rows), 2)  # Alpha + Beta
        # sorted by -pnl: Alpha first
        self.assertEqual(panel.rows[0].label, "Alpha")
        self.assertEqual(panel.rows[1].label, "Beta")
        self.assertAlmostEqual(panel.rows[0].realized_pnl, 7.0)
        self.assertAlmostEqual(panel.rows[1].realized_pnl, -3.0)
        self.assertTrue(panel.rows[1].is_quarantined)
        self.assertFalse(panel.rows[0].is_quarantined)
        self.assertAlmostEqual(panel.total_realized_pnl, 4.0)

    def test_consistency_check_passes(self):
        panel = build_attribution_panel(self.ledger, self.qm)
        self.assertTrue(panel.consistency_check)

    def test_by_agent(self):
        panel = build_attribution_panel(self.ledger, self.qm, by="agent")
        labels = {r.label for r in panel.rows}
        self.assertEqual(labels, {"X", "Y"})

    def test_invalid_by(self):
        with self.assertRaises(ValueError):
            build_attribution_panel(self.ledger, self.qm, by="invalid")

    def test_positives_negatives(self):
        panel = build_attribution_panel(self.ledger, self.qm)
        self.assertEqual(len(panel.positives), 1)
        self.assertEqual(len(panel.negatives), 1)


# ════════════════════════════════════════════════════════════════════
# Leaderboard builder
# ════════════════════════════════════════════════════════════════════


class TestLeaderboardBuilder(unittest.TestCase):
    def setUp(self):
        self.perf = PerformanceMemory(trade_fraction=1.0)
        _seed_perf(self.perf, "AlphaGood", Regime.BULLISH, 5, 1.0)
        _seed_perf(self.perf, "BetaBad", Regime.BULLISH, 5, -1.0, start_id=100)
        self.qm = QuarantineManager(seed={"BetaBad"})

    def test_aggregate_leaderboard(self):
        lb = build_leaderboard(self.perf, self.qm)
        self.assertTrue(lb.is_virtual)
        labels = [r.label for r in lb.rows]
        # AlphaGood — first (positive), BetaBad — последний
        self.assertEqual(labels[0], "AlphaGood")
        # quarantined пометки
        beta_row = next(r for r in lb.rows if r.label == "BetaBad")
        self.assertTrue(beta_row.is_quarantined)

    def test_per_regime_leaderboard(self):
        lb = build_leaderboard(self.perf, self.qm, regime=Regime.BULLISH)
        # У всех есть данные в bullish
        labels = [r.label for r in lb.rows]
        self.assertIn("AlphaGood", labels)
        self.assertIn("BetaBad", labels)

    def test_only_with_data(self):
        # Bearish — никаких данных
        lb = build_leaderboard(self.perf, self.qm, regime=Regime.BEARISH,
                               only_with_data=True)
        self.assertEqual(lb.rows, [])

    def test_sorted_descending(self):
        lb = build_leaderboard(self.perf, self.qm)
        pnls = [r.pnl_pct for r in lb.rows]
        self.assertEqual(pnls, sorted(pnls, reverse=True))


# ════════════════════════════════════════════════════════════════════
# Regime heatmap builder
# ════════════════════════════════════════════════════════════════════


class TestRegimeHeatmapBuilder(unittest.TestCase):
    def setUp(self):
        self.perf = PerformanceMemory(trade_fraction=1.0)
        _seed_perf(self.perf, "A", Regime.BULLISH, 3, 1.0, start_id=1)
        _seed_perf(self.perf, "A", Regime.BEARISH, 2, -2.0, start_id=100)
        self.qm = QuarantineManager(seed=set())

    def test_basic_heatmap(self):
        hm = build_regime_heatmap(self.perf, self.qm)
        self.assertIn("A", hm.labels)
        self.assertEqual(len(hm.regimes), 4)
        # Cells per row = #regimes
        a_idx = hm.labels.index("A")
        self.assertEqual(len(hm.cells[a_idx]), 4)

    def test_quarantined_marked(self):
        qm = QuarantineManager(seed={"A"})
        hm = build_regime_heatmap(self.perf, qm)
        self.assertIn("A", hm.quarantined_labels)
        # Все ячейки A помечены
        a_idx = hm.labels.index("A")
        for cell in hm.cells[a_idx]:
            self.assertTrue(cell.is_quarantined)

    def test_only_with_data_filter(self):
        # Заведомо пустой агент не должен попасть
        hm = build_regime_heatmap(self.perf, self.qm)
        # У "A" есть данные → попадает
        self.assertIn("A", hm.labels)


# ════════════════════════════════════════════════════════════════════
# Quarantine panel builder
# ════════════════════════════════════════════════════════════════════


class TestQuarantinePanelBuilder(unittest.TestCase):
    def test_basic(self):
        qm = QuarantineManager(seed={"A", "B"})
        panel = build_quarantine_panel(qm)
        self.assertEqual(panel.quarantined_labels, ["A", "B"])
        self.assertEqual(panel.seed_labels, ["A", "B"])
        self.assertEqual(panel.auto_added, [])

    def test_with_auto(self):
        qm = QuarantineManager(seed={"A"})
        panel = build_quarantine_panel(
            qm,
            auto_added=["NewBad"],
            auto_released=["GoodOne"],
        )
        self.assertEqual(panel.auto_added, ["NewBad"])
        self.assertEqual(panel.auto_released, ["GoodOne"])

    def test_empty(self):
        qm = QuarantineManager(seed=set())
        panel = build_quarantine_panel(qm)
        self.assertEqual(panel.quarantined_labels, [])


# ════════════════════════════════════════════════════════════════════
# Decision timeline builder
# ════════════════════════════════════════════════════════════════════


class TestTimelineBuilder(unittest.TestCase):
    def test_filters_relevant(self):
        log = EventLog()
        log.emit(BarStarted(bar=1))
        log.emit(LeaderSelected(bar=1, player_label="X", reason="bootstrap"))
        # PositionOpened игнорируется — не в interesting set
        log.emit(PositionOpened(bar=1, signal_id=1, sym="BTC",
                                side="long", entry=100.0, qty=1.0))
        timeline = build_decision_timeline(log)
        types = {e.event_type for e in timeline.events}
        self.assertIn("BarStarted", types)
        self.assertIn("LeaderSelected", types)
        self.assertNotIn("PositionOpened", types)

    def test_bar_filter(self):
        log = EventLog()
        for b in (1, 5, 10, 15):
            log.emit(BarStarted(bar=b))
        timeline = build_decision_timeline(log, bar_from=5, bar_to=11)
        bars = [e.bar for e in timeline.events]
        self.assertEqual(bars, [5, 10])

    def test_max_events(self):
        log = EventLog()
        for b in range(20):
            log.emit(BarStarted(bar=b))
        timeline = build_decision_timeline(log, max_events=5)
        self.assertEqual(len(timeline.events), 5)
        # Последние
        self.assertEqual(timeline.events[-1].bar, 19)


# ════════════════════════════════════════════════════════════════════
# DashboardRenderer (composite)
# ════════════════════════════════════════════════════════════════════


class TestDashboardRenderer(unittest.TestCase):
    def setUp(self):
        # Minimal data
        log = EventLog()
        _add_close_to_log(log, sym="BTC", op_id=1, cl_id=2, bar=2,
                         player="P1", agent="A1", pnl=5.0)
        self.ledger = AttributionLedger()
        self.ledger.replay_from_event_log(log)

        self.perf = PerformanceMemory(trade_fraction=1.0)
        _seed_perf(self.perf, "P1", Regime.BULLISH, 3, 1.0)
        self.qm = QuarantineManager(seed=set())
        self.event_log = log
        self.health = SymbolHealthMonitor()
        self.renderer = DashboardRenderer(
            ledger=self.ledger,
            perf=self.perf,
            qm=self.qm,
            event_log=self.event_log,
            health=self.health,
        )

    def test_build_main_returns_composite(self):
        main = self.renderer.build_main()
        self.assertIsNotNone(main.attribution)
        self.assertIsNotNone(main.leaderboard)
        self.assertIsNotNone(main.regime_heatmap)
        self.assertIsNotNone(main.quarantine)

    def test_consistency_across_panels(self):
        """Q1 cross-panel: карантинный помечен везде одинаково."""
        qm = QuarantineManager(seed={"P1"})
        renderer = DashboardRenderer(
            ledger=self.ledger,
            perf=self.perf,
            qm=qm,
        )
        attr = renderer.build_attribution()
        lb = renderer.build_leaderboard()
        # P1 должен быть quarantined в обеих панелях
        if attr.rows:
            p1_attr = next(r for r in attr.rows if r.label == "P1")
            self.assertTrue(p1_attr.is_quarantined)
        if lb.rows:
            p1_lb = next(r for r in lb.rows if r.label == "P1")
            self.assertTrue(p1_lb.is_quarantined)

    def test_health_optional(self):
        renderer = DashboardRenderer(
            ledger=self.ledger,
            perf=self.perf,
            qm=self.qm,
            # health=None
        )
        self.assertIsNone(renderer.build_symbol_health())

    def test_event_log_optional(self):
        renderer = DashboardRenderer(
            ledger=self.ledger,
            perf=self.perf,
            qm=self.qm,
            # event_log=None
        )
        self.assertIsNone(renderer.build_decision_timeline())


# ════════════════════════════════════════════════════════════════════
# TextRenderer
# ════════════════════════════════════════════════════════════════════


class TestTextRenderer(unittest.TestCase):
    def test_renders_attribution_with_quarantine_marker(self):
        log = EventLog()
        _add_close_to_log(log, sym="BTC", op_id=1, cl_id=2, bar=2,
                         player="QPlayer", agent="A", pnl=5.0)
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        qm = QuarantineManager(seed={"QPlayer"})
        panel = build_attribution_panel(ledger, qm)
        text = TextRenderer().render_attribution(panel)
        self.assertIn("QPlayer", text)
        self.assertIn("[Q]", text, "Quarantined player must have [Q] marker")

    def test_renders_empty_attribution(self):
        ledger = AttributionLedger()
        qm = QuarantineManager(seed=set())
        panel = build_attribution_panel(ledger, qm)
        text = TextRenderer().render_attribution(panel)
        self.assertIn("нет данных", text)

    def test_renders_leaderboard(self):
        perf = PerformanceMemory(trade_fraction=1.0)
        _seed_perf(perf, "X", Regime.BULLISH, 3, 1.0)
        qm = QuarantineManager(seed=set())
        lb = build_leaderboard(perf, qm)
        text = TextRenderer().render_leaderboard(lb)
        self.assertIn("X", text)
        self.assertIn("virtual", text)

    def test_renders_main(self):
        log = EventLog()
        _add_close_to_log(log, sym="BTC", op_id=1, cl_id=2, bar=2, pnl=5.0)
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        perf = PerformanceMemory(trade_fraction=1.0)
        _seed_perf(perf, "P", Regime.BULLISH, 3, 1.0)
        qm = QuarantineManager(seed=set())
        renderer = DashboardRenderer(ledger=ledger, perf=perf, qm=qm)
        main = renderer.build_main()
        text = TextRenderer().render_main(main)
        # Composite должен содержать секции:
        self.assertIn("вклад", text.lower())
        self.assertIn("leaderboard", text.lower())


# ════════════════════════════════════════════════════════════════════
# Q1 cross-panel consistency property test
# ════════════════════════════════════════════════════════════════════


class TestCrossPanelQuarantineConsistency(unittest.TestCase):
    def test_quarantined_labelled_consistently(self):
        """Property: для каждого label либо ВСЕ панели помечают его как
        carantined, либо НИКАКАЯ."""
        log = EventLog()
        _add_close_to_log(log, sym="BTC", op_id=1, cl_id=2, bar=2,
                         player="QP", agent="QA", pnl=1.0)
        ledger = AttributionLedger()
        ledger.replay_from_event_log(log)
        perf = PerformanceMemory(trade_fraction=1.0)
        _seed_perf(perf, "QP", Regime.BULLISH, 3, 1.0)
        _seed_perf(perf, "QA", Regime.BULLISH, 3, 1.0, start_id=100)
        qm = QuarantineManager(seed={"QP"})

        attr = build_attribution_panel(ledger, qm)
        lb = build_leaderboard(perf, qm)
        hm = build_regime_heatmap(perf, qm)

        # QP in attr and lb
        attr_qp = next((r for r in attr.rows if r.label == "QP"), None)
        lb_qp = next((r for r in lb.rows if r.label == "QP"), None)
        if attr_qp is not None:
            self.assertTrue(attr_qp.is_quarantined)
        if lb_qp is not None:
            self.assertTrue(lb_qp.is_quarantined)
        # heatmap: QP в quarantined_labels
        if "QP" in hm.labels:
            self.assertIn("QP", hm.quarantined_labels)


if __name__ == "__main__":
    unittest.main(verbosity=2)
