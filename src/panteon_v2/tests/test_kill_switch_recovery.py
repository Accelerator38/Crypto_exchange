"""Phase 0 (A7): авто-recovery kill-switch.

Проверяем:
  • при выключенном авто-recovery (дефолт) защёлка НИКОГДА не снимается;
  • peak_drawdown-защёлка снимается после cooldown И восстановления equity;
  • cooldown обязателен;
  • недостаточное восстановление equity блокирует recovery;
  • операционная защёлка (failed_orders) снимается по cooldown и сбрасывает счётчики.
"""

from __future__ import annotations

import types
import unittest

from panteon_v2.app.bootstrap import KillSwitchState, LiveExecutionConfig
from panteon_v2.app.main_loop import _kill_switch_reason, _record_order_failure


def _market(bar: int):
    return types.SimpleNamespace(bar=bar)


def _pipeline(cfg: LiveExecutionConfig, *, balance: float = 1000.0, peak: float = 1000.0):
    ks = KillSwitchState(peak_equity_usd=peak)
    return types.SimpleNamespace(
        live_execution=cfg,
        kill_switch=ks,
        initial_capital=1000.0,
        current_balance=balance,
    )


class TestKillSwitchRecovery(unittest.TestCase):
    def _trip_drawdown(self, pipe) -> None:
        # При balance=700, peak=1000, порог 20% → floor=800 → trip.
        pipe.current_balance = 700.0
        reason = _kill_switch_reason(pipe, _market(10))
        self.assertIn("drawdown", reason)
        self.assertEqual(pipe.kill_switch.disabled_kind, "peak_drawdown")
        self.assertEqual(pipe.kill_switch.disabled_bar, 10)

    def test_recovery_disabled_by_default_never_clears(self):
        cfg = LiveExecutionConfig(max_equity_peak_drawdown_pct=20.0)
        pipe = _pipeline(cfg, peak=1000.0)
        self._trip_drawdown(pipe)
        # Восстановили equity полностью и прошло много баров — но recovery выключен.
        pipe.current_balance = 1000.0
        self.assertNotEqual(_kill_switch_reason(pipe, _market(1000)), "")

    def test_peak_drawdown_recovers_after_cooldown_and_equity(self):
        cfg = LiveExecutionConfig(
            max_equity_peak_drawdown_pct=20.0,
            kill_switch_auto_recovery_enabled=True,
            kill_switch_recovery_cooldown_bars=5,
            kill_switch_recovery_max_drawdown_pct=2.0,  # допускаем остаточную просадку ≤2%
        )
        pipe = _pipeline(cfg, peak=1000.0)
        self._trip_drawdown(pipe)
        # equity восстановилась до 990 (просадка 1% ≤ 2%), прошло 6 баров ≥ cooldown=5
        pipe.current_balance = 990.0
        self.assertEqual(_kill_switch_reason(pipe, _market(16)), "")
        self.assertEqual(pipe.kill_switch.disabled_reason, "")
        self.assertEqual(pipe.kill_switch.disabled_kind, "")

    def test_no_recovery_before_cooldown(self):
        cfg = LiveExecutionConfig(
            max_equity_peak_drawdown_pct=20.0,
            kill_switch_auto_recovery_enabled=True,
            kill_switch_recovery_cooldown_bars=50,
            kill_switch_recovery_max_drawdown_pct=2.0,
        )
        pipe = _pipeline(cfg, peak=1000.0)
        self._trip_drawdown(pipe)
        pipe.current_balance = 1000.0
        # Прошло лишь 5 баров (10→15) < cooldown 50
        self.assertNotEqual(_kill_switch_reason(pipe, _market(15)), "")

    def test_no_recovery_when_equity_not_recovered(self):
        cfg = LiveExecutionConfig(
            max_equity_peak_drawdown_pct=20.0,
            kill_switch_auto_recovery_enabled=True,
            kill_switch_recovery_cooldown_bars=5,
            kill_switch_recovery_max_drawdown_pct=2.0,
        )
        pipe = _pipeline(cfg, peak=1000.0)
        self._trip_drawdown(pipe)
        # Cooldown прошёл, но equity всё ещё 850 (просадка 15% > 2%)
        pipe.current_balance = 850.0
        self.assertNotEqual(_kill_switch_reason(pipe, _market(100)), "")

    def test_operational_latch_recovers_by_cooldown_and_resets_counters(self):
        cfg = LiveExecutionConfig(
            max_consecutive_failed_orders=2,
            kill_switch_auto_recovery_enabled=True,
            kill_switch_recovery_cooldown_bars=3,
        )
        pipe = _pipeline(cfg)
        # Установим последний бар, затем дважды провалим ордер → trip.
        _kill_switch_reason(pipe, _market(0))
        _record_order_failure(pipe, "exchange error: timeout")
        _record_order_failure(pipe, "exchange error: timeout")
        self.assertIn("consecutive failed orders", _kill_switch_reason(pipe, _market(1)))
        self.assertEqual(pipe.kill_switch.disabled_kind, "failed_orders")
        # Прошло ≥3 баров → recovery, счётчики обнулены.
        self.assertEqual(_kill_switch_reason(pipe, _market(10)), "")
        self.assertEqual(pipe.kill_switch.consecutive_failed_orders, 0)


if __name__ == "__main__":
    unittest.main()
