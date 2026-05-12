"""Тесты SymbolHealthMonitor."""

from __future__ import annotations

import unittest

from panteon_v2.execution.symbol_health import (
    SymbolHealthConfig,
    SymbolHealthMonitor,
)


class _FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class TestConfig(unittest.TestCase):
    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            SymbolHealthConfig(failure_threshold=0)

    def test_invalid_window(self):
        with self.assertRaises(ValueError):
            SymbolHealthConfig(failure_window_sec=0)

    def test_invalid_block_duration(self):
        with self.assertRaises(ValueError):
            SymbolHealthConfig(block_duration_sec=-1)


class TestSymbolHealthMonitor(unittest.TestCase):
    def setUp(self):
        self.clock = _FakeClock()
        self.health = SymbolHealthMonitor(now_fn=self.clock)

    def test_initial_state(self):
        self.assertFalse(self.health.is_blocked("BTC"))
        self.assertEqual(self.health.all_blocked(), {})

    def test_first_failure_no_block(self):
        self.assertFalse(self.health.record_pending_failure("BTC"))
        self.assertFalse(self.health.is_blocked("BTC"))

    def test_third_failure_triggers_block(self):
        for _ in range(2):
            self.assertFalse(self.health.record_pending_failure("BTC"))
        self.assertTrue(self.health.record_pending_failure("BTC"))
        self.assertTrue(self.health.is_blocked("BTC"))

    def test_fourth_failure_no_new_block(self):
        for _ in range(3):
            self.health.record_pending_failure("BTC")
        # Уже заблокирован — 4й не возвращает True (только первый transition)
        self.assertFalse(self.health.record_pending_failure("BTC"))

    def test_success_clears(self):
        for _ in range(3):
            self.health.record_pending_failure("BTC")
        self.assertTrue(self.health.is_blocked("BTC"))
        self.health.record_success("BTC")
        self.assertFalse(self.health.is_blocked("BTC"))

    def test_block_expires_after_duration(self):
        config = SymbolHealthConfig(
            failure_threshold=2, failure_window_sec=100, block_duration_sec=10,
        )
        h = SymbolHealthMonitor(config=config, now_fn=self.clock)
        h.record_pending_failure("BTC")
        h.record_pending_failure("BTC")
        self.assertTrue(h.is_blocked("BTC"))
        # advance time past block_duration
        self.clock.t = 11.0
        self.assertFalse(h.is_blocked("BTC"))

    def test_failures_outside_window_dont_count(self):
        config = SymbolHealthConfig(
            failure_threshold=3, failure_window_sec=10, block_duration_sec=100,
        )
        h = SymbolHealthMonitor(config=config, now_fn=self.clock)
        # Первый failure в t=0
        h.record_pending_failure("BTC")
        # Перешагиваем окно
        self.clock.t = 20.0
        h.record_pending_failure("BTC")
        h.record_pending_failure("BTC")
        # Только 2 в текущем окне → не блокируется
        self.assertFalse(h.is_blocked("BTC"))

    def test_status_view(self):
        self.health.record_pending_failure("BTC")
        st = self.health.status("BTC")
        self.assertEqual(st.failures_in_window, 1)
        self.assertFalse(st.is_blocked)

    def test_per_symbol_isolation(self):
        for _ in range(3):
            self.health.record_pending_failure("BTC")
        self.assertTrue(self.health.is_blocked("BTC"))
        self.assertFalse(self.health.is_blocked("ETH"))

    def test_normalize_symbol(self):
        self.health.record_pending_failure("btc")
        st = self.health.status("BTC")
        self.assertEqual(st.failures_in_window, 1)
        st2 = self.health.status("  btc  ")
        self.assertEqual(st2.failures_in_window, 1)

    def test_all_blocked_clears_expired(self):
        config = SymbolHealthConfig(
            failure_threshold=2, failure_window_sec=100, block_duration_sec=10,
        )
        h = SymbolHealthMonitor(config=config, now_fn=self.clock)
        for _ in range(2):
            h.record_pending_failure("BTC")
        self.assertIn("BTC", h.all_blocked())
        self.clock.t = 100.0
        self.assertEqual(h.all_blocked(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
