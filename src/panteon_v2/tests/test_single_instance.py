"""Tests for live entrypoint single-instance locking."""

from __future__ import annotations

import tempfile
import unittest


class TestSingleInstanceLock(unittest.TestCase):
    def test_second_acquire_returns_none_until_release(self):
        from panteon_v2.app.single_instance import acquire_single_instance

        with tempfile.TemporaryDirectory() as td:
            first = acquire_single_instance("panteon_v2_mexc", lock_dir=td)
            self.assertIsNotNone(first)

            second = acquire_single_instance("panteon_v2_mexc", lock_dir=td)
            self.assertIsNone(second)

            first.release()

            third = acquire_single_instance("panteon_v2_mexc", lock_dir=td)
            self.assertIsNotNone(third)
            third.release()

    def test_lock_file_contains_pid(self):
        from pathlib import Path

        from panteon_v2.app.single_instance import acquire_single_instance

        with tempfile.TemporaryDirectory() as td:
            lock = acquire_single_instance("panteon_v2_bitget", lock_dir=td)
            self.assertIsNotNone(lock)
            try:
                text = Path(lock.path).read_text(encoding="utf-8")
                self.assertIn("pid=", text)
                self.assertIn("name=panteon_v2_bitget", text)
            finally:
                lock.release()


if __name__ == "__main__":
    unittest.main()
