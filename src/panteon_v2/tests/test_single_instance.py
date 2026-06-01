"""Tests for live entrypoint single-instance locking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class TestSingleInstanceLock(unittest.TestCase):
    def test_second_acquire_returns_none_until_release(self):
        from panteon_v2.app.single_instance import acquire_single_instance

        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            first = acquire_single_instance("panteon_v2_mexc", lock_dir=td)
            self.assertIsNotNone(first)

            second = acquire_single_instance("panteon_v2_mexc", lock_dir=td)
            self.assertIsNone(second)

            first.release()

            third = acquire_single_instance("panteon_v2_mexc", lock_dir=td)
            self.assertIsNotNone(third)
            third.release()

    def test_lock_file_contains_pid(self):
        from panteon_v2.app.single_instance import acquire_single_instance

        with tempfile.TemporaryDirectory(dir=ROOT) as td:
            lock = acquire_single_instance("panteon_v2_bitget", lock_dir=td)
            self.assertIsNotNone(lock)
            try:
                text = Path(lock.info_path).read_text(encoding="utf-8")
                self.assertIn("pid=", text)
                self.assertIn("parent_pid=", text)
                self.assertIn("name=panteon_v2_bitget", text)
                self.assertIn("sys_executable=", text)
                self.assertIn("sys_prefix=", text)
                self.assertIn("sys_base_prefix=", text)
                self.assertIn("argv=", text)
                self.assertLess(len(text.encode("utf-8")), 4096)
            finally:
                info_path = lock.info_path
                lock.release()
                self.assertFalse(Path(info_path).exists())


if __name__ == "__main__":
    unittest.main()
