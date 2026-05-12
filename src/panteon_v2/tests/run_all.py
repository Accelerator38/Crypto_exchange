"""Smoke runner — запускает все unittest-suite-ы пакета без pytest.

Использование:
    cd src && python -m panteon_v2.tests.run_all
"""

from __future__ import annotations

import sys
import unittest


def main() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir="panteon_v2/tests",
        pattern="test_*.py",
        top_level_dir=".",
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
