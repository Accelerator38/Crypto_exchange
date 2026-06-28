"""Compatibility wrapper for the unified ML launcher.

Legacy direct Bitget live loops are intentionally closed. Keep this filename for
old shortcuts, but route execution through Start_ML.py so startup health,
reconcile, dashboards and order lifecycle are identical to the main launcher.
"""

from __future__ import annotations

import os

import Start_ML


def main() -> int:
    os.environ["ML_EXCHANGES"] = "BITGET"
    return Start_ML.main()


if __name__ == "__main__":
    raise SystemExit(main())
