"""Thin entrypoint for running Panteon trading on Bitget."""

from __future__ import annotations

import os
import runpy


required = ("BITGET_API_KEY", "BITGET_SECRET_KEY", "BITGET_PASSPHRASE")
missing = [name for name in required if not os.getenv(name)]
if missing:
    raise RuntimeError(
        "Set the following environment variables before launch: "
        + ", ".join(missing)
    )

os.environ.setdefault("CRYPTO_EXCHANGE", "BITGET")

runpy.run_module("Panteon_Trade", run_name="__main__")
