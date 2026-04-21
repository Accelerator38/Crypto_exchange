"""Thin entrypoint for running Panteon trading on MEXC."""

from __future__ import annotations

import os
import runpy


os.environ.setdefault("CRYPTO_EXCHANGE", "MEXC")

runpy.run_module("Panteon_Trade", run_name="__main__")
