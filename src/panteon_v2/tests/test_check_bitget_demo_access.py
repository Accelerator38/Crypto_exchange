from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    path = ROOT / "tools" / "check_bitget_demo_access.py"
    spec = importlib.util.spec_from_file_location("check_bitget_demo_access", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_check_fails_closed_without_demo_credentials(monkeypatch):
    module = _load_tool()
    monkeypatch.setattr(module, "_load_env", lambda: None)
    for name in (
        "BITGET_DEMO_API_KEY",
        "BITGET_DEMO_SECRET_KEY",
        "BITGET_DEMO_PASSPHRASE",
    ):
        monkeypatch.delenv(name, raising=False)

    payload = module.run_check()

    assert payload["passed"] is False
    assert payload["reason"] == "demo_credentials_missing"
    assert payload["orders_sent"] == 0


def test_read_check_uses_demo_header_and_never_sends_orders(monkeypatch):
    module = _load_tool()
    monkeypatch.setattr(module, "_load_env", lambda: None)
    monkeypatch.setenv("BITGET_DEMO_API_KEY", "key")
    monkeypatch.setenv("BITGET_DEMO_SECRET_KEY", "secret")
    monkeypatch.setenv("BITGET_DEMO_PASSPHRASE", "passphrase")

    class FakeClient:
        def __init__(self, api_key, api_secret, api_passphrase, *, demo):
            assert (api_key, api_secret, api_passphrase) == (
                "key",
                "secret",
                "passphrase",
            )
            assert demo is True
            self.swap = SimpleNamespace(headers={"paptrading": "1"})

        def get_futures_account(self):
            return {"equity": 1000.0, "available": 900.0}

        def get_futures_positions(self):
            return []

    monkeypatch.setitem(
        sys.modules,
        "bitget_api",
        SimpleNamespace(BitgetDirectClient=FakeClient),
    )

    payload = module.run_check()

    assert payload == {
        "passed": True,
        "reason": "demo_read_access_confirmed",
        "demo_header": True,
        "equity": 1000.0,
        "available": 900.0,
        "open_positions": 0,
        "orders_sent": 0,
    }
