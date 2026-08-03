from __future__ import annotations

from unittest.mock import Mock

from panteon_v2.app import startup
from panteon_v2.app.live_freeze import legacy_panteon_live_freeze_reason


def test_bitget_live_is_frozen_without_environment_bypass(monkeypatch):
    monkeypatch.setenv("PANTEON_BYPASS_LIVE_FREEZE", "1")

    assert legacy_panteon_live_freeze_reason("BITGET", "live_futures")
    assert legacy_panteon_live_freeze_reason("BITGET", "production")


def test_bitget_virtual_and_demo_modes_remain_available():
    for mode in ("paper", "paper_live_feed", "shadow_live_feed", "demo_futures"):
        assert legacy_panteon_live_freeze_reason("BITGET", mode) == ""


def test_direct_startup_stops_before_exchange_resolution(monkeypatch):
    resolve_exchange = Mock(side_effect=AssertionError("must not be called"))
    monkeypatch.setattr(startup, "resolve_exchange", resolve_exchange)

    result = startup.start_production(exchange="BITGET", mode="live_futures")

    assert result == 2
    resolve_exchange.assert_not_called()
