"""Permanent fail-closed guard for the legacy Pantheon Bitget live path."""

from __future__ import annotations


LEGACY_PANTEON_BITGET_LIVE_FROZEN = True
LEGACY_PANTEON_FREEZE_REASON = (
    "legacy Pantheon BITGET external-order path is frozen; "
    "use paper/research modes or the isolated freqtrade_reset path"
)


def legacy_panteon_live_freeze_reason(exchange: str, mode: str) -> str:
    """Return a blocker for legacy Bitget modes that can reach real orders."""
    exchange_name = str(exchange or "").strip().upper()
    mode_name = str(mode or "").strip().lower()
    if exchange_name != "BITGET":
        return ""
    if mode_name in {
        "paper",
        "paper_live_feed",
        "shadow_live_feed",
        "demo_futures",
    }:
        return ""
    if LEGACY_PANTEON_BITGET_LIVE_FROZEN:
        return LEGACY_PANTEON_FREEZE_REASON
    return ""
