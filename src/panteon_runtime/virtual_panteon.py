"""Virtual Panteon mirror helpers.

The mirror is intentionally small: it does not run its own strategy. It replays
the already planned actions emitted by the real Panteon capture wrapper on the
same bar into a virtual portfolio.
"""

from __future__ import annotations

from typing import Mapping


VIRTUAL_PANTEON_NAME = "V_Virtual_Panteon"
VIRTUAL_PANTEON_DISPLAY_NAME = "Virtual_Panteon"


def is_virtual_panteon_name(name: object) -> bool:
    text = str(name or "").strip()
    if text.startswith("V_"):
        text = text[2:]
    return text == VIRTUAL_PANTEON_DISPLAY_NAME


class VirtualPanteonMirror:
    """Replay real Panteon planned actions into a shadow portfolio."""

    PLAYER_STATUS = "shadow_only"
    PLAYER_STATUS_REASON = "virtual mirror of real Panteon planned actions"

    def __init__(self, source) -> None:
        self._source = source
        self._last_risk_multipliers = {}
        self._last_regime = "neutral"

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        try:
            current_bar = int(bar_index or 0)
            source_bar = int(getattr(self._source, "_last_actions_bar", -1) or -1)
        except Exception:
            return {}
        if current_bar != source_bar:
            return {}

        raw_actions = getattr(self._source, "_last_actions", {}) or {}
        if not isinstance(raw_actions, Mapping):
            return {}

        raw_risk = getattr(self._source, "_last_risk_multipliers", {}) or {}
        self._last_risk_multipliers = dict(raw_risk) if isinstance(raw_risk, Mapping) else {}
        self._last_regime = str(getattr(self._source, "_last_regime", "") or "neutral")
        return dict(raw_actions)
