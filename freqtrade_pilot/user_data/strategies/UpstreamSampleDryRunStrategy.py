from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
UPSTREAM_TEMPLATE_CANDIDATES = (
    PROJECT_ROOT / "vendor" / "freqtrade" / "freqtrade" / "templates",
    Path("/freqtrade/upstream_templates"),
)
UPSTREAM_TEMPLATES = next(
    (path for path in UPSTREAM_TEMPLATE_CANDIDATES if path.is_dir()),
    UPSTREAM_TEMPLATE_CANDIDATES[0],
)
if str(UPSTREAM_TEMPLATES) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_TEMPLATES))

from sample_strategy import SampleStrategy  # noqa: E402


class UpstreamSampleDryRunStrategy(SampleStrategy):
    """Unmodified upstream signals with a fail-closed Bitget dry-run adapter."""

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
        "stoploss_price_type": "mark",
    }

    def bot_start(self, **kwargs: object) -> None:
        if self.config.get("dry_run") is not True:
            raise RuntimeError("upstream pilot is permanently restricted to dry_run=true")
