"""Shadow run (Phase 8): v2 рядом с v1, dry-run pipeline на реальных данных.

Архитектура:
  adapters    — V1AgentAdapter, make_market_snapshot, regime_to_v1_string
  feed        — MarketFeed Protocol + ReplayFeed/PollingFeed/CallableFeed
  runner      — ShadowRunner.step() / .run_until_exhausted()
  comparator  — compare_v1_vs_v2() + render_comparison()
  cli         — python -m panteon_v2.shadow.cli

Использование (replay-стиль на v1-логе):
    cd src
    python -m panteon_v2.shadow.cli --session /path/to/v1/session
"""

from .adapters import (
    GeneticsRegimeRouterV2AgentAdapter,
    GeneticsV2AgentAdapter,
    V1AgentAdapter,
    map_genetics_legacy_action,
    make_market_snapshot,
    regime_to_v1_string,
)
from .comparator import (
    ComparisonReport,
    LeaderDiff,
    compare_v1_vs_v2,
    render_comparison,
)
from .feed import (
    CallableFeed,
    MarketFeed,
    PollingFeed,
    ReplayFeed,
)
from .runner import (
    ShadowRunner,
    StepResult,
)

__all__ = [
    # adapters
    "GeneticsRegimeRouterV2AgentAdapter",
    "GeneticsV2AgentAdapter",
    "V1AgentAdapter",
    "map_genetics_legacy_action",
    "make_market_snapshot",
    "regime_to_v1_string",
    # feed
    "CallableFeed",
    "MarketFeed",
    "PollingFeed",
    "ReplayFeed",
    # runner
    "ShadowRunner",
    "StepResult",
    # comparator
    "ComparisonReport",
    "LeaderDiff",
    "compare_v1_vs_v2",
    "render_comparison",
]
