"""Panteon Legend profile.

Legend keeps the legacy direct-trading bias while reusing v2 observability,
shadow accounting, current agent adapters, and dashboards.
"""

from .config import LegendProfile, build_legend_profile, legend_actor_labels

__all__ = [
    "LegendProfile",
    "build_legend_profile",
    "legend_actor_labels",
]
