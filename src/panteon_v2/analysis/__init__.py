"""Offline analysis helpers for Panteon v2."""

from .retro_whatif import (
    RetroWhatIfConfig,
    build_recomposed_candidates,
    run_retrodate_whatif,
    run_retrodate_whatif_for_exchange,
    score_candidate_with_session_overlay,
    select_whatif_candidate,
)
from .genetics_degradation_monitoring import build_genetics_degradation_report
from .oracle_regret import LeaderPnL, RegretReport, compute_leader_regret
from .soft_allocator import (
    ShadowPnLEvent,
    SoftAllocatorPolicy,
    SoftAllocatorReport,
    simulate_soft_allocator_policies,
)
from .walk_forward import build_walk_forward_report, write_walk_forward_report

__all__ = [
    "LeaderPnL",
    "RegretReport",
    "RetroWhatIfConfig",
    "ShadowPnLEvent",
    "SoftAllocatorPolicy",
    "SoftAllocatorReport",
    "build_genetics_degradation_report",
    "build_recomposed_candidates",
    "build_walk_forward_report",
    "compute_leader_regret",
    "run_retrodate_whatif",
    "run_retrodate_whatif_for_exchange",
    "score_candidate_with_session_overlay",
    "select_whatif_candidate",
    "simulate_soft_allocator_policies",
    "write_walk_forward_report",
]
