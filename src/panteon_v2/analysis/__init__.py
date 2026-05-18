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
from .allocation_diagnostics import (
    AllocationDiagnostics,
    LeaderActionability,
    allocation_diagnostics_to_dict,
    analyze_trading_log,
    write_allocation_diagnostics,
)
from .oracle_regret import LeaderPnL, RegretReport, compute_leader_regret
from .soft_allocator import (
    PerfectMonthSelection,
    PerfectPanteonReport,
    ShadowPnLEvent,
    SoftAllocatorPolicy,
    SoftAllocatorReport,
    simulate_perfect_monthly_panteon,
    simulate_soft_allocator_policies,
)
from .walk_forward import build_walk_forward_report, write_walk_forward_report

__all__ = [
    "LeaderPnL",
    "AllocationDiagnostics",
    "LeaderActionability",
    "PerfectMonthSelection",
    "PerfectPanteonReport",
    "RegretReport",
    "RetroWhatIfConfig",
    "ShadowPnLEvent",
    "SoftAllocatorPolicy",
    "SoftAllocatorReport",
    "allocation_diagnostics_to_dict",
    "analyze_trading_log",
    "build_genetics_degradation_report",
    "build_recomposed_candidates",
    "build_walk_forward_report",
    "compute_leader_regret",
    "run_retrodate_whatif",
    "run_retrodate_whatif_for_exchange",
    "score_candidate_with_session_overlay",
    "select_whatif_candidate",
    "simulate_perfect_monthly_panteon",
    "simulate_soft_allocator_policies",
    "write_allocation_diagnostics",
    "write_walk_forward_report",
]
