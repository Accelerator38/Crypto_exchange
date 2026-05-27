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
from .flash_selected_subset_manifest import (
    build_flash_selected_subset_manifest,
    write_flash_selected_subset_manifest,
)
from .flash_profitability_gates import (
    DEFAULT_PROFITABILITY_GATES,
    ProfitabilityGates,
    build_profitability_gate_report,
    evaluate_profitability_gates,
    load_flash_run_metrics,
    write_profitability_gate_report,
)
from .flash_partial_profit_lock_sweep import (
    build_partial_profit_lock_ab_variants,
    build_partial_profit_lock_command,
    write_partial_profit_lock_ab_plan,
)

__all__ = [
    "LeaderPnL",
    "AllocationDiagnostics",
    "LeaderActionability",
    "PerfectMonthSelection",
    "PerfectPanteonReport",
    "ProfitabilityGates",
    "RegretReport",
    "RetroWhatIfConfig",
    "ShadowPnLEvent",
    "SoftAllocatorPolicy",
    "SoftAllocatorReport",
    "DEFAULT_PROFITABILITY_GATES",
    "allocation_diagnostics_to_dict",
    "analyze_trading_log",
    "build_genetics_degradation_report",
    "build_partial_profit_lock_ab_variants",
    "build_partial_profit_lock_command",
    "build_recomposed_candidates",
    "build_profitability_gate_report",
    "build_walk_forward_report",
    "build_flash_selected_subset_manifest",
    "compute_leader_regret",
    "evaluate_profitability_gates",
    "load_flash_run_metrics",
    "run_retrodate_whatif",
    "run_retrodate_whatif_for_exchange",
    "score_candidate_with_session_overlay",
    "select_whatif_candidate",
    "simulate_perfect_monthly_panteon",
    "simulate_soft_allocator_policies",
    "write_allocation_diagnostics",
    "write_flash_selected_subset_manifest",
    "write_partial_profit_lock_ab_plan",
    "write_profitability_gate_report",
    "write_walk_forward_report",
]
