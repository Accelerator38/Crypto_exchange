from __future__ import annotations

from pathlib import Path

from panteon_v2.analysis.flash_partial_profit_lock_sweep import (
    build_partial_profit_lock_ab_variants,
    build_partial_profit_lock_command,
)


def test_partial_profit_lock_ab_variants_are_fixed_matrix_and_skip_protected():
    variants = build_partial_profit_lock_ab_variants()

    assert len(variants) == 9
    assert {item["trigger_pnl_pct"] for item in variants} == {2.0, 3.0, 4.0}
    assert {item["close_fraction"] for item in variants} == {0.25, 0.33, 0.5}
    assert all(item["skip_protected_signal_keys"] is True for item in variants)
    assert all(item["production_default_enabled"] is False for item in variants)


def test_partial_profit_lock_command_does_not_disable_protected_skip():
    variant = build_partial_profit_lock_ab_variants()[0]
    command = build_partial_profit_lock_command(
        python="python",
        candidate_runner=Path("tools/run_flash_selected_subset_candidate.py"),
        manifest=Path("manifest.json"),
        results_root=Path("Results/PartialLock"),
        years="2026",
        max_bars=100,
        variant=variant,
    )

    assert "--enable-partial-profit-lock" in command
    assert "--disable-partial-profit-lock-skip-protected-signal-keys" not in command
    assert "--partial-profit-lock-trigger-pnl-pct" in command
    assert "--partial-profit-lock-close-fraction" in command
    assert "--risk-capital-fraction" in command
    assert command[command.index("--risk-capital-fraction") + 1] == "0.1"
    assert command[command.index("--risk-max-open-positions") + 1] == "8"
    assert command[command.index("--max-new-opens-per-bar") + 1] == "1"
