# Panteon Flash PnL LCB Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add standalone-vs-Flash selected diagnostics, then use per-signal-key PnL lower confidence bounds to reduce bad Flash selections and validate on 2025.

**Architecture:** Keep the first iteration inside the existing retrodate/reporting and Flash promotion boundaries. Add deterministic report helpers, extend promotion manifest rows with PnL-per-trade LCB, then expose knobs through CLI/matrix without changing the agent vote contract.

**Tech Stack:** Python dataclasses, pytest, existing Retrodate runner artifacts, Panteon Flash allocator and promotion manifest.

---

### Task 1: Standalone vs Flash Selected Report

**Files:**
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

- [ ] Write failing tests for a helper that compares standalone shadow PnL events with Flash selected attribution for `Solo_MomentumScalper` and `LiveOIBreakout`.
- [ ] Implement JSON and Markdown report writers: `standalone_vs_flash_selected_report.json` and `.md`.
- [ ] Include actor label, standalone PnL, Flash-selected PnL, alpha, selected/closed counts, and top signal keys by positive/negative selected PnL when available.
- [ ] Run targeted tests.

### Task 2: PnL-Per-Trade LCB in Promotion Manifest

**Files:**
- Modify: `src/panteon_v2/selection/promotion_manifest.py`
- Test: `src/panteon_v2/tests/test_promotion_manifest.py`

- [ ] Write failing tests for `mean - z * std / sqrt(n)` lower bound on per-trade PnL.
- [ ] Add config fields: `min_full_pnl_per_trade_lcb`, `min_latest_pnl_per_trade_lcb`, and `pnl_lcb_z`.
- [ ] Persist the LCB values and rejection reasons in manifest output.
- [ ] Run targeted manifest tests.

### Task 3: CLI and Matrix Exposure

**Files:**
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Modify: `tools/run_panteon_flash_profitability_matrix.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_panteon_flash_profitability_matrix.py`

- [ ] Write failing tests for new CLI flags and matrix experiment args.
- [ ] Add CLI flags for PnL-LCB gates.
- [ ] Add matrix candidates for manifest PnL-LCB experiments.
- [ ] Run targeted tests.

### Task 4: Selection Policy Against Over-Selection

**Files:**
- Modify: `src/panteon_v2/selection/flash_allocator.py`
- Test: `src/panteon_v2/tests/test_flash_allocator.py`

- [ ] Evaluate report output before changing selection.
- [ ] Add only the smallest gate needed by data, preferably via promotion manifest signal keys.
- [ ] Do not loosen execution/risk limits; previous 2025 run proved wider execution is harmful.

### Task 5: CI Gate Hardening

**Files:**
- Modify: `tools/run_panteon_flash_profitability_matrix.py`
- Test: `src/panteon_v2/tests/test_panteon_flash_profitability_matrix.py`

- [ ] Ensure matrix rows expose `panteon_lcb_dominance`, `ensemble_lift`, `regime_floor`, and churn metrics.
- [ ] Treat failed 2025 dominance as a failed production gate.

### Task 6: Verification and Backtests

**Files:**
- Output only: `Reports/` and `Results/`

- [ ] Run targeted tests after each behavior change.
- [ ] Run full pytest.
- [ ] Run 2025 backtest for the best PnL-LCB manifest candidate.
- [ ] Compare against current `cap10` baseline: Panteon `+3.756%`, best component `+18.153%`.
- [ ] Document whether the candidate is deployable.
