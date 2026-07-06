# Prelive Hardening From Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the 2026-07-02 pre-live report into guarded diagnostics and offline/R&D gates without enabling live trading.

**Architecture:** Keep live admission conservative. Add observable diagnostics and accounting checks around existing canary/matrix/readiness tools, and keep genetics changes in reselection/promotion tooling.

**Tech Stack:** Python CLI tools under `tools/`, pytest tests under `src/panteon_v2/tests/`, JSON report artifacts under `Reports/`.

---

### Task 1: Canary Cost Credibility

**Files:**
- Modify: `tools/run_panteon3_live_canary_check.py`
- Test: `src/panteon_v2/tests/test_panteon3_live_canary_check.py`

- [ ] Add a failing test where a canary has fills and gross PnL but no fee/slippage fields; expected summary includes a `cost_attribution_missing` warning and does not pass by silent zero costs.
- [ ] Run `python -m pytest src/panteon_v2/tests/test_panteon3_live_canary_check.py::test_canary_summary_warns_when_filled_trade_has_no_cost_attribution -q` and confirm it fails because the warning is absent.
- [ ] Add minimal cost attribution detection to the canary summary.
- [ ] Re-run the test and then the whole canary test module.

### Task 2: Matrix Accounting Credibility

**Files:**
- Modify: `tools/run_panteon3_pre_live_matrix.py`
- Test: `src/panteon_v2/tests/test_panteon3_pre_live_matrix.py`

- [ ] Add a failing test where losing trades exist but gross loss is zero; expected summary reports `accounting_warnings` and caps profit factor as unreliable instead of treating it as real infinity.
- [ ] Run the focused test and confirm it fails.
- [ ] Add matrix accounting warnings for losing trades with missing gross loss, empty skip windows, and unreliable profit factor.
- [ ] Re-run focused and matrix tests.

### Task 3: Live Diagnostics For MEXC Zero Signal Path

**Files:**
- Modify: `tools/run_panteon3_live_canary_check.py`
- Test: `src/panteon_v2/tests/test_panteon3_live_canary_check.py`

- [ ] Add a failing test that compares MEXC and BITGET canary summaries and exports a per-symbol/exchange diagnostic payload when one exchange has context rows but zero signals.
- [ ] Run the focused test and confirm it fails.
- [ ] Add `zero_signal_diagnostics` with context rows, signal counts, and first rejection reason counts when available.
- [ ] Re-run focused and canary tests.

### Task 4: Readiness Stale/Orphan Cleanup Guidance

**Files:**
- Modify: `tools/build_panteon_prelive_readiness.py`
- Test: `src/panteon_v2/tests/test_prelive_readiness_tool.py`

- [ ] Add a failing test that stale/orphan live status warnings include concrete cleanup commands or paths.
- [ ] Run the focused test and confirm it fails.
- [ ] Add cleanup hints to stale/orphan warning payloads without converting them to blockers.
- [ ] Re-run focused and readiness tests.

### Task 5: Genetics R&D Gate Hardening

**Files:**
- Modify: `tools/select_genetics_candidate.py`
- Modify: `src/panteon_v2/analysis/genetics_promotion.py` if promotion helpers own the contract logic
- Test: relevant genetics tests under `src/panteon_v2/tests/`

- [ ] Add a failing test that `selected_is_baseline=True` produces a hard failed selection, not a tie-style eligible candidate.
- [ ] Add a failing test that NaN metrics and missing `min_per_symbol_lcb` are explicit failures.
- [ ] Add a failing test that turnover pressure contributes to candidate score before promotion, not only at the final gate.
- [ ] Add minimal selector changes to expose those failures and keep live flags off.
- [ ] Re-run focused genetics tests.

### Task 6: Matrix Policy Output, Not Live Enablement

**Files:**
- Modify: `tools/run_panteon3_pre_live_matrix.py`
- Modify: `src/panteon_v2/app/live_preflight.py` only if a new policy field is required
- Test: `src/panteon_v2/tests/test_live_preflight.py`, `src/panteon_v2/tests/test_panteon3_pre_live_matrix.py`

- [ ] Add a failing test that matrix summary can recommend `single_component_candidate=LiveOIBreakout` as a separate artifact path while live preflight still blocks if current candidate does not beat best component.
- [ ] Add minimal summary fields for the single-component path and risk-adjusted research fields.
- [ ] Re-run live preflight and matrix tests.
