# Panteon Flash Selected-Subset Profit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add conservative profit-improvement mechanisms based on Flash-selected subset statistics, then test them against the current best profile.

**Architecture:** Build a manifest from historical Flash-selected attribution rows instead of using broad shadow LCB. FlashAllocator consumes the manifest as opt-in score boosts, do-not-demote keys, and bounded risk multipliers. Partial profit lock is inspected separately and only wired if the current Signal/executor contract already supports it safely.

**Tech Stack:** Python, pytest, existing Panteon v2 retro runner, existing FlashAllocator.

---

### Task 1: Selected-Subset Manifest Builder

**Files:**
- Create: `src/panteon_v2/analysis/flash_selected_subset_manifest.py`
- Create: `tools/build_flash_selected_subset_manifest.py`
- Test: `src/panteon_v2/tests/test_flash_selected_subset_manifest.py`

- [ ] Write failing tests for cross-run positive cells, sparse-alpha allowlist, and bounded risk multiplier output.
- [ ] Implement a pure `build_selected_subset_manifest` function that joins multiple `flash_attribution_summary.json` files by `actor|symbol|action`.
- [ ] Require positive selected PnL in all configured confirmation runs before assigning score boost.
- [ ] Mark sparse/weak-shadow profitable selected cells as `do_not_demote_signal_keys`.
- [ ] Assign risk multipliers in the conservative range `0.75..1.15`.
- [ ] Add a CLI wrapper that writes JSON and Markdown.

### Task 2: FlashAllocator Manifest Consumption

**Files:**
- Modify: `src/panteon_v2/selection/flash_allocator.py`
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Modify: `src/panteon_v2/app/startup.py`
- Modify: `src/panteon_v2/app/output_writer.py`
- Test: `src/panteon_v2/tests/test_flash_allocator.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

- [ ] Add config fields for selected-subset score boosts, do-not-demote keys, and risk multiplier overrides.
- [ ] Add candidate audit fields for selected-subset boost/protection/risk multiplier.
- [ ] Apply score boost after ordinary gate score and before ranking.
- [ ] Skip LCB-based demotion for do-not-demote keys.
- [ ] Multiply open-signal `risk_mult` by the selected-subset override, bounded by config.
- [ ] Add CLI/startup parsing and run-summary output.

### Task 3: Partial Profit Lock Feasibility

**Files:**
- Inspect: `src/panteon_v2/domain/types.py`
- Inspect: `src/panteon_v2/execution/*`
- Test only if contract exists.

- [ ] Verify whether `Signal` and executor support partial closes without changing core position accounting.
- [ ] If unsupported, document as post-deploy work rather than changing execution semantics before live.

### Task 4: Testing and Report

**Files:**
- Create/update report under `Reports/`.

- [ ] Generate selected-subset manifest from 2025 + full best runs.
- [ ] Run 2026 H1 early-stop validation.
- [ ] If 2026 H1 improves or does not regress, run 2025 and full.
- [ ] Run sweep around current profile: risk `8/10/12`, max positions `6/8/10`, stale age `120/168/240`.
- [ ] Keep the current best profile unless a candidate improves full run without 2026 H1 regression.
- [ ] Run targeted and full pytest suites.
