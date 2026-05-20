# Panteon Real-Loss Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce NoTrade/raw-zero caused by over-strict real-loss gating while preserving protection against persistent losers.

**Architecture:** Add an opt-in Strategist rescue path that bypasses short-horizon `v3_real_loss_kill` only when a candidate has moderate real loss, strong virtual edge, and recent actionable shadow evidence. Keep persistent-loss kill and hard policy unchanged.

**Tech Stack:** Python, pytest, Panteon v2 Strategist/Retrodate runner.

---

### Task 1: Real-Loss Rescue Config And Behavior

**Files:**
- Modify: `src/panteon_v2/selection/strategist.py`
- Test: `src/panteon_v2/tests/test_strategist.py`

- [ ] **Step 1: Write the failing test**

Add a test where `RiskyPlayer` has small real loss beyond `v3_real_loss_kill_pnl_pct`, strong virtual/shadow actionability, and rescue enabled. Expected leader: `RiskyPlayer`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_strategist.py -k real_loss_rescue`
Expected: FAIL because config fields and rescue behavior do not exist.

- [ ] **Step 3: Implement minimal rescue**

Add config fields:
- `v3_real_loss_rescue_enabled`
- `v3_real_loss_rescue_min_virtual_pnl_pct`
- `v3_real_loss_rescue_max_virtual_dd_pct`
- `v3_real_loss_rescue_min_actionable_share`
- `v3_real_loss_rescue_min_recent_filled`
- `v3_real_loss_rescue_max_real_loss_pct`

In `_validate_v3_real_loss`, return no issue only when all rescue thresholds pass.

- [ ] **Step 4: Run targeted tests**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_strategist.py -k "real_loss_kill or real_loss_rescue"`
Expected: PASS.

### Task 2: Retrodate CLI

**Files:**
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

- [ ] **Step 1: Write CLI/config test**

Assert rescue flags are parsed and passed into `StrategistConfig`.

- [ ] **Step 2: Run test to verify it fails**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_retrodate_market_runner.py -k real_loss_rescue`
Expected: FAIL because CLI flags do not exist.

- [ ] **Step 3: Implement CLI/config wiring**

Add `--enable-v3-real-loss-rescue` and threshold flags to `RetrodateMarketConfig`, parser, summary, and `_build_strategist_config`.

- [ ] **Step 4: Run full tests**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests`
Expected: PASS.

### Task 3: Five-Year A/B

**Files:**
- Read: `Results/RetrodateMarket/RETRODATE_MARKET/*`

- [ ] **Step 1: Run 5-year replay**

Run the existing best fixed+solo10 command with `--enable-v3-real-loss-rescue`.

- [ ] **Step 2: Compare against best current run**

Compare PnL, DD, NoTrade, raw-zero, real trades, candidate diagnostics, soft allocator, and perfect oracle.

- [ ] **Step 3: Keep or reject**

Keep rescue only if PnL improves without materially worsening DD. Otherwise leave it opt-in and report rejected hypothesis.
