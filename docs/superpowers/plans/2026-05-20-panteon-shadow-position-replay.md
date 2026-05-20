# Panteon Shadow Position Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let real Panteon causally replay a selected leader's current executable shadow position when the leader has no fresh raw signal but its shadow book still has a valid open opportunity.

**Architecture:** Extend the shadow open-position registry with executable fields, then add a real-signal replay path in `main_loop` that converts a selected leader's fresh shadow open position into guarded real open signals. Keep this behind existing handoff flags so current safety policy still controls late entries.

**Tech Stack:** Python dataclasses, existing Panteon v2 main loop, shadow tournament, pytest.

---

### Task 1: Extend Shadow Position Registry

**Files:**
- Modify: `src/panteon_v2/app/shadow_tournament.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [ ] **Step 1: Write failing test**

Add an assertion that `ProductionShadowTournament.last_player_open_positions()` includes `entry_price`, `qty`, `age_bars`, `unrealized_pnl_usd`, and `fresh` for an open player position.

- [ ] **Step 2: Run test**

Run: `.\.venv\Scripts\python.exe -m pytest -q src/panteon_v2/tests/test_app.py::TestProductionApp::test_shadow_tournament_exposes_player_open_positions_for_position_gate`

- [ ] **Step 3: Implement payload fields**

Update `_position_payloads()` to accept the current market and emit deterministic primitive fields from each `TrackedPosition`.

- [ ] **Step 4: Re-run test**

Run the same test and confirm it passes.

### Task 2: Replay Selected Shadow Position

**Files:**
- Modify: `src/panteon_v2/app/main_loop.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [ ] **Step 1: Write failing test**

Add a test where the strategist selects a leader, `leader.vote()` emits no raw signals, but `_pending_shadow_player_positions` contains a fresh long BTC position for that selected leader. The expected real step fills one replayed open signal owned by the selected leader.

- [ ] **Step 2: Run test**

Run: `.\.venv\Scripts\python.exe -m pytest -q src/panteon_v2/tests/test_app.py::<new_test>`

- [ ] **Step 3: Implement replay helper**

Add helpers in `main_loop` to read current shadow position payloads, enforce freshness/flat-real compatibility, convert side to a half-size futures open signal at current market price, pass through `filter_real_signals_against_tracker`, and mark the fallback reason as shadow-position replay.

- [ ] **Step 4: Re-run focused tests**

Run the new test plus existing fallback tests.

### Task 3: Verify And Retrotest

**Files:**
- No code changes unless tests or retro diagnostics identify a regression.

- [ ] **Step 1: Full unit tests**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

- [ ] **Step 2: Short diagnostic retrotest**

Run a short RetrodateMarket replay with executable-soft confirmed score and fresh handoff enabled.

- [ ] **Step 3: Full 2022-2026 retrotest if short improves**

Run full benchmark only if the short run improves PnL/regret without simply increasing NoTrade.
