# Panteon Antonius And Static Rotator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Antonius-specific degradation, add a stronger static agent-set player with current-signal rotation, and verify the allocator on the 2022-2026 retrotest.

**Architecture:** Keep the existing candidate/scoring pipeline intact. Add a focused `RotatingAgentPlayer` that owns a fixed set of agents but chooses the first currently actionable agent from a regime-specific priority list. Protect composite/regime-switch labels from premature all-regime quarantine when only one regime has evidence.

**Tech Stack:** Python, `pytest`, existing Panteon v2 retrodate runner and report artifacts.

---

### Task 1: Quarantine Guard For Composite Players

**Files:**
- Modify: `src/panteon_v2/memory/quarantine.py`
- Modify: `src/panteon_v2/app/bootstrap.py`
- Modify: `src/panteon_v2/app/main_loop.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [ ] Add an optional protected label set to `QuarantineManager`.
- [ ] Ensure protected labels are not dynamically quarantined by `hopeless_in_all_regimes`.
- [ ] Wire default regime-switch and rotating labels as protected in the production pipeline.
- [ ] Add tests proving `Antonius_strategy` is not quarantined after one bad regime sample.

### Task 2: Better Antonius Regime Map

**Files:**
- Modify: `src/panteon_v2/app/bootstrap.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [ ] Replace the old Antonius map with a more defensible current result based map: `bullish=LiveOIBreakout`, `bearish=GeneticsCore`, `neutral=ResearchValidatorAgent`, `crash=LiveCrashHunter`.
- [ ] Add a second conservative Antonius variant that avoids untrained genetics when it is not available.
- [ ] Test that both variants compose when their agents are registered.

### Task 3: Static Rotating Agent Player

**Files:**
- Modify: `src/panteon_v2/selection/player.py`
- Modify: `src/panteon_v2/selection/__init__.py`
- Modify: `src/panteon_v2/app/bootstrap.py`
- Modify: `src/panteon_v2/app/main_loop.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [ ] Implement `RotatingAgentPlayer`, preserving the full fixed agent set in `agent_labels`.
- [ ] On each bar, use regime-specific priority order and emit the first non-HOLD agent signal.
- [ ] Add `Optimal_StaticRotator` with a compact static set: `LiveOIBreakout`, `ResearchValidatorAgent`, `GeneticsCore`, `LiveCrashHunter`, `MomentumScalper`, `Fixed_BearDefense` components where available as agents only.
- [ ] Compose rotating candidates in the main loop and include them in shadow and real selection.

### Task 4: Verification And Report

**Files:**
- Modify or create report scripts under `Results/neiro_genetics/RetrodateMarket_AB/`

- [ ] Run targeted tests for app/player/quarantine behavior.
- [ ] Run the full test suite if targeted tests pass.
- [ ] Run a 2022-2026 retrotest with optional genetics, fixed players, actionable fallback, executable soft score, and the new rotator.
- [ ] Compare against the last strict run: PnL, DD, NoTrade, raw-zero, regret, profitable leaders, Antonius, and the new static rotator.
- [ ] Produce a Markdown report and, if practical, a DOCX report.
