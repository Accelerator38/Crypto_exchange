# Production Shadow Tournament Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Panteon v2 match the intended architecture where all agents and players trade virtually each bar, while only the selected best player for the current regime trades on the real exchange.

**Architecture:** Add a stateful production shadow tournament with isolated virtual execution state per actor. The tournament updates shared `PerformanceMemory` but writes virtual order events only to local event logs, so real dashboards and open positions still reflect only real exchange execution.

**Tech Stack:** Python unittest, existing `panteon_v2` domain/execution/selection modules, existing `FakeExchange`, `TradeExecutor`, `PerformanceMemory`.

---

### Task 1: Add Shadow Tournament Tests

**Files:**
- Modify: `src/panteon_v2/tests/test_app.py`

- [ ] Write tests proving solo agents trade virtually even when no real leader is available.
- [ ] Write tests proving player profiles get virtual performance without double-counting agent labels.
- [ ] Write tests proving real exchange receives only selected leader signals, not all virtual actor signals.
- [ ] Run targeted tests and confirm they fail before implementation.

### Task 2: Implement `ProductionShadowTournament`

**Files:**
- Create: `src/panteon_v2/app/shadow_tournament.py`

- [ ] Add `ShadowStepSummary`.
- [ ] Add per-actor virtual runtime containing `FakeExchange`, `PositionTracker`, `SymbolHealthMonitor`, `RiskLimits`, and `TradeExecutor`.
- [ ] Execute every registered agent as a solo virtual actor.
- [ ] Execute every composed candidate player as a virtual actor.
- [ ] Use local event logs for virtual order events and shared `PerformanceMemory` for ratings.

### Task 3: Integrate Into Production Loop

**Files:**
- Modify: `src/panteon_v2/app/bootstrap.py`
- Modify: `src/panteon_v2/app/startup.py`
- Modify: `src/panteon_v2/app/main_loop.py`
- Modify: `src/panteon_v2/app/output_writer.py`

- [ ] Attach a tournament to the production pipeline during startup.
- [ ] Run the tournament after regime detection and before quarantine recompute/leader selection.
- [ ] Recompose real candidates after virtual updates and quarantine recompute.
- [ ] Expose shadow counters in `StepResult`, `status.json`, and `trading.log`.

### Task 4: Verify and Commit

**Files:**
- Test: `src/panteon_v2/tests/test_app.py`
- Test: full `panteon_v2.tests.run_all`

- [ ] Run targeted tests until green.
- [ ] Run full v2 unittest suite.
- [ ] Run dashboard/positions tests.
- [ ] Compile modified files.
- [ ] Commit and push to `Panteon_v2`.
