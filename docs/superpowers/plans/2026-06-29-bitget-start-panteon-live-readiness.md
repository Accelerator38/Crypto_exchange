# Bitget Start Panteon Live Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace fragmented root launchers with `Start_panteon.py`, support `trade_regime=multi` and `trade_regime=singlton(<actor>)`, verify BITGET-only paper/matrix gates, and only attempt real live if preflight passes.

**Architecture:** Keep the existing `start_production` runtime and live preflight guard. Move launcher defaults into one root script with exchange switches and singleton actor parsing, while preserving shadow-stat collection for non-selected actors through the existing pipeline. Safety remains gate-driven: canary and matrix must pass before `live_futures` workers launch.

**Tech Stack:** Python, pytest, existing `panteon_v2` startup/live preflight, `tools/run_panteon3_single_component_canary.py`, `tools/run_panteon3_pre_live_matrix.py`.

---

### Task 1: Unified Launcher Tests

**Files:**
- Create: `src/panteon_v2/tests/test_start_panteon.py`
- Create/Modify: `Start_panteon.py`

- [x] Write tests that `BITGET=ON`, `MEXC=OFF` enables only BITGET by default.
- [x] Write tests that `_parse_trade_regime("multi")` returns multi mode with no singleton actor.
- [x] Write tests that `_parse_trade_regime("singlton(GeneticsCore)")` returns singleton mode and actor `GeneticsCore`.
- [x] Write tests that `singlton(genetic_core)` normalizes to `GeneticsCore`.
- [x] Write tests that worker args use `Start_panteon.py`, not legacy launchers.
- [x] Write tests that failed live preflight blocks `live_futures` launch and makes `main()` return nonzero.

### Task 2: Unified Launcher Implementation

**Files:**
- Create: `Start_panteon.py`
- Delete: root legacy start scripts after tests pass.

- [x] Port safe process spawning, locks, env loading, and live preflight checks into `Start_panteon.py`.
- [x] Add top-level exchange switches: `BITGET = "ON"`, `MEXC = "OFF"`.
- [x] Add `trade_regime = "multi"` and support `singlton(<actor>)`.
- [x] In singleton worker mode, configure Flash allocator whitelist to the selected actor and keep other agents available only for shadow/stat collection where runtime supports it.
- [x] Keep `live_futures` blocked unless `run_live_preflight(exchange, mode)` passes.

### Task 3: BITGET-Only Verification

**Files:**
- Use: `tools/run_panteon3_single_component_canary.py`
- Use: `tools/run_panteon3_pre_live_matrix.py`
- Use: `tools/run_panteon3_live_canary_check.py`

- [x] Run BITGET-only paper canary after negative-context deny.
- [x] Rebuild canary summary and confirm `signals/orders/fills`, expectancy, flat positions, and reconcile status.
- [x] Run fresh BITGET-focused matrix or nearest available pre-live matrix with current flags.
- [x] Do not start real orders if any preflight reason remains.

### Task 4: Documentation and Cleanup

**Files:**
- Create: `docs/PANTEON_3_OPERATOR_GUIDE.md`
- Update: legacy cleanup inventory if needed.

- [x] Document launcher usage, exchange switches, `trade_regime=multi`, `trade_regime=singlton(<actor>)`.
- [x] Document agents vs players, Flash routing, market regime routing, canary/matrix/preflight gates.
- [x] Delete tracked legacy root launchers after `Start_panteon.py` tests pass.
- [ ] Commit and push `panteon-3`.

### Task 5: Live Decision

**Files:**
- Use: `Start_panteon.py`

- [x] Re-run live preflight for BITGET.
- [ ] If passed, start BITGET live worker with tiny risk cap and singleton/multi settings explicitly documented.
- [x] If failed, report exact blockers and do not send real orders.

Current decision: no-go for real BITGET orders. Latest pre-flight fails with `matrix_failed`, `canary_failed`, and `canary_nonpositive_expectancy`.
