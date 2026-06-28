# Panteon Live-Ready Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Panteon refuse unsafe live-size/live-restart attempts until futures replay, causal component memory, signal-source diagnostics, and paper/shadow canary checks pass.

**Architecture:** Keep real orders gated by an explicit pre-flight verifier. Add causal component-memory as a replay/runtime artifact that only uses prior-bar data. Add futures data builders as separate tools so replay data preparation is auditable and does not change live adapters.

**Tech Stack:** Python, pytest, existing `panteon_v2` runtime, `tools/run_panteon3_pre_live_matrix.py`, `Start_panteon_v3.py`.

---

### Task 1: Live Pre-Flight Guard

**Files:**
- Create: `src/panteon_v2/app/live_preflight.py`
- Modify: `Start_panteon_v3.py`
- Test: `src/panteon_v2/tests/test_live_preflight.py`, `src/panteon_v2/tests/test_start_panteon_v3.py`

- [ ] Write tests that a failed/stale matrix summary blocks `live_futures`.
- [ ] Write tests that shadow/paper canary requirements are checked: nonzero signals, orders/fills, positive expectancy, reconcile OK.
- [ ] Implement pure verifier returning structured pass/fail reasons.
- [ ] Wire launcher to require pass for `live_futures`, while allowing `paper`, `paper_live_feed`, and `shadow_live_feed`.
- [ ] Run targeted launcher/preflight tests.

### Task 2: Causal Component-Memory Feed

**Files:**
- Create: `src/panteon_v2/selection/component_memory.py`
- Modify: `src/panteon_v2/selection/flash_allocator.py`
- Test: `src/panteon_v2/tests/test_component_memory.py`, `src/panteon_v2/tests/test_flash_allocator.py`

- [ ] Write tests for prior-bar filtering by actor/symbol/regime/action.
- [ ] Write tests that current-bar or future component stats are ignored.
- [ ] Implement component-memory loading/scoring helpers.
- [ ] Teach promotion-derived router to consume optional causal memory before falling back to current candidate rows.
- [ ] Run Flash/router tests.

### Task 3: Futures Replay Data

**Files:**
- Create: `tools/build_exchange_futures_retrodate.py`
- Modify: `tools/run_panteon3_pre_live_matrix.py`
- Test: `src/panteon_v2/tests/test_build_exchange_futures_retrodate.py`, `src/panteon_v2/tests/test_panteon3_pre_live_matrix.py`

- [ ] Write tests for normalizing MEXC/BITGET OHLCV rows into `crypto_1m_<year>_all_symbols.csv`.
- [ ] Implement offline parser and downloader shell around exchange-specific endpoints.
- [ ] Add matrix helper flags for exchange futures data dirs.
- [ ] Run tool tests and a small offline fixture build.

### Task 4: Specialist Signal-Source Diagnostics

**Files:**
- Create or modify: `tools/diagnose_component_signal_sources.py`
- Modify: `tools/diagnose_promotion_derived_components.py`
- Test: `src/panteon_v2/tests/test_component_signal_source_diagnostics.py`

- [ ] Write tests that per-bar reasons distinguish `inactive`, `closed<trades`, `expectancy<=0`, `non-open action`, `foreign owner`, and missing source stream.
- [ ] Add actor/action/source-stream grouping for `CarryFlowAgentV2`, `MomentumScalper`, `LiveVolCompress`, `LiveCrashHunter`.
- [ ] Run diagnostics against latest matrix JSONL.

### Task 5: Shadow/Paper Canary Runner

**Files:**
- Create: `tools/run_panteon3_live_canary_check.py`
- Test: `src/panteon_v2/tests/test_panteon3_live_canary_check.py`

- [ ] Write tests that zero signals/orders/fills fails.
- [ ] Write tests that negative expectancy after costs fails.
- [ ] Write tests that reconcile warnings fail.
- [ ] Implement read-only checker over latest Results sessions.
- [ ] Run against current MEXC/BITGET sessions and report verdict.

### Task 6: Verification

**Files:**
- Update: `docs/PANTEON_3_LIVE_READY_FINAL_BACKLOG_2026-06-27.md`

- [ ] Run targeted pytest set.
- [ ] Run py_compile for new modules/tools.
- [ ] Run a matrix/canary smoke check.
- [ ] Do not restart live workers unless all pre-flight gates pass.
