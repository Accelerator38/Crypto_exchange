# Panteon Flash Structural Retro Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the remaining Panteon Flash V8 structural items, rerun verification, run a five-year retrodate check with early-stop discipline, and produce a report.

**Architecture:** Move audit event emission and the Flash per-symbol decision path out of `app/main_loop.py` into focused modules while preserving the existing production behavior. Keep `main_loop.py` compatibility wrappers for tests and callers that import private helpers.

**Tech Stack:** Python, pytest, existing `panteon_v2` retrodate runner and report tooling.

---

### Task 1: Plan And Scope Lock

**Files:**
- Create: `docs/superpowers/plans/2026-05-24-panteon-flash-structural-retro-report.md`

- [x] **Step 1: Capture the requested scope**

Record that this iteration implements V8 structural cleanup, preserves existing Panteon Flash trading defaults, reruns automated tests, then runs a five-year retrodate test unless a shorter smoke run exposes a clear regression.

### Task 2: Audit Emission Module

**Files:**
- Create: `src/panteon_v2/app/audit_emission.py`
- Modify: `src/panteon_v2/app/main_loop.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [x] **Step 1: Write failing module-boundary tests**

Add tests that import `emit_candidate_audit_events` and `emit_flash_audit_events` from `panteon_v2.app.audit_emission` and assert they emit the same event types used by the existing main loop.

- [x] **Step 2: Run the focused tests and verify import failure**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_app.py -k "audit_emission or decision_path_module" -q`

Expected: FAIL because `panteon_v2.app.audit_emission` does not exist yet.

- [x] **Step 3: Move audit emission implementation**

Create `audit_emission.py` with the existing candidate and Flash audit emission logic. Replace main loop bodies with thin compatibility wrappers that call the new functions.

- [x] **Step 4: Run focused tests**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_app.py -k "audit_emission or decision_path_module" -q`

Expected: PASS.

### Task 3: Flash Decision Path Module

**Files:**
- Create: `src/panteon_v2/app/decision_paths/__init__.py`
- Create: `src/panteon_v2/app/decision_paths/flash.py`
- Modify: `src/panteon_v2/app/main_loop.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [x] **Step 1: Write failing module-boundary test**

Add a test that imports `FlashDecisionPathCallbacks` and `run_flash_decision_path` from `panteon_v2.app.decision_paths.flash`.

- [x] **Step 2: Run the focused test and verify import failure**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_app.py -k "decision_path_module" -q`

Expected: FAIL because `panteon_v2.app.decision_paths.flash` does not exist yet.

- [x] **Step 3: Extract the Flash path**

Move the body of `_run_flash_decision_path` into `run_flash_decision_path`. Pass local main-loop dependencies through a `FlashDecisionPathCallbacks` dataclass to avoid circular imports and keep the behavior unchanged.

- [x] **Step 4: Keep main loop wrapper stable**

Keep `_run_flash_decision_path` in `main_loop.py` as a wrapper that constructs callbacks and delegates to `run_flash_decision_path`.

### Task 4: Verification

**Files:**
- No production files expected beyond Tasks 2-3.

- [x] **Step 1: Run focused tests**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_app.py src\panteon_v2\tests\test_flash_allocator.py -q`

- [x] **Step 2: Run full test suite**

Run: `.\.venv\Scripts\python.exe -m pytest`

- [x] **Step 3: Run syntax check**

Run: `$env:PYTHONPYCACHEPREFIX='C:\Work\Crypto_exchange\.tmp_compile_cache'; .\.venv\Scripts\python.exe -m compileall -q src\panteon_v2`

Remove `.tmp_compile_cache` only after verifying it is inside `C:\Work\Crypto_exchange`.

### Task 5: Five-Year Retrodate And Report

**Files:**
- Inspect: existing `tools/run_panteon_flash_*.py`
- Generate: `Reports/PanteonFlashPreLive_20260524/...`

- [x] **Step 1: Identify the exact existing best-run command**

Read the existing Round3/Deny8/EntryRegime runner configuration and reuse it as the baseline unless the repo has a newer pre-live runner.

- [x] **Step 2: Run a shorter smoke period first**

Run a one-year or half-year period. Stop if it errors, collapses versus the previous baseline, or produces obviously invalid metrics.

- [x] **Step 3: Run the full five-year period**

Run the same configuration across the 2022-2026 period.

- [x] **Step 4: Build the report**

Generate or update the markdown/docx report with charts and a concise production-readiness conclusion.
