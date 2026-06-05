# Panteon v3 Production Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the current Panteon v3 changes across live trading, memory, agents, players, exchange adapters, Flash selection, and regime detection; fix blocking compatibility defects; verify; commit and push `panteon_v3`.

**Architecture:** Treat live runtime evidence, code contracts, and regression tests as separate evidence streams. Fix root causes only where tests or static/runtime checks identify broken contracts.

**Tech Stack:** Python 3.12, pytest, PowerShell, Panteon v2/v3 runtime modules, MEXC/BITGET live adapters.

---

### Task 1: Baseline And Live Health

**Files:**
- Read: `Start_panteon_v3.py`
- Read: `Results/MEXC/*_v2/status.json`
- Read: `Results/BITGET/*_v2/status.json`
- Read: `logs/v3_*_runtime.log`

- [ ] Verify current branch and dirty state with `git status --short`.
- [ ] Verify live workers with `Get-CimInstance Win32_Process`.
- [ ] Read latest non-RETRODATE live result folders for MEXC and BITGET.
- [ ] Confirm `step_error`, `kill_switch_reason`, stale status, and leader mismatch are absent or explained.

### Task 2: Static Contract Audit

**Files:**
- Modify only files with failing imports, syntax, or broken public contracts.
- Test: affected `src/panteon_v2/tests/test_*.py`

- [ ] Run compile checks on `Start_panteon_v3.py`, `src/panteon_runtime`, `src/panteon_v2`, `tools`, and `Genetics_DL_Agents`.
- [ ] Run focused import checks for startup, memory, agents, players, Flash, selector, regime detector, and exchange adapters.
- [ ] Fix any syntax/import/API contract failure with a minimal patch.

### Task 3: Behavioral Regression Audit

**Files:**
- Modify only modules implicated by failing tests.
- Test: `src/panteon_v2/tests`, `tests`

- [ ] Run focused tests for launcher, adapters, live state, performance memory, players, selector, Flash, app loop, regime detector, dashboards, and genetics integration.
- [ ] For each failure, identify root cause from the failing assertion or stack trace before editing.
- [ ] Add or update regression coverage when the missing behavior is not already covered.

### Task 4: Production Verification

**Files:**
- No planned code edits.

- [ ] Run the full practical pytest suite for Panteon v2 and top-level dashboard tests.
- [ ] Re-check live workers after tests.
- [ ] Re-read latest live `status.json` for MEXC and BITGET.
- [ ] Confirm no live config was silently changed.

### Task 5: Commit And Push

**Files:**
- Stage only intentional repo changes.

- [ ] Review final `git diff --stat` and `git status --short`.
- [ ] Commit with a production-audit message.
- [ ] Push current `panteon_v3` branch.
