# Panteon Flash Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an opt-in Panteon_Flash decision path that selects one best actor per symbol and records auditable per-symbol decisions.

**Architecture:** Add a focused `FlashAllocator` module in `panteon_v2.selection`, wire it into `ProductionPipeline` as an opt-in path, and keep the legacy `Strategist` path intact. The allocator compares solo agents and ensemble players through a shared actor protocol and returns structured decision rows for debugging.

**Tech Stack:** Python, existing `panteon_v2` domain/selection/memory types, pytest, existing production `main_loop`.

---

### Task 1: Flash Allocator Kernel

**Files:**
- Create: `src/panteon_v2/selection/flash_allocator.py`
- Create: `src/panteon_v2/tests/test_flash_allocator.py`
- Modify: `src/panteon_v2/selection/__init__.py`

- [x] **Step 1: Write failing allocator tests**

Create tests that import `FlashAllocator`, `FlashAllocatorConfig`, and `FlashDecision`. Cover per-symbol actor selection, quarantine rejection, ensemble actor metadata, and `NoTrade` when all candidates are inactive.

- [x] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_flash_allocator.py -q`

Expected: fail because `panteon_v2.selection.flash_allocator` does not exist.

- [x] **Step 3: Implement allocator**

Add immutable dataclasses `FlashCandidateAudit`, `FlashDecision`, and `FlashAllocatorConfig`. Implement `FlashAllocator.decide(market, agents, players, signal_id_start, actionable_labels=())`.

- [x] **Step 4: Run GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_flash_allocator.py -q`

Expected: pass.

### Task 2: Actor Metadata

**Files:**
- Modify: `src/panteon_v2/selection/player.py`
- Test: `src/panteon_v2/tests/test_flash_allocator.py`

- [x] **Step 1: Write metadata assertions**

Assert that `EnsemblePlayer` decisions report `actor_type == "ensemble"` and solo agents report `actor_type == "agent"`.

- [x] **Step 2: Implement metadata properties**

Add simple `actor_type` properties where useful, but keep compatibility with existing `Player` protocol and tests.

- [x] **Step 3: Run focused tests**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_flash_allocator.py -q`

Expected: pass.

### Task 3: Pipeline Wiring

**Files:**
- Modify: `src/panteon_v2/app/bootstrap.py`
- Modify: `src/panteon_v2/app/main_loop.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [x] **Step 1: Write failing main-loop test**

Add a test showing Flash mode executes BTC from one actor and ETH from another actor on the same bar, and `StepResult.causal_decision["flash_decisions"]` contains both rows.

- [x] **Step 2: Run RED**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_app.py -q`

Expected: the new test fails because Flash mode is not wired.

- [x] **Step 3: Wire opt-in mode**

Add `flash_enabled: bool` and `flash_allocator: Optional[FlashAllocator]` to `ProductionPipeline`. In `main_loop`, branch after shadow updates: if enabled, collect composed players and registry agents, run the allocator, filter resulting signals through existing real-signal guard, execute them, and build `StepResult`.

- [x] **Step 4: Run GREEN**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_app.py src\panteon_v2\tests\test_flash_allocator.py -q`

Expected: pass.

### Task 4: Regression Verification

**Files:**
- No new files.

- [x] **Step 1: Run selection/app focused suite**

Run: `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_flash_allocator.py src\panteon_v2\tests\test_app.py src\panteon_v2\tests\test_strategist.py -q`

Expected: pass.

- [x] **Step 2: Run full suite if focused suite is clean**

Run: `.\.venv\Scripts\python.exe -m pytest -q`

Expected: pass or report exact failures.

Verified result: `712 passed, 13 subtests passed in 18.19s`.
