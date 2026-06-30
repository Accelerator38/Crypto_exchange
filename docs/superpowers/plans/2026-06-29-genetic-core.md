# Genetic Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `GeneticsCore` a stricter singleton candidate with hard genome loading, walk-forward-only validation, robust fitness penalties, abstention, exchange costs, signal-key penalties, and probation-only live execution.

**Architecture:** Keep the existing `GeneticsCore` label and `GeneticsV2AgentAdapter`, but add enforceable contracts around genome loading, action admission, and training/evaluation inputs. Training quality gates live in analysis helpers and scripts; live risk limits stay in startup/bootstrap settings.

**Tech Stack:** Python, pytest, existing Panteon v2 tooling, PowerShell launch scripts.

---

### Task 1: Runtime Contract And Abstention

**Files:**
- Modify: `src/panteon_v2/app/agent_bootstrap.py`
- Modify: `src/panteon_v2/shadow/adapters.py`
- Test: `src/panteon_v2/tests/test_panteon_genetics_manifest_loader.py`
- Test: `src/panteon_v2/tests/test_shadow.py`

- [x] Write failing tests that `GeneticsCore` registration raises when no explicit/default real genome exists.
- [x] Write failing tests that weak `regime_confidence`, low action confidence, or weak logit margin maps to `HOLD`.
- [x] Implement minimal genome-source requirement and adapter-level abstention.
- [x] Run targeted tests and keep existing manifest/router tests green.

### Task 2: Walk-Forward Training Contract

**Files:**
- Create: `src/panteon_v2/analysis/genetic_core_contracts.py`
- Modify: `tools/run_genetics_v4_post_training.ps1`
- Modify: `tools/start_genetics_v4_heavy_training.ps1`
- Test: `src/panteon_v2/tests/test_genetic_core_contracts.py`

- [x] Write failing tests for the exact walk-forward windows: train `2022-01-01..2023-12-31`, validation `2024`, OOS `2025`, sanity `2026-01-01..2026-06-30`.
- [x] Write failing tests rejecting alternate training/evaluation windows.
- [x] Implement reusable contract helpers and wire scripts to those defaults.

### Task 3: Robust Fitness Inputs

**Files:**
- Modify: `src/panteon_v2/analysis/genetic_core_contracts.py`
- Modify: `src/panteon_v2/analysis/genetics_validation.py`
- Test: `src/panteon_v2/tests/test_genetic_core_contracts.py`
- Test: `src/panteon_v2/tests/test_genetics_validation.py`

- [x] Write failing tests that fitness uses LCB, Calmar, CVaR, positive-window rate, turnover, invalid-open pressure, concentration, costs, and bad signal-key penalties.
- [x] Implement minimal scoring helpers that can be consumed by selection/training tools.
- [x] Run targeted validation tests.

### Task 4: Live Probation Only

**Files:**
- Modify: `settings.txt`
- Modify: `src/panteon_v2/app/startup.py`
- Test: `src/panteon_v2/tests/test_production_hardening.py`

- [x] Write failing tests requiring `GeneticsCore` risk <= `0.08`, daily trades <= `1`, shadow confirmation enabled, and realized-loss kill switch configured.
- [x] Implement parser/default hardening if settings are incomplete.
- [x] Run production-hardening tests.

### Task 5: Old-vs-New Retro Comparison

**Files:**
- Create or modify: `tools/run_genetic_core_comparison.py`
- Test: `src/panteon_v2/tests/test_genetic_core_contracts.py`

- [ ] Write a focused comparison helper test that reports old/new PnL, closed trades, and expectancy from selected leaderboards or retro outputs.
- [x] Run the fastest available retro/canary comparison; if full retro is too slow or blocked by missing genome, report the exact blocker.

Note: old full genome artifacts were not present in the workspace, so the completed comparison uses the saved old five-year retro report and the new walk-forward/OOS artifacts rather than a same-contract old-genome replay.
