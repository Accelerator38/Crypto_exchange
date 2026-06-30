# GeneticCore V2 Collapse Guards Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add train-time and selection-time guards against HOLD/month collapse, one-sided direction collapse, and specialist-router collapse.

**Architecture:** Keep the current genome topology and singleton deployment shape. Extend the existing fitness and v4 metrics with explicit empty-period and mean-direction penalties, then add a router-level collapse gate when all regimes select too few unique genomes. Do not change live promotion policy.

**Tech Stack:** Python, pytest, PowerShell launch scripts, existing GeneticCore v2 reporting.

---

### Task 1: Train-Time Collapse Penalties

**Files:**
- Modify: `Genetics_DL_Agents/crypto_genetics.py`
- Modify: `Genetics_DL_Agents/settings_genetic.txt`
- Modify: `tools/start_genetics_v4_heavy_training.ps1`
- Modify: `src/panteon_v2/tests/test_genetic_simulator_contracts.py`

- [x] Add failing tests that `_compute_fitness` demotes equal-return policies with too many near-zero periods.
- [x] Add failing tests that `_compute_fitness` demotes equal-return policies with high mean absolute direction bias, even when max-bias behavior already exists.
- [x] Add config constants for max zero-period rate, zero-period penalty weight, mean direction-bias max, and mean direction-bias penalty weight.
- [x] Add the two penalties to `_compute_fitness`.
- [x] Add heavy-training overrides for the new penalties.
- [x] Run focused simulator contract tests.

### Task 2: V4/Router Collapse Diagnostics

**Files:**
- Modify: `src/panteon_v2/analysis/genetics_validation.py`
- Modify: `src/panteon_v2/analysis/genetic_core_v2_experiments.py`
- Modify: `src/panteon_v2/tests/test_genetics_validation.py`
- Modify: `src/panteon_v2/tests/test_genetic_core_v2_experiments.py`

- [x] Add failing tests that `fitness_v4_robust_score` reports zero-period rate and mean direction-bias penalty.
- [x] Add failing tests that `score_specialist_router` rejects maps selecting fewer than two unique genomes across regimes.
- [x] Extend v4 metrics with zero-period rate/excess/penalty and mean direction-bias excess/penalty.
- [x] Extend router scoring with `router_collapse` failure and `unique_selected_genomes`.
- [x] Run focused analysis tests.

### Task 3: Evidence

**Files:**
- Output: pytest and compile logs only.

- [x] Run the focused GeneticCore test set.
- [x] Compile modified Python files.
- [x] Parse-check modified PowerShell launcher.
- [x] Report that no live promotion/trading was performed.
