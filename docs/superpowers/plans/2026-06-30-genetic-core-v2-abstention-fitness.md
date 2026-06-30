# GeneticCore V2 Abstention Fitness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add train-time pressure for abstention-quality opens and tighten concentration/direction penalties before the next GeneticCore v2 run.

**Architecture:** Keep the current genome topology stable. Add proxy confidence and logit-margin diagnostics to the existing position-aware forward pass, pass those period metrics into `_compute_fitness`, and penalize opens that would be rejected by live abstention. Strengthen v2 launch overrides for one-sided direction and single-month concentration.

**Tech Stack:** Python, pytest, PowerShell launch scripts, existing `crypto_genetics.py` evaluator.

---

### Task 1: Abstention Fitness Signals

**Files:**
- Modify: `Genetics_DL_Agents/crypto_genetics.py`
- Modify: `src/panteon_v2/tests/test_genetic_simulator_contracts.py`

- [x] Add failing tests that position-aware forward reports weak open confidence and weak logit-margin pressure.
- [x] Add failing tests that `_compute_fitness` demotes equal-return policies with weak open-confidence pressure.
- [x] Add config constants for minimum open confidence, minimum open margin, and penalty weights.
- [x] Accumulate per-genome weak-open confidence and margin pressure in `_batch_forward_position_aware_numpy`.
- [x] Pass those period metrics through GPU/single/CPU evaluator paths into `_compute_fitness`.
- [x] Penalize the new metrics in `_compute_fitness`.
- [x] Run focused simulator contract tests.

### Task 2: Stronger V2 Launch Penalties

**Files:**
- Modify: `Genetics_DL_Agents/settings_genetic.txt`
- Modify: `tools/start_genetics_v4_heavy_training.ps1`

- [x] Lower direction-bias tolerance and raise penalty weight for v2 training.
- [x] Lower outlier/month concentration tolerance and raise penalty weight.
- [x] Add abstention penalty overrides to the heavy-training launch script.
- [x] Parse-check PowerShell scripts.

### Task 3: Evidence

**Files:**
- Output: `Results/neiro_genetics/MEXC/<run>/...`

- [x] Run unit tests and Python compile checks.
- [x] Run a bounded smoke training probe with the new penalties.
- [x] Report whether confidence/margin pressure appears in training contract metrics.
