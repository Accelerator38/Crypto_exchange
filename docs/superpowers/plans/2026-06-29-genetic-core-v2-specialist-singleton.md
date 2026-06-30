# GeneticCore V2 Specialist-to-Singleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a measurable GeneticCore v2 experiment that tests regime specialists, teacher-feature seeding, rolling walk-forward validation, directional-bias penalties, regime-collapse penalties, month concentration, and per-symbol LCB while keeping live deployment as one probation-only `GeneticsCore`.

**Architecture:** Keep the current genetic trainer and post-training evaluator as the execution engine. Add a thin experiment layer that scores specialist-to-singleton candidates from existing contract reports, extends v4 fitness with direction/regime collapse penalties, and wires a bounded v2 training launch with teacher-agent BC seeds and stricter promotion diagnostics. Do not introduce live ensembles.

**Tech Stack:** Python, pytest, PowerShell launch scripts, existing `crypto_genetics.py` evaluator and Panteon v2 analysis helpers.

---

### Task 1: V2 Robust Fitness Penalties

**Files:**
- Modify: `src/panteon_v2/analysis/genetics_validation.py`
- Modify: `src/panteon_v2/tests/test_genetics_validation.py`

- [x] Add failing tests that `fitness_v4_robust_score` penalizes a one-sided short-only profile even when returns match a balanced profile.
- [x] Add failing tests that `fitness_v4_robust_score` penalizes regime collapse when only one regime produces positive returns.
- [x] Add `period_long_slot_rates`, `period_short_slot_rates`, `period_net_direction_biases`, `max_direction_bias_abs`, `direction_bias_penalty_weight`, `min_regime_positive_rate`, and `regime_collapse_penalty_weight` parameters.
- [x] Return `mean_long_slot_rate`, `mean_short_slot_rate`, `mean_abs_net_direction_bias`, `regime_positive_rate`, `direction_bias_penalty`, and `regime_collapse_penalty`.
- [x] Gate failures with `direction_bias` and `regime_collapse`.
- [x] Run `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_genetics_validation.py -q`.

### Task 2: Specialist-to-Singleton Report Scoring

**Files:**
- Create: `src/panteon_v2/analysis/genetic_core_v2_experiments.py`
- Create: `src/panteon_v2/tests/test_genetic_core_v2_experiments.py`

- [x] Add tests for selecting one genome per regime from validation reports while returning one singleton manifest payload.
- [x] Add tests that promotion is rejected when OOS/sanity fail even if validation improves.
- [x] Implement `candidate_metrics_from_mode`, `score_specialist_router`, and `summarize_singleton_candidate`.
- [x] Include diagnostics for selected regime map, validation/OOS/sanity metrics, promotion failures, and live policy `probation_only`.
- [x] Run `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_genetic_core_v2_experiments.py -q`.

### Task 3: Selector Wiring

**Files:**
- Modify: `tools/select_genetics_candidate.py`
- Modify: `tools/evaluate_genetics_contract.py`
- Modify: `src/panteon_v2/tests/test_genetic_core_v2_experiments.py`

- [x] Pass long/short/net-bias period fields from contract metrics into `fitness_v4_robust_score`.
- [x] Preserve existing JSON schema while adding v2 diagnostics under new keys.
- [x] Confirm old selection JSON can still be read by tests.
- [x] Run `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_genetics_validation.py src\panteon_v2\tests\test_genetic_core_v2_experiments.py -q`.

### Task 4: Bounded V2 Experiment Launch

**Files:**
- Create: `tools/start_genetic_core_v2_specialist_experiment.ps1`
- Modify: `tools/run_genetics_v4_post_training.ps1` only if the new report requires an extra post-training artifact.

- [x] Add a bounded launch script that calls the existing heavy-training script with `Population=180`, `Generations=12`, MEXC walk-forward data, position-state features, and teacher BC seeds.
- [x] Add v2 launch manifest fields: `experiment = GeneticCoreV2SpecialistToSingleton`, `specialist_regimes = crash,bearish,neutral,bullish`, `teacher_feature_agents = LiveOIBreakout,MomentumScalper,ResearchValidatorAgent`, and `live_policy = probation_only`.
- [x] Run the full bounded launch if data and runtime are available.
- [x] Run report replay on the current archive and a short `60x1` v2 direction-bias probe.
- [x] Run full `180x12` v2 training on `2026-06-30` and emit `genetic_core_v2_experiment_summary.json`.

### Task 5: Evidence Report

**Files:**
- Create: `tools/report_genetic_core_v2_experiment.py`
- Optional output: `Results/neiro_genetics/MEXC/<run>/genetic_core_v2_experiment_summary.json`

- [x] Read train/validation/OOS/sanity contract reports and selection outputs.
- [x] Emit a compact JSON summary comparing old `best_genome`, archive candidates, and specialist-router candidate under the v2 penalties.
- [x] Include explicit theory verdicts: accepted, rejected, or inconclusive.
- [x] Run the report on the latest GeneticCore run and summarize results for the user.

### Self-Review

- Spec coverage: tasks cover specialists, teacher BC seed tracking, rolling/WFA diagnostics through existing contract reports, directional/regime/month/per-symbol penalties, and probation-only live policy.
- Scope control: this plan intentionally avoids a large neural rewrite until penalties and specialist-router evidence show a robust OOS lift.
- Ambiguity check: "teacher features" is implemented first as teacher BC seed metadata and scoring diagnostics; runtime teacher logits are a later task only if this experiment improves OOS.
