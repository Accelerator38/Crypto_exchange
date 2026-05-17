# Panteon v3 Allocator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first v3 allocator layer so Pantheon selection is measurable by regret and driven by rolling/decayed real decision quality.

**Architecture:** Add a small analysis module for post-run oracle/regret metrics, a focused rolling score module for decision-time scoring, then wire `Strategist` to optionally use that score. Keep the v2 pipeline intact and make v3 behavior opt-in through `StrategistConfig`.

**Tech Stack:** Python, existing `panteon_v2` domain/selection/memory types, pytest, existing Retrodate market runner and dashboards.

---

### Task 1: Oracle/Regret Analyzer

**Files:**
- Create: `src/panteon_v2/analysis/oracle_regret.py`
- Test: `src/panteon_v2/tests/test_oracle_regret.py`
- Modify: `src/panteon_v2/analysis/__init__.py`

- [ ] **Step 1: Write failing tests**

Cover:
- `compute_leader_regret()` compares selected leader PnL against best realized leader PnL.
- `compute_leader_regret()` reports NoTrade share, profitable leader share, best leader, selected PnL, best PnL, and regret.
- Empty input returns zeroed metrics.

- [ ] **Step 2: Run RED**

Run: `.\\.venv\\Scripts\\python.exe -m pytest src\\panteon_v2\\tests\\test_oracle_regret.py -q`

Expected: fail because module does not exist.

- [ ] **Step 3: Implement analyzer**

Create dataclasses:
- `LeaderPnL(label, pnl_usd, bars, trades)`
- `RegretReport(selected_label, selected_pnl_usd, best_label, best_pnl_usd, regret_usd, no_trade_share_pct, profitable_leader_share_pct, leader_count)`

Implement deterministic pure function:
- `compute_leader_regret(rows: Sequence[LeaderPnL], selected_label: str = "Panteon") -> RegretReport`

- [ ] **Step 4: Run GREEN**

Run: `.\\.venv\\Scripts\\python.exe -m pytest src\\panteon_v2\\tests\\test_oracle_regret.py -q`

Expected: pass.

### Task 2: Rolling/Decayed Decision Score

**Files:**
- Create: `src/panteon_v2/selection/rolling_score.py`
- Test: `src/panteon_v2/tests/test_rolling_score.py`

- [ ] **Step 1: Write failing tests**

Cover:
- recent positive trades beat old positive trades when decay is enabled;
- drawdown and negative expectancy reduce score;
- no data returns configured no-data score;
- enough recent real trades can override stale virtual strength.

- [ ] **Step 2: Run RED**

Run: `.\\.venv\\Scripts\\python.exe -m pytest src\\panteon_v2\\tests\\test_rolling_score.py -q`

Expected: fail because module does not exist.

- [ ] **Step 3: Implement rolling score**

Create:
- `RollingDecisionScoreConfig`
- `RollingDecisionScoreInput`
- `score_rolling_decision()`

Use existing `Metrics` inputs for real and virtual memory. Score formula:
- real PnL and win rate are primary when real closed trades are present;
- virtual PnL is a fallback with lower weight;
- max drawdown, negative PnL, and inactivity penalize;
- confidence scales by real closed trades.

- [ ] **Step 4: Run GREEN**

Run: `.\\.venv\\Scripts\\python.exe -m pytest src\\panteon_v2\\tests\\test_rolling_score.py -q`

Expected: pass.

### Task 3: Strategist Integration

**Files:**
- Modify: `src/panteon_v2/selection/strategist.py`
- Test: `src/panteon_v2/tests/test_strategist.py`

- [ ] **Step 1: Write failing tests**

Cover:
- when v3 rolling score is enabled, a player with stronger real metrics outranks a player with stronger virtual-only metrics;
- `NoTrade` is selected when all candidates have negative rolling real score;
- existing default v2 behavior remains unchanged when the flag is disabled.

- [ ] **Step 2: Run RED**

Run: `.\\.venv\\Scripts\\python.exe -m pytest src\\panteon_v2\\tests\\test_strategist.py -q`

Expected: new v3 tests fail.

- [ ] **Step 3: Implement opt-in config**

Add `use_v3_rolling_score: bool = False` to `StrategistConfig`.
When enabled, `_score_player_detail()` computes candidate score from `score_rolling_decision()` using:
- `real_perf.get(player.label, regime)` and aggregate fallback;
- `perf.get(player.label, regime)` and agent aggregate fallback;
- existing affinity, execution penalty, hysteresis, TTL, and promotion guards remain active.

- [ ] **Step 4: Run GREEN**

Run: `.\\.venv\\Scripts\\python.exe -m pytest src\\panteon_v2\\tests\\test_strategist.py -q`

Expected: pass.

### Task 4: Retrodate Runner Metrics

**Files:**
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

- [ ] **Step 1: Write failing tests**

Cover:
- runner accepts `--use-v3-rolling-score`;
- run summary includes `use_v3_rolling_score`;
- comparison report can include regret metrics when available.

- [ ] **Step 2: Implement CLI plumbing**

Pass `use_v3_rolling_score=True` into `StrategistConfig` when flag is provided.

- [ ] **Step 3: Run focused tests**

Run:
- `.\\.venv\\Scripts\\python.exe -m pytest src\\panteon_v2\\tests\\test_retrodate_market_runner.py -q`
- `.\\.venv\\Scripts\\python.exe -m pytest src\\panteon_v2\\tests\\test_strategist.py src\\panteon_v2\\tests\\test_rolling_score.py src\\panteon_v2\\tests\\test_oracle_regret.py -q`

### Task 5: 5-Year Retest and Comparison

**Files:**
- Output only under `Results/RetrodateMarket/RETRODATE_MARKET/<run_id>/`

- [ ] **Step 1: Run full suite**

Run: `.\\.venv\\Scripts\\python.exe -m pytest -q`

- [ ] **Step 2: Run 5-year retest**

Run existing Retrodate command with `--use-v3-rolling-score`, optional `GeneticsGenomeEnsemble`, and real-promotion gate enabled.

- [ ] **Step 3: Compare**

Report:
- PnL;
- NoTrade%;
- regret;
- profitable leader share;
- drawdown/open-position state;
- top real contributors and regressions.
