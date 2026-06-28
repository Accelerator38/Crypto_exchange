# Panteon 3 Activity Execution Edge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a live-ready Panteon 3.0 candidate that explains and reduces structural NoTrade, prevents phantom positions, and proves any activity/edge improvement through replay, backtest, walk-forward, and paper/shadow before live arming.

**Architecture:** Split Panteon 3.0 into six contracts: observable gate funnel, causal actor router, controlled exploration admission, fill-confirmed execution, realistic cost/sizing model, and operator dashboards. Hard safety gates stay hard; statistical/economic gates become scored or risk-scaled when evidence is sparse. ML/admission and Genetics can add quality evidence, but cannot be the only path to eternal NoTrade.

**Tech Stack:** Python 3.12 in `.\.venv`, pytest, Panteon v2 Flash allocator/runtime, MEXC/BITGET adapters, JSONL replay artifacts, dashboard PNG/HTML renderers.

---

## Audit Findings To Preserve

- Current branch: `Panteon_ML_Predict`.
- Dirty tree: modified `Start_ML.py`, `Start_ML_BITGET.py`, runtime MEXC/BITGET connectors, `src/panteon_v2/app/main_loop.py`, `output_writer.py`, ML executor/runner/dashboard/adapter, tests, and run cmd files. New files include `src/panteon_v2/ml/admission.py`, `admission_promotion.py`, `supervisor.py`, ML runner tests, admission profile, and ops scripts.
- No current `python.exe` live bot process was found during audit.
- Latest saved live-like status snapshots are stale: MEXC `Results/MEXC/2026-06-16_21-29-17_v2/status.json` and BITGET `Results/BITGET/2026-06-16_21-29-25_v2/status.json` both timestamp around `2026-06-18T07:01Z`.
- Last MEXC run: `12505` bars, zero real raw/executable/signals/fills. Candidate totals: `1587890`, rejected `1511605`, eligible `76285`. Top rejection counts: `inactive=1363552`, `quarantined=79510`, `expected_edge_below_cost=37255`, `range_low_vol_actor_not_allowed=30314`.
- Last BITGET run: `12330` bars, zero real raw/executable/signals/fills. Candidate totals: `1954715`, rejected `1855756`, eligible `98959`. Top rejection counts: `inactive=1714022`, `range_low_vol_actor_not_allowed=94365`, `flash_symbol_degraded=44805`, `real_loss_veto=2526`.
- BITGET also reports `min_executable_notional ... exceeds approved $0.79`, so small account sizing and min-notional policy must be treated as a first-class blocker, not hidden behind NoTrade.
- Existing compact causal output drops candidate rows for `NoTrade` decisions in `src/panteon_v2/app/output_writer.py:699-704`, losing actor/action/score/LCB/risk detail exactly when it is needed most.
- Targeted tests with the project venv pass:
  - `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_executor.py src\panteon_v2\tests\test_ml_executor.py src\panteon_v2\tests\test_png_renderer.py src\panteon_v2\tests\test_allocation_diagnostics.py -q`
  - `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_flash_allocator.py -q`
  - `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_exchange_adapters.py src\panteon_v2\tests\test_live_state.py src\panteon_v2\tests\test_production_hardening.py -q`

---

### Task 1: Gate Funnel Diagnostics First

**Files:**
- Modify: `src/panteon_v2/app/output_writer.py`
- Modify: `src/panteon_v2/app/main_loop.py`
- Test: `src/panteon_v2/tests/test_app.py`
- Test: `src/panteon_v2/tests/test_png_renderer.py`

- [ ] **Step 1: Write failing tests for NoTrade candidate preservation**

Add a test near `test_flash_payload_summarizes_top_notrade_blockers` that builds a `FlashDecision` with selected `NoTrade`, one rejected positive candidate, and verifies `top_rejected_candidates` is not emptied.

```python
def test_no_trade_payload_keeps_top_rejected_candidate_details(self):
    decision = _flash_decision_with_candidates(
        selected_actor="NoTrade",
        reason="no_real_admission:expected_edge_below_cost",
        rejected=[
            {
                "label": "DefaultEnsemble",
                "actor_type": "ensemble",
                "action": "FUT_SHORT_FULL",
                "reason": "expected_edge_below_cost",
                "score": 5.2,
                "shadow_pnl_per_trade_lcb_usd": 0.18,
                "risk_mult": 1.0,
            }
        ],
    )
    compact = OutputWriter._compact_flash_decision(decision, top_rejected_candidates=3)
    assert compact["top_rejected_candidates"][0]["label"] == "DefaultEnsemble"
    assert compact["top_rejected_candidates"][0]["reason"] == "expected_edge_below_cost"
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_app.py -k "no_trade_payload_keeps_top_rejected_candidate_details or flash_payload_summarizes_top_notrade_blockers" -q
```

Expected before fix: the new assertion fails because `top_rejected_candidates` is empty for selected `NoTrade`.

- [ ] **Step 3: Preserve compact top rejected rows for NoTrade**

Change `OutputWriter._compact_flash_decision` so selected `NoTrade` still clears the full `candidates` array, but keeps `top_rejected_candidates` populated with compact candidate rows.

```python
if selected_actor == "NoTrade" and not decision.get("signal"):
    out["candidates"] = []
    return out
```

- [ ] **Step 4: Add per-symbol gate funnel summary to status**

Extend `_flash_causal_decision_payload` to include a `flash_gate_funnel_by_symbol` object with `candidate_count`, `eligible_candidate_count`, `rejected_candidate_count`, `top_blocker`, `original_selected_actor`, `original_actor_type`, and `reason` for every symbol.

- [ ] **Step 5: Verify diagnostics**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_app.py -k "flash_payload_summarizes_top_notrade_blockers or gate_funnel" -q
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_png_renderer.py -k "flash_diagnostics_explain_no_trade" -q
```

Expected: diagnostics tests pass and a selected `NoTrade` still exposes the rejected actor details needed for audit.

---

### Task 2: Offline Gate Funnel Analyzer

**Files:**
- Create: `tools/analyze_panteon_gate_funnel.py`
- Test: `src/panteon_v2/tests/test_gate_funnel_analyzer.py`

- [ ] **Step 1: Write analyzer tests**

Create tests that feed two small `causal_entry_decisions.jsonl` rows and assert counts for bars, symbols, selected actors, rejection reasons, nonzero real signals, and min-notional blockers.

```python
def test_gate_funnel_analyzer_counts_rejections(tmp_path):
    path = tmp_path / "causal_entry_decisions.jsonl"
    path.write_text(
        '{"timestamp":"t1","raw_signal_count":0,"n_signals":0,'
        '"real_universe_symbol_reject_reasons":{"BTC":"min_executable_notional $6 exceeds approved $1"},'
        '"flash_decisions":[{"symbol":"BTC","selected_actor":"NoTrade","reason":"no_real_admission:expected_edge_below_cost",'
        '"candidate_count":2,"rejected_candidate_count":1,"eligible_candidate_count":1,'
        '"candidate_rejection_counts":{"expected_edge_below_cost":1}}]}\n',
        encoding="utf-8",
    )
    report = analyze_files([path])
    assert report["bars"] == 1
    assert report["real_signal_bars"] == 0
    assert report["decision_reasons"]["no_real_admission:expected_edge_below_cost"] == 1
    assert report["real_universe_reasons"]["min_executable_notional"] == 1
```

- [ ] **Step 2: Implement `analyze_files` and CLI**

The script must stream JSONL line by line and output `gate_funnel_report.json` plus a short markdown table. It must not load 100MB files into memory at once.

- [ ] **Step 3: Run analyzer on latest stale MEXC/BITGET folders**

Run:

```powershell
.\.venv\Scripts\python.exe tools\analyze_panteon_gate_funnel.py Results\MEXC\2026-06-16_21-29-17_v2\causal_entry_decisions.jsonl --out tmp\gate_funnel_mexc_2026-06-16.json
.\.venv\Scripts\python.exe tools\analyze_panteon_gate_funnel.py Results\BITGET\2026-06-16_21-29-25_v2\causal_entry_decisions.jsonl --out tmp\gate_funnel_bitget_2026-06-16.json
```

Expected: reports reproduce the audit counts for zero real signal bars and top blockers.

---

### Task 3: Controlled Exploration Admission

**Files:**
- Modify: `src/panteon_v2/selection/flash_allocator.py`
- Modify: `src/panteon_v2/app/startup.py`
- Test: `src/panteon_v2/tests/test_flash_allocator.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [ ] **Step 1: Add failing tests for expected-edge and range-low-vol exploration**

Create tests where an actor has a current open signal, positive shadow score/closed sample, is blocked by `expected_edge_below_cost` or `range_low_vol_actor_not_allowed`, and is admitted only when exploration is enabled with a tiny `risk_mult`.

```python
def test_controlled_exploration_risk_sizes_expected_edge_below_cost_candidate(self):
    cfg = FlashAllocatorConfig(
        controlled_exploration_enabled=True,
        controlled_exploration_allowed_reasons=("expected_edge_below_cost",),
        controlled_exploration_risk_mult=0.05,
        controlled_exploration_max_daily_trades=1,
        fee_aware_admission_enabled=True,
    )
    decision = decide_single_candidate(cfg, label="DefaultEnsemble", reason="expected_edge_below_cost")
    assert decision.selected_actor == "DefaultEnsemble"
    assert decision.signal is not None
    assert decision.signal.risk_mult == 0.05
```

- [ ] **Step 2: Add config fields**

Add fields to `FlashAllocatorConfig`:

```python
controlled_exploration_enabled: bool = False
controlled_exploration_allowed_reasons: Tuple[str, ...] = ()
controlled_exploration_risk_mult: float = 0.05
controlled_exploration_min_shadow_score: float = 2.0
controlled_exploration_min_shadow_closed: int = 10
controlled_exploration_max_daily_trades: int = 1
controlled_exploration_max_open_positions: int = 1
```

- [ ] **Step 3: Implement admission as a risk-scaled override, not gate removal**

When the top rejected candidate fails only an allowed statistical/economic gate, and hard safety gates are clean, return the candidate with `selected_reasons` including `controlled_exploration` and cap `risk_mult`.

Hard gates that must not be bypassed: exchange health, symbol health, terminal deny unless exact probation override, denied symbol, denied regime, foreign position owner, promotion manifest hard deny, no market price, duplicate open, max positions, kill switch.

- [ ] **Step 4: Parse settings**

Add settings names in `startup.py`:

```text
v2_flash_controlled_exploration_enabled
v2_flash_controlled_exploration_allowed_reasons
v2_flash_controlled_exploration_risk_mult
v2_flash_controlled_exploration_min_shadow_score
v2_flash_controlled_exploration_min_shadow_closed
v2_flash_controlled_exploration_max_daily_trades
v2_flash_controlled_exploration_max_open_positions
```

- [ ] **Step 5: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_flash_allocator.py -k "controlled_exploration or range_low_vol or expected_edge_below_cost" -q
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_app.py -k "controlled_exploration" -q
```

Expected: exploration admits only explicitly configured candidates and audits the risk cap.

---

### Task 4: Realistic Small-Account Sizing And Min-Notional

**Files:**
- Modify: `src/panteon_v2/execution/risk_limits.py`
- Modify: `src/panteon_v2/execution/executor.py`
- Modify: `src/panteon_v2/app/output_writer.py`
- Test: `src/panteon_v2/tests/test_risk_limits.py`
- Test: `src/panteon_v2/tests/test_executor.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [ ] **Step 1: Write tests for min-notional as explicit blocker**

The test must assert that when approved notional is below exchange minimum by more than allowed upscale, the signal is blocked with reason `min_notional_exceeds_approved_risk`, not silently converted to NoTrade.

- [ ] **Step 2: Preserve realistic sizing**

Do not auto-upsize tiny balances into positions that violate configured risk. Add optional `min_notional_upscale_max_mult`, defaulting to the current safe behavior, and include actual approved notional, exchange min notional, and required multiplier in event/status diagnostics.

- [ ] **Step 3: Verify BITGET evidence**

Use a fixture with `approved=$0.79` and `min_executable_notional=$6.40` and assert the dashboard/status shows `capital_too_small_for_symbol` or `min_notional_exceeds_approved_risk`.

---

### Task 5: Causal Actor Router Instead Of Mean Ensemble Bias

**Files:**
- Create: `src/panteon_v2/selection/causal_actor_router.py`
- Modify: `src/panteon_v2/selection/flash_allocator.py`
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_causal_actor_router.py`
- Test: `src/panteon_v2/tests/test_flash_allocator.py`

- [ ] **Step 1: Write router tests**

Tests must cover:
- winner-take-best causal actor per symbol/regime;
- no ex-post oracle: only metrics available before the current bar are used;
- exploration floor for sparse positive actors;
- ensemble score cannot be `mean(component_scores)` unless the ensemble has its own trade sample.

- [ ] **Step 2: Implement router contract**

Expose a pure function:

```python
def route_actor(candidates, memory, *, symbol, regime, bar, config) -> RoutedActor:
    ...
```

`RoutedActor` must include `label`, `actor_type`, `score`, `score_source`, `sample_closed`, `expectancy`, `pnl_lcb`, `risk_mult`, and `reason`.

- [ ] **Step 3: Integrate behind an opt-in flag**

Add `v2_flash_causal_actor_router_enabled = off` default. Keep existing behavior when off.

- [ ] **Step 4: Verify router does not reduce activity to zero**

Use replay fixtures with MEXC and BITGET stale causal rows. Expected: router report should select a non-NoTrade controlled exploration candidate where hard safety permits, or explicitly mark the reason as hard safety/min-notional.

---

### Task 6: Fill-Confirmed Execution And Recovery Regression Pack

**Files:**
- Modify only if tests expose a defect: `src/panteon_v2/execution/executor.py`, `src/panteon_v2/app/v1_futures_adapter.py`, `src/panteon_v2/app/live_state.py`, `src/panteon_runtime/mexc_connector.py`, `src/panteon_runtime/bitget_connector.py`
- Test: `src/panteon_v2/tests/test_executor.py`
- Test: `src/panteon_v2/tests/test_exchange_adapters.py`
- Test: `src/panteon_v2/tests/test_live_state.py`
- Test: `src/panteon_v2/tests/test_production_hardening.py`
- Test: `src/panteon_v2/tests/test_ml_executor.py`

- [ ] **Step 1: Add or confirm tests for required contracts**

Required assertions:
- order sent then filled updates tracker;
- rejected order does not create a position;
- pending order does not create a position until fill confirmation;
- duplicate execution key blocks second send;
- close without owned position is blocked;
- restart recovers pending orders and reconciles real positions.

- [ ] **Step 2: Run the full execution pack**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_executor.py src\panteon_v2\tests\test_exchange_adapters.py src\panteon_v2\tests\test_live_state.py src\panteon_v2\tests\test_production_hardening.py src\panteon_v2\tests\test_ml_executor.py -q
```

Expected: all pass before any paper/live validation.

---

### Task 7: Activity And Edge Metrics

**Files:**
- Modify: `src/panteon_v2/analysis/walk_forward.py`
- Modify: `src/panteon_v2/app/output_writer.py`
- Modify: `src/panteon_v2/dashboards/png_renderer.py`
- Test: `src/panteon_v2/tests/test_walk_forward_report.py`
- Test: `src/panteon_v2/tests/test_png_renderer.py`

- [ ] **Step 1: Add metrics tests**

Add tests for `trades_per_day`, `signals_per_day`, `filled_signal_ratio`, `no_trade_ratio`, average holding bars/time, exposure time, realized/unrealized PnL, fees, funding, slippage, max drawdown, profit factor, expectancy, Sharpe, and Sortino.

- [ ] **Step 2: Implement metrics from actual events**

Do not infer trading activity from shadow signals. Use `SignalEmitted`, `OrderSent`, `OrderFilled`, `OrderRejected`, `PositionOpened`, and `PositionClosed` only.

- [ ] **Step 3: Update dashboards**

Dashboards must show `NO REAL ACTIVITY` when real signals/orders/fills are zero even if shadow activity is high.

---

### Task 8: Backtest And Walk-Forward Evidence

**Files:**
- Modify: `tools/run_panteon_3_compare.py`
- Create if missing: `tools/run_panteon_3_compare.py`
- Test: `src/panteon_v2/tests/test_panteon_3_compare.py`

- [ ] **Step 1: Create comparison runner**

Runner must execute the same data window for:
- old Panteon current baseline;
- Panteon 3.0 candidate flags;
- best causal component baseline;
- Legend/profile baseline;
- ML-only or Genetics-only when configured.

- [ ] **Step 2: Emit comparison table**

Required columns:

```text
name, pnl_pct, pnl_usd, max_dd_pct, profit_factor, expectancy,
trade_count, trades_per_day, no_trade_ratio, drawdown_duration,
fees_pct, funding_pct, slippage_pct, sharpe, sortino
```

- [ ] **Step 3: Run smoke windows first**

Run:

```powershell
.\.venv\Scripts\python.exe tools\run_panteon_3_compare.py --years 2024 --stride-minutes 60 --initial-capital 1000 --out Results\Panteon3Compare_smoke_2024
.\.venv\Scripts\python.exe tools\run_panteon_3_compare.py --years 2025,2026 --stride-minutes 60 --initial-capital 1000 --out Results\Panteon3Compare_smoke_2025_2026
```

- [ ] **Step 4: Promote only if stability is visible**

Promotion requires: positive post-cost expectancy, nonzero but controlled trade count, max drawdown not worse than baseline beyond tolerance, no hidden min-notional violations, and no single-symbol/single-period overfit.

---

### Task 9: Legacy Cleanup Inventory Only

**Files:**
- Create: `docs/PANTEON_3_LEGACY_CLEANUP_INVENTORY.md`

- [ ] **Step 1: Inventory candidates**

Classify files into:
- delete;
- archive to `docs/archive`;
- keep as baseline;
- move to `legacy`.

- [ ] **Step 2: Do not delete in this phase**

No removal is allowed until the inventory is reviewed after Panteon 3.0 evidence exists.

---

## Verification Command Set

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_flash_allocator.py -q
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_executor.py src\panteon_v2\tests\test_exchange_adapters.py src\panteon_v2\tests\test_live_state.py src\panteon_v2\tests\test_production_hardening.py src\panteon_v2\tests\test_ml_executor.py -q
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_png_renderer.py src\panteon_v2\tests\test_walk_forward_report.py -q
.\.venv\Scripts\python.exe tools\analyze_panteon_gate_funnel.py Results\MEXC\2026-06-16_21-29-17_v2\causal_entry_decisions.jsonl --out tmp\gate_funnel_mexc_2026-06-16.json
.\.venv\Scripts\python.exe tools\analyze_panteon_gate_funnel.py Results\BITGET\2026-06-16_21-29-25_v2\causal_entry_decisions.jsonl --out tmp\gate_funnel_bitget_2026-06-16.json
```

## Live Safety

No task in this plan sends real orders. Live or live-ready arming requires a separate user confirmation after unit tests, replay/backtest, walk-forward, paper/shadow, dashboard inspection, and final risk review.
