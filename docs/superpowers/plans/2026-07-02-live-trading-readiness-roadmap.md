# Live Trading Readiness Roadmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Довести Panteon/Genetics контур от текущего pre-live hard-block состояния до контролируемого micro-live запуска без обхода promotion/canary/matrix gates.

**Architecture:** Live запуск разрешается только через последовательность артефактов: clean settings -> replay matrix -> single-component/route proof -> paper/live-feed canary -> readiness -> micro-live. GeneticsCore остается R&D до отдельного promotion pass; основной кандидат для ближайшего цикла - LiveOIBreakout или router, но только если его actual selected/fill path дает сделки после costs.

**Tech Stack:** Python, pytest, existing Panteon v2 runners, `tools/run_panteon3_pre_live_matrix.py`, `tools/run_panteon3_single_component_canary.py`, `tools/run_panteon3_live_canary_check.py`, `tools/build_panteon_prelive_readiness.py`, MEXC/BITGET paper/live-feed and later real-order runtime.

---

## Current State, 2026-07-02

- `settings.txt` has no GeneticsCore live bypass/admission flags and must stay unchanged until explicit micro-live switch.
- Latest useful router matrix v3 passed formal promotion gates and accounting warnings were removed.
- Separate `single_component__LiveOIBreakout` matrix artifact was generated, but actual Flash-selected path had `filled_signals=0`, `closed_trades=0`, `expectancy_usd=0.0`.
- Fresh risk-1 canary showed no min-notional block, but also `zero_signals`, `zero_orders`, `zero_fills` on MEXC and BITGET.
- Readiness remains hard-blocked by dirty worktree, matrix actual-fill failure, and canary zero signal/order/fill.
- Stale live status and GeneticsCore `promotion_eligible=False` are warnings, not live blockers, because no matching live process is running and GeneticsCore is not in live path.

## File Map

- `tools/run_panteon3_pre_live_matrix.py` - replay matrix plan/summary/gates, single-component artifact support.
- `tools/run_panteon3_single_component_canary.py` - isolated paper/live-feed canary for candidate actors.
- `tools/run_panteon3_live_canary_check.py` - canary summary, zero-signal diagnostics, execution block diagnostics.
- `tools/build_panteon_prelive_readiness.py` - final readiness aggregation and live blockers.
- `src/panteon_v2/app/live_preflight.py` - live preflight gate before live runtime.
- `src/panteon_v2/analysis/retrodate_market_runner.py` - replay runner and Flash/agent execution surface.
- `src/panteon_v2/app/main_loop.py` and `src/panteon_v2/selection/*` - candidate selection, Flash active/inactive reasons, selected/fill path.
- `src/panteon_v2/tests/test_panteon3_pre_live_matrix.py` - matrix contract tests.
- `src/panteon_v2/tests/test_panteon3_live_canary_check.py` - canary summary contract tests.
- `src/panteon_v2/tests/test_live_preflight.py` - preflight gate contract tests.
- `src/panteon_v2/tests/test_retrodate_market_runner.py` - replay runner CLI/config tests.

## Phase 0: Safety Freeze And Baseline Hygiene

### Task 0.1: Freeze Live Admission

- [ ] Verify `settings.txt` has no GeneticsCore live admission or bypass keys.

Run:

```powershell
rg -n "genetics.*(bypass|admission|probation|live)" settings.txt
```

Expected: no enabled live-admission or bypass key. If matches are present, inspect values and keep them disabled.

- [ ] Verify no project live/canary/matrix Python process is running.

Run:

```powershell
Get-CimInstance Win32_Process -Filter "name = 'python.exe'" |
  Select-Object ProcessId,CommandLine
```

Expected: no command line from `E:\Crypto_exchange` except intentional short checks.

- [ ] Commit or stash the current hardening work before any live-adjacent change.

Run:

```powershell
git status --short
```

Expected: reviewed, intentional files only. Do not use `git reset --hard`.

### Task 0.2: Archive Stale Status Artifacts

- [ ] Add a non-destructive archive helper for stale/orphan live status files.

Create: `tools/archive_stale_live_status.py`

Behavior:
- accepts `--status-path` repeatedly;
- writes copied artifacts under `Reports/PreLive/stale_status_archive/<timestamp>/`;
- writes `archive_manifest.json`;
- never deletes original files unless `--move` is explicitly passed.

- [ ] Add tests.

Test: `src/panteon_v2/tests/test_archive_stale_live_status.py`

Required assertions:
- copy mode preserves original file;
- move mode removes only the exact provided file;
- manifest includes original path, archived path, sha256, and timestamp.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_archive_stale_live_status.py -q
```

Expected: all tests pass.

## Phase 1: Fix The Model/Selection Mismatch

### Task 1.1: Explain Why LiveOIBreakout Is Profitable Standalone But Inactive In Flash

- [ ] Add a diagnostic report that compares standalone component rows to Flash candidate rows.

Modify: `tools/run_panteon3_pre_live_matrix.py`

Add a summary block for single-component variants:
- `standalone_component_label`;
- `standalone_component_pnl_usd`;
- `standalone_component_closed_trades`;
- `flash_selected_signals`;
- `flash_inactive_rejections`;
- `first_inactive_examples`;
- `activation_gap_reason`.

- [ ] Add a regression test.

Test: `src/panteon_v2/tests/test_panteon3_pre_live_matrix.py`

Test name:

```python
def test_single_component_summary_reports_activation_gap_when_flash_selected_zero():
    ...
```

Expected: when component benchmark has `LiveOIBreakout` trades but `flash_attribution_summary.selected_signals == 0`, summary includes `activation_gap_reason == "standalone_component_not_flash_active"`.

- [ ] Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_panteon3_pre_live_matrix.py -q
```

Expected: pass.

### Task 1.2: Add Replay-Only Single Component Execution Candidate

- [ ] Add a replay-only config/CLI option to `src/panteon_v2/analysis/retrodate_market_runner.py`.

Name:
- config field: `single_component_execution_label: str = ""`
- CLI flag: `--single-component-execution-label LiveOIBreakout`

Contract:
- only works in retrodate runner;
- must not alter `settings.txt`;
- must not enable real orders;
- creates a candidate path where the specified agent's actual signal can become selected and flow through existing risk/execution attribution;
- output actor key must be explicit, e.g. `single_component:LiveOIBreakout`, so readiness can distinguish it from router.

- [ ] Add config parser tests.

Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

Test name:

```python
def test_cli_config_accepts_single_component_execution_label():
    config = runner._parse_cli_config([
        "--years", "2026",
        "--enable-flash",
        "--single-component-execution-label", "LiveOIBreakout",
    ])
    assert config.single_component_execution_label == "LiveOIBreakout"
```

- [ ] Add behavior test using a fake agent or fixture where the single component emits one signal.

Expected:
- `flash_attribution_summary.selected_signals >= 1`;
- `filled_signals >= 1` in replay when risk limits allow;
- actor label is traceable to `LiveOIBreakout`.

- [ ] Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_retrodate_market_runner.py -q
```

Expected: pass.

### Task 1.3: Calibrate LiveOIBreakout Thresholds Per Symbol And Regime

- [ ] Build an offline calibration report from historical replay decisions.

Create: `tools/calibrate_live_oibreakout_thresholds.py`

Inputs:
- `--results-root Results/Panteon3PreLiveMatrix/<latest>`;
- `--label LiveOIBreakout`;
- `--out Reports/PreLive/liveoibreakout_calibration_<timestamp>.json`.

Metrics:
- signal count by symbol/regime/action;
- closed trades by symbol/regime/action;
- net PnL after fee/slippage;
- per-symbol lower confidence bound;
- first rejection reasons: `inactive`, `score_below_threshold`, `check_interval_wait`, `per-symbol actor selection`.

- [ ] Add tests for parser and aggregation.

Test: `src/panteon_v2/tests/test_live_oibreakout_calibration.py`

Required cases:
- positive symbol/regime bucket is retained;
- negative or zero LCB bucket is marked blocked;
- empty bucket is marked insufficient_evidence.

- [ ] Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_live_oibreakout_calibration.py -q
```

Expected: pass.

## Phase 2: Replay Matrix Acceptance

### Task 2.1: Run Router And Single Component Matrices With Real Costs

- [ ] Run router matrix with no empty `skip_480` window.

```powershell
.\.venv\Scripts\python.exe tools\run_panteon3_pre_live_matrix.py `
  --years 2024,2025,2026 `
  --max-bars 240 `
  --window-skip-bars 0 `
  --window-skip-bars 240 `
  --results-root Results\Panteon3PreLiveMatrix\router_costed_<timestamp> `
  --reports-dir Reports\Panteon3PreLiveMatrix\router_costed_<timestamp> `
  --stop-on-failure
```

Expected:
- `promotion_verdict.passed == true`;
- `accounting_warnings == []`;
- `gross_loss` is non-zero when losing trades exist;
- costs are explicit or attributed.

- [ ] Run single-component execution matrix after Task 1.2.

```powershell
.\.venv\Scripts\python.exe tools\run_panteon3_pre_live_matrix.py `
  --years 2024,2025,2026 `
  --max-bars 240 `
  --window-skip-bars 0 `
  --window-skip-bars 240 `
  --single-component-candidate-label LiveOIBreakout `
  --candidate-variant single_component__LiveOIBreakout `
  --results-root Results\Panteon3PreLiveMatrix\single_liveoibreakout_costed_<timestamp> `
  --reports-dir Reports\Panteon3PreLiveMatrix\single_liveoibreakout_costed_<timestamp> `
  --stop-on-failure
```

Expected:
- `filled_signals >= 20`;
- `closed_trades >= 10`;
- `expectancy_usd > 0`;
- `profit_factor_reliable == true`;
- `panteon_beats_best_component` is not used to unblock router if the candidate is single-component.

### Task 2.2: Tighten Promotion Gates Beyond Minimum Pass

- [ ] Add optional stricter gates to `tools/run_panteon3_pre_live_matrix.py`.

Flags:
- `--min-profit-factor 1.2`;
- `--min-positive-symbols 3`;
- `--max-drawdown-usd <value>`;
- `--require-cost-attribution`.

- [ ] Add tests proving each gate fails independently.

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_panteon3_pre_live_matrix.py -q
```

Expected: pass.

## Phase 3: Canary Acceptance

### Task 3.1: Run Short Signal-Path Canary

Run after matrix pass:

```powershell
.\.venv\Scripts\python.exe tools\run_panteon3_single_component_canary.py `
  --exchange MEXC `
  --exchange BITGET `
  --actor LiveOIBreakout `
  --results-root Results\Panteon3SingleComponentCanary_isolated\liveoibreakout_signalpath_<timestamp> `
  --reports-dir Reports\Panteon3Canary\liveoibreakout_signalpath_<timestamp> `
  --symbols BTC,ETH,SOL,BNB,XRP,DOGE,ADA,LINK `
  --max-bars 180 `
  --max-idle-polls 240 `
  --sleep-between-polls-sec 1 `
  --warmup-bars 1440 `
  --lookback-minutes 0
```

Expected:
- no `min_notional_blocked`;
- if no signals, zero-signal diagnostics identify first rejection reason;
- if signals occur, at least one signal reaches order/fill in paper runtime.

### Task 3.2: Run Extended Paper/Live-Feed Canary

Run only if Task 3.1 produces at least one signal or if zero-signal reason was fixed.

Acceptance gates:
- both exchanges complete;
- `signals > 0` on at least one exchange;
- `orders == fills` for paper fills or every non-fill has explicit reason;
- `reconcile_ok == true`;
- `owned_open_position_count == 0` at end;
- no external open positions;
- `expectancy_after_costs > 0` or a written exception explaining why expectancy cannot be evaluated yet.

## Phase 4: Readiness Gate

### Task 4.1: Build Final Pre-Live Readiness

Run:

```powershell
.\.venv\Scripts\python.exe tools\build_panteon_prelive_readiness.py `
  --exchange MEXC `
  --exchange BITGET `
  --exchange-rules Reports\PreLive\exchange_futures_rules_latest.json `
  --promotion-selection Reports\PreLive\genetics_reselection_20260701_175339\selection_router_fitness_v4.json `
  --write-settings-snapshot `
  --out-dir Reports\PreLive\ready_candidate_<timestamp>
```

Expected:
- `passed == true`;
- no `matrix_failed`;
- no `canary_zero_signals`;
- no `canary_zero_orders`;
- no `canary_zero_fills`;
- no `matrix_not_beating_best_component` unless candidate is router and intentionally accepted by a documented risk-adjusted gate;
- genetics warning remains acceptable only if GeneticsCore is not in live path.

## Phase 5: Micro-Live Launch

### Task 5.1: Prepare Micro-Live Settings Patch

Create a patch file, not immediate edit:

Create: `Reports/PreLive/micro_live_settings_patch_<timestamp>.md`

Required settings:
- one exchange only for first launch;
- one or two symbols only;
- real orders enabled only for the selected promoted candidate;
- GeneticsCore disabled;
- max open positions: 1;
- max new opens per bar: 1;
- notional capped to exchange minimum plus a small buffer;
- daily loss stop enabled;
- orphan position reconciliation enabled;
- kill switch documented.

### Task 5.2: Start Micro-Live Only After Manual Approval

Manual prerequisites:
- readiness `passed == true`;
- fresh canary generated within 24 hours;
- settings patch reviewed;
- exchange account contains only the intended test capital;
- user explicitly approves the exact exchange, symbols, and notional cap.

Initial live limits:
- use the smallest exchange-valid notional;
- one exchange;
- one candidate;
- one open position max;
- stop after first fill if reconciliation or attribution is unclear;
- stop after any unexpected real position.

### Task 5.3: Monitor And Stop Rules

Stop immediately if any condition occurs:
- order placed for unexpected symbol;
- notional exceeds configured cap;
- rejected order without classified reason;
- open position is not owned by the strategy;
- live status stale while process is running;
- PnL drawdown exceeds daily stop;
- fee/slippage exceeds replay assumption materially;
- event log misses `SignalEmitted -> OrderSubmitted/ExecutionAttributed -> Fill/Reject`.

## Phase 6: Post-Live Expansion

- [ ] After 20 closed micro-live trades, build a live attribution report.
- [ ] Compare live fee/slippage to replay assumptions.
- [ ] Re-run matrix with measured costs.
- [ ] Increase symbols only if live expectancy after costs is positive and reject/block rates are stable.
- [ ] Add second exchange only after first exchange has clean reconciliation and no orphan positions.
- [ ] Keep GeneticsCore in R&D until it passes validation + OOS + sanity + per-symbol LCB + baseline comparison.

## Go/No-Go Summary

No live start if any of these is true:
- dirty worktree contains unreviewed code;
- `settings.txt` has GeneticsCore bypass/admission enabled;
- matrix candidate has zero actual selected/fill path;
- canary has zero orders/fills without a documented and fixed cause;
- stale/orphan status is not understood;
- readiness report is red;
- user has not approved exact micro-live parameters.

Live start is allowed only for micro-live after all gates are green and the settings patch is reviewed.
