# Panteon Live Doctor And Metrics Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** reduce pre-live operator error by consolidating live status, settings safety, matrix, canary, activation, and genetics promotion evidence into one readiness report, then use that report to drive the next metrics work before any real trading.

**Architecture:** Keep existing hard gates intact and simplify their use through a single readiness/doctor command. The first implementation layer extends `tools/build_panteon_prelive_readiness.py` with artifact sections for matrix, canary, and activation reports, while existing `src/panteon_v2/app/live_preflight.py` remains the final executable live block.

**Tech Stack:** Python 3.12, pytest, existing Panteon v2 tools and JSON artifacts.

---

### Task 1: Consolidated Artifact Doctor

**Files:**
- Modify: `tools/build_panteon_prelive_readiness.py`
- Modify: `src/panteon_v2/tests/test_prelive_readiness_tool.py`

- [ ] **Step 1: Write failing tests**

Add tests that call `build_readiness_report(..., include_live_status=False, include_live_preflight=False, include_exchange_rules=False, include_promotion=False)` against temporary JSON artifacts:

```python
def test_readiness_blocks_failed_matrix_canary_and_activation(tmp_path):
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    (tmp_path / "settings.txt").write_text("mexc_v2_genetics_probation_execution_enabled = off\n")
    matrix_dir = tmp_path / "Reports" / "Panteon3PreLiveMatrix" / "latest"
    canary_dir = tmp_path / "Reports" / "Panteon3Canary" / "latest"
    matrix_dir.mkdir(parents=True)
    canary_dir.mkdir(parents=True)
    (matrix_dir / "panteon3_pre_live_matrix_summary.json").write_text(
        '{"promotion_verdict":{"passed":false,"fail_reasons":["min_filled"]},"candidate":{"filled_signals":7}}'
    )
    (canary_dir / "panteon3_live_canary_summary.json").write_text(
        '{"passed":false,"calibration_only":true,"actor_overrides":{"MOM_MIN":0.0015},'
        '"exchanges":{"BITGET":{"passed":false,"fail_reasons":["nonpositive_expectancy"],'
        '"signals":14,"orders":4,"fills":4,"expectancy_after_costs":-0.01}}}'
    )
    (canary_dir / "live_oi_breakout_activation_report.json").write_text(
        '{"hard_blocked":true,"activation_blockers":["zero_candidate_signals"],"candidate_signal_count":0}'
    )

    report = tool.build_readiness_report(
        project_root=tmp_path,
        exchanges=("BITGET",),
        include_git=False,
        include_live_status=False,
        include_live_preflight=False,
        include_exchange_rules=False,
        include_promotion=False,
    )

    ids = {item["id"] for item in report["blockers"]}
    assert "matrix.failed" in ids
    assert "canary.failed" in ids
    assert "canary.calibration_only" in ids
    assert "canary.actor_overrides" in ids
    assert "activation.hard_blocked" in ids
```

- [ ] **Step 2: Verify red**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_prelive_readiness_tool.py -q
```

Expected: FAIL because the readiness report has no `matrix`, `canary`, or `activation` sections.

- [ ] **Step 3: Implement artifact discovery and checks**

Add latest-file discovery for:

```text
Reports/Panteon3PreLiveMatrix/**/panteon3_pre_live_matrix_summary.json
Reports/Panteon3Canary/**/panteon3_live_canary_summary.json
Reports/Panteon3Canary/**/live_oi_breakout_activation_report.json
```

Add blockers for failed matrix verdict, failed canary, canary calibration-only, canary actor overrides, nonpositive expectancy, zero signals/orders/fills, and activation `hard_blocked=true`.

- [ ] **Step 4: Verify green**

Run the same pytest command and expect PASS.

### Task 2: Operator Markdown

**Files:**
- Modify: `tools/build_panteon_prelive_readiness.py`
- Modify: `src/panteon_v2/tests/test_prelive_readiness_tool.py`

- [ ] **Step 1: Write failing markdown test**

Assert rendered markdown lists artifact paths and the top blocker IDs so the operator does not need to manually inspect each JSON.

- [ ] **Step 2: Implement concise section rendering**

Render matrix, canary, activation, promotion, and live status verdicts with source paths.

- [ ] **Step 3: Verify**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_prelive_readiness_tool.py -q
```

### Task 3: Current Workspace Readiness Run

**Files:**
- Generate: `Reports/PreLive/prelive_readiness_*/prelive_readiness_report.json`
- Generate: `Reports/PreLive/prelive_readiness_*/prelive_readiness_report.md`

- [ ] **Step 1: Run doctor without real orders**

```powershell
.\.venv\Scripts\python.exe tools\build_panteon_prelive_readiness.py --exchange MEXC --exchange BITGET --write-settings-snapshot
```

Expected: non-zero exit while blockers remain.

- [ ] **Step 2: Summarize blockers**

Report stale/orphan live status, failed matrix, failed canary, calibration-only overrides, negative expectancy, activation state, and genetics promotion status.

### Task 4: Metrics Work After Doctor

**Files:**
- Modify only after evidence points to a narrow cause.

- [ ] **Step 1: If no-probe canary still has zero signals**

Use `tools/analyze_live_oi_breakout_activation.py` to separate warmup/check-interval failures from real filter failures.

- [ ] **Step 2: If calibration canary has negative expectancy**

Attribute losses by `actor|symbol|action|regime` and only test policy changes that remove consistently negative contexts from paper/replay. Current evidence points to `LiveOIBreakout|BNB/BTC/SOL|FUT_SHORT_HALF|range_low_vol`.

- [ ] **Step 3: If genetics remains not promotion eligible**

Keep GeneticsCore in R&D. Do not add it to live allowlists. Continue improving turnover penalties, candidate-vs-baseline hard-fail, per-regime evolution, per-symbol LCB, and NaN handling in separate TDD slices.

### Task 5: Live Launch Gate

**Files:**
- No code unless all reports pass.

- [ ] **Step 1: Paper gates**

Require passing strict matrix, no calibration-only flags, positive expectancy after fees/slippage, non-zero orders/fills, no activation hard-blocks, clean settings snapshot, and no stale live statuses.

- [ ] **Step 2: Manual approval**

Micro-live is blocked until the user confirms exact exchange, symbols, max notional, leverage, kill-switch thresholds, and runtime window.
