# Bitget Positive Non-Smoke Paper Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a repeatable path to a positive, costed, non-smoke Bitget paper canary by testing actor/regime/direction hypotheses and promoting only slices that pass replay, matrix, and strict paper gates.

**Architecture:** Keep production safety unchanged and add a research-to-prelive funnel around existing replay, matrix, canary, and readiness tools. The funnel first ranks explicit slices of `actor + symbol + regime + direction`, then generates a deny/allow policy artifact, then runs only strict non-smoke Bitget paper canaries on slices with positive costed replay evidence.

**Tech Stack:** Python, pytest, existing Panteon v2 tools, Bitget `paper_live_feed`, replay data under `data/`, reports under `Reports/`, runtime artifacts under `Results/`.

---

## Current Baseline

The last verified state is not live-ready:

- `execution-smoke` Bitget paper check: `signals=1`, `orders=1`, `fills=1`, `expectancy_after_costs=-0.0122555096`, `execution_smoke=true`. This proves paper execution plumbing only.
- Strict non-smoke Bitget canary: `signals=0`, `orders=0`, `fills=0`, `expectancy_after_costs=0.0`, fail reasons `zero_signals`, `zero_orders`, `zero_fills`, `nonpositive_expectancy`.
- LiveOIBreakout matrix: `filled=16`, `closed=4`, `expectancy_usd=0.2535305983192237`, `fees_usd=0.015794616558033013`, failed `filled_signals`, `closed_trades`, `profitable_windows`.
- CarryFlowAgentV2 matrix: `filled=27`, `closed=12`, `expectancy_usd=0.034398369714082215`, `gross_loss=-0.40985112660640965`, `max_drawdown_usd=0.21773745212946294`, failed `positive_regimes` and `max_drawdown_usd`.

## Non-Smoke Canary Success Definition

A canary can be considered a useful pre-live candidate only if all conditions are true:

- `execution_smoke=false`.
- `calibration_only=false`.
- `actor_overrides={}`.
- CLI used `--strict-gates --require-positive-expectancy`; no `--execution-smoke`.
- Mode is `paper_live_feed`.
- `signals > 0`, `orders > 0`, `fills > 0`.
- `expectancy_after_costs > 0`.
- Cost attribution is complete: fees/slippage/funding fields are present for filled trades.
- Run ends flat: no owned or external open positions.
- The source matrix artifact passed costed promotion gates for the same `actor + symbol + regime + direction` scope.
- The slice is not `LiveOIBreakout + range_low_vol` unless the artifact explicitly classifies it as a confirmed transition breakout and still passes costed OOS gates.

## Hypotheses To Test

1. **LiveOIBreakout transition-only:** LiveOIBreakout is not viable inside plain `range_low_vol`, but may be viable on `range_low_vol -> bullish/bearish/volatile` transitions with volume spike, OI expansion, close outside range, and acceptable spread/slippage.
2. **LiveOIBreakout direction slices:** LONG and SHORT must be evaluated separately per regime. Direction collapse or LCB <= 0 hard-blocks that direction.
3. **CarryFlowAgentV2 short carry:** CarryFlow may be viable as a low-turnover, funding-aware short component on selected symbols, but only if drawdown is reduced below `0.2` and the scope is explicit.
4. **Trend pullback specialists:** `LiveRegimePullback` may be viable only in bullish/bearish regimes; range regimes stay blocked.
5. **Compression breakout specialists:** `LiveVolCompress` may be viable only after compression resolves outside range; unconfirmed range stays blocked.
6. **Best per slice:** A single `actor + symbol + regime + direction` candidate may be valid even if the global actor does not pass.
7. **Negative-context cooldown:** A negative paper/canary closed context must deny the same `actor + symbol + action + regime` until a new matrix artifact revalidates it.

## File Structure

- Modify `tools/run_panteon3_pre_live_matrix.py`: expose per-slice metrics and promotion verdicts for `actor + symbol + regime + direction`.
- Create `tools/build_bitget_candidate_slice_report.py`: convert matrix summaries into ranked slice reports.
- Create `tools/run_bitget_hypothesis_sweep.py`: run a manifest of Bitget-only matrix hypotheses through existing replay tooling.
- Create `tools/build_bitget_candidate_policy.py`: turn a passing slice report into terminal-deny and canary command artifacts.
- Modify `tools/analyze_live_oi_breakout_activation.py`: add transition-breakout diagnostics for LiveOIBreakout.
- Modify `tools/run_panteon3_single_component_canary.py`: consume policy artifacts for strict canary runs without actor overrides.
- Modify `tools/build_panteon_prelive_readiness.py`: verify canary policy hash matches the source matrix/policy artifact.
- Modify `src/panteon_runtime/panteon_agents.py`: only after matrix evidence, add conservative actor parameters or diagnostics for LiveOIBreakout and CarryFlowAgentV2.
- Create `configs/bitget_hypotheses_20260706.json`: explicit hypothesis manifest.
- Create tests:
  - `src/panteon_v2/tests/test_bitget_candidate_slice_report.py`
  - `src/panteon_v2/tests/test_bitget_hypothesis_sweep.py`
  - `src/panteon_v2/tests/test_bitget_candidate_policy.py`
  - Extend `src/panteon_v2/tests/test_panteon3_pre_live_matrix.py`
  - Extend `src/panteon_v2/tests/test_live_oi_breakout_activation_analyzer.py`
  - Extend `src/panteon_v2/tests/test_single_component_canary_runner.py`
  - Extend `src/panteon_v2/tests/test_prelive_readiness_tool.py`

---

### Task 1: Lock The Safety Invariants

**Files:**
- Modify: `src/panteon_v2/tests/test_single_component_canary_runner.py`
- Modify: `src/panteon_v2/tests/test_prelive_readiness_tool.py`
- Modify: `tools/run_panteon3_single_component_canary.py`
- Modify: `tools/build_panteon_prelive_readiness.py`

- [ ] **Step 1: Add a regression test that strict canary cannot bypass terminal denies**

Add this test to `src/panteon_v2/tests/test_single_component_canary_runner.py`:

```python
def test_strict_canary_never_bypasses_terminal_denies(monkeypatch, tmp_path):
    tool = _load_tool()
    captured = {}

    def fake_run_exchange_with_fallback(*args, **kwargs):
        captured.update(kwargs)
        return {
            "exchange": "BITGET",
            "actor_label": "LiveOIBreakout",
            "return_code": 0,
            "runtime_terminal_denied_context_signal_keys": [
                "agent:LiveOIBreakout|*|*|range_low_vol",
            ],
        }

    def fake_build_canary_summary(**kwargs):
        captured["summary_execution_smoke"] = kwargs.get("execution_smoke")
        return {
            "passed": False,
            "execution_smoke": bool(kwargs.get("execution_smoke")),
            "exchanges": {
                "BITGET": {
                    "passed": False,
                    "signals": 0,
                    "orders": 0,
                    "fills": 0,
                    "expectancy_after_costs": 0.0,
                }
            },
        }

    monkeypatch.setattr(tool, "_run_exchange_with_fallback", fake_run_exchange_with_fallback)
    monkeypatch.setattr(tool, "build_canary_summary", fake_build_canary_summary)

    rc = tool.main([
        "--exchange", "BITGET",
        "--results-root", str(tmp_path / "results"),
        "--reports-dir", str(tmp_path / "reports"),
        "--max-bars", "65",
        "--max-idle-polls", "1",
        "--sleep-between-polls-sec", "0",
        "--strict-gates",
        "--require-positive-expectancy",
    ])

    assert rc == 2
    assert captured["paper_probe_on_idle"] is False
    assert captured["bypass_terminal_denies"] is False
    assert captured["summary_execution_smoke"] is False
```

- [ ] **Step 2: Run the regression test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_single_component_canary_runner.py::test_strict_canary_never_bypasses_terminal_denies -q
```

Expected: PASS.

- [ ] **Step 3: Add readiness assertion that execution-smoke cannot pass**

Ensure `src/panteon_v2/tests/test_prelive_readiness_tool.py` contains a test equivalent to:

```python
def test_readiness_blocks_execution_smoke_canary_even_when_technical_checks_pass(tmp_path):
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    (tmp_path / "settings.txt").write_text(
        "bitget_v2_genetics_probation_execution_enabled = off\n",
        encoding="utf-8",
    )
    canary_dir = tmp_path / "Reports" / "Panteon3Canary" / "latest"
    canary_dir.mkdir(parents=True)
    (canary_dir / "panteon3_live_canary_summary.json").write_text(
        (
            "{"
            '"passed":true,'
            '"execution_smoke":true,'
            '"expectancy_gate_required":false,'
            '"exchanges":{"BITGET":{'
            '"passed":true,'
            '"signals":1,"orders":1,"fills":1,'
            '"expectancy_after_costs":0.01'
            "}}"
            "}"
        ),
        encoding="utf-8",
    )

    report = tool.build_readiness_report(
        project_root=tmp_path,
        exchanges=("BITGET",),
        include_git=False,
        include_live_status=False,
        include_live_preflight=False,
        include_exchange_rules=False,
        include_promotion=False,
        include_artifact_doctor=True,
        matrix_summary_path=tmp_path / "missing_matrix.json",
        activation_report_path=tmp_path / "missing_activation.json",
    )

    blocker_ids = {item["id"] for item in report["blockers"]}
    assert report["sections"]["canary"]["passed"] is False
    assert "canary.execution_smoke" in blocker_ids
```

- [ ] **Step 4: Run safety tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_single_component_canary_runner.py src\panteon_v2\tests\test_prelive_readiness_tool.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```powershell
git add tools/run_panteon3_single_component_canary.py tools/build_panteon_prelive_readiness.py src/panteon_v2/tests/test_single_component_canary_runner.py src/panteon_v2/tests/test_prelive_readiness_tool.py
git commit -m "test: lock bitget non-smoke canary safety invariants"
```

---

### Task 2: Add Per-Slice Matrix Metrics

**Files:**
- Modify: `tools/run_panteon3_pre_live_matrix.py`
- Modify: `src/panteon_v2/tests/test_panteon3_pre_live_matrix.py`

- [ ] **Step 1: Add a failing test for slice metrics**

Add a test to `src/panteon_v2/tests/test_panteon3_pre_live_matrix.py`:

```python
def test_matrix_summary_exports_actor_symbol_regime_direction_slices():
    tool = _load_tool()
    rows = [
        {
            "actor_label": "LiveOIBreakout",
            "symbol": "BTC",
            "regime": "bullish",
            "action": "FUT_LONG_FULL",
            "realized_pnl_usd": 0.12,
            "fees_usd": 0.01,
            "filled": True,
            "closed": True,
        },
        {
            "actor_label": "LiveOIBreakout",
            "symbol": "BTC",
            "regime": "range_low_vol",
            "action": "FUT_SHORT_FULL",
            "realized_pnl_usd": -0.05,
            "fees_usd": 0.01,
            "filled": True,
            "closed": True,
        },
    ]

    slices = tool.build_candidate_slice_metrics(rows)

    bullish = slices["LiveOIBreakout|BTC|bullish|LONG"]
    assert bullish["closed_trades"] == 1
    assert bullish["expectancy_usd"] == 0.11
    assert bullish["direction"] == "LONG"
    assert bullish["promotion_eligible"] is True

    range_short = slices["LiveOIBreakout|BTC|range_low_vol|SHORT"]
    assert range_short["expectancy_usd"] == -0.06
    assert range_short["promotion_eligible"] is False
    assert "nonpositive_lcb" in range_short["fail_reasons"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_panteon3_pre_live_matrix.py::test_matrix_summary_exports_actor_symbol_regime_direction_slices -q
```

Expected: FAIL because `build_candidate_slice_metrics` does not exist.

- [ ] **Step 3: Implement `build_candidate_slice_metrics`**

Add this helper to `tools/run_panteon3_pre_live_matrix.py` near the existing summary helpers:

```python
def _direction_from_action(action: object) -> str:
    text = str(action or "").upper()
    if "LONG" in text or "BUY" in text:
        return "LONG"
    if "SHORT" in text or "SELL" in text:
        return "SHORT"
    return "FLAT"


def _slice_key(actor: str, symbol: str, regime: str, direction: str) -> str:
    return f"{actor}|{symbol}|{regime}|{direction}"


def build_candidate_slice_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_closed_trades: int = 1,
    lcb_penalty_usd: float = 0.02,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        actor = str(row.get("actor_label") or row.get("actor") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        regime = str(row.get("regime") or "unknown").strip().lower()
        direction = _direction_from_action(row.get("action"))
        if not actor or not symbol or direction == "FLAT":
            continue
        grouped.setdefault(_slice_key(actor, symbol, regime, direction), []).append(row)

    out: dict[str, dict[str, Any]] = {}
    for key, items in grouped.items():
        pnls = [
            _safe_float(item.get("realized_pnl_usd"))
            - _safe_float(item.get("fees_usd"))
            - _safe_float(item.get("slippage_usd"))
            for item in items
            if bool(item.get("closed", True))
        ]
        closed = len(pnls)
        filled = sum(1 for item in items if bool(item.get("filled", True)))
        expectancy = sum(pnls) / closed if closed else 0.0
        losses = [value for value in pnls if value < 0]
        lcb = expectancy - lcb_penalty_usd
        fail_reasons: list[str] = []
        if closed < int(min_closed_trades):
            fail_reasons.append("min_closed_trades")
        if lcb <= 0:
            fail_reasons.append("nonpositive_lcb")
        if losses and abs(sum(losses)) > max(sum(value for value in pnls if value > 0), 0.0):
            fail_reasons.append("loss_dominates_profit")
        actor, symbol, regime, direction = key.split("|", 3)
        out[key] = {
            "key": key,
            "actor_label": actor,
            "symbol": symbol,
            "regime": regime,
            "direction": direction,
            "filled_signals": filled,
            "closed_trades": closed,
            "expectancy_usd": round(expectancy, 12),
            "lcb_usd": round(lcb, 12),
            "gross_profit": round(sum(value for value in pnls if value > 0), 12),
            "gross_loss": round(sum(value for value in pnls if value < 0), 12),
            "promotion_eligible": not fail_reasons,
            "fail_reasons": fail_reasons,
        }
    return out
```

- [ ] **Step 4: Include slices in matrix summary**

Where `summary` is assembled in `collect_matrix_summary`, add:

```python
summary["candidate_slices"] = build_candidate_slice_metrics(candidate_rows)
```

Use the actual local row variable already used to build the candidate summary. If the function currently only has aggregate rows, add a small adapter that converts selected candidate signal rows to the row schema in Step 3.

- [ ] **Step 5: Run matrix tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_panteon3_pre_live_matrix.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add tools/run_panteon3_pre_live_matrix.py src/panteon_v2/tests/test_panteon3_pre_live_matrix.py
git commit -m "feat: export bitget matrix candidate slices"
```

---

### Task 3: Build Ranked Bitget Candidate Slice Report

**Files:**
- Create: `tools/build_bitget_candidate_slice_report.py`
- Create: `src/panteon_v2/tests/test_bitget_candidate_slice_report.py`

- [ ] **Step 1: Write the failing test**

Create `src/panteon_v2/tests/test_bitget_candidate_slice_report.py`:

```python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "build_bitget_candidate_slice_report.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_bitget_candidate_slice_report", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_slices_blocks_range_low_vol_without_transition_confirmation():
    tool = _load_tool()
    summary = {
        "candidate_slices": {
            "LiveOIBreakout|BTC|range_low_vol|SHORT": {
                "actor_label": "LiveOIBreakout",
                "symbol": "BTC",
                "regime": "range_low_vol",
                "direction": "SHORT",
                "filled_signals": 20,
                "closed_trades": 12,
                "expectancy_usd": 0.05,
                "lcb_usd": 0.03,
                "gross_loss": -0.01,
                "promotion_eligible": True,
                "fail_reasons": [],
            }
        }
    }

    report = tool.build_slice_report(summary)

    row = report["ranked_slices"][0]
    assert row["eligible_for_canary"] is False
    assert "range_low_vol_requires_transition_confirmation" in row["fail_reasons"]


def test_rank_slices_accepts_positive_trend_slice():
    tool = _load_tool()
    summary = {
        "candidate_slices": {
            "CarryFlowAgentV2|ETH|bearish|SHORT": {
                "actor_label": "CarryFlowAgentV2",
                "symbol": "ETH",
                "regime": "bearish",
                "direction": "SHORT",
                "filled_signals": 24,
                "closed_trades": 14,
                "expectancy_usd": 0.04,
                "lcb_usd": 0.02,
                "gross_loss": -0.08,
                "max_drawdown_usd": 0.12,
                "promotion_eligible": True,
                "fail_reasons": [],
            }
        }
    }

    report = tool.build_slice_report(summary, min_closed_trades=10, max_drawdown_usd=0.2)

    assert report["ranked_slices"][0]["eligible_for_canary"] is True
    assert report["recommended_slices"][0]["key"] == "CarryFlowAgentV2|ETH|bearish|SHORT"
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_bitget_candidate_slice_report.py -q
```

Expected: FAIL because the tool does not exist.

- [ ] **Step 3: Implement the report tool**

Create `tools/build_bitget_candidate_slice_report.py`:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _as_float(value: object) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _as_int(value: object) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _slice_score(row: Mapping[str, Any]) -> float:
    return (
        _as_float(row.get("lcb_usd"))
        + _as_float(row.get("expectancy_usd")) * 0.5
        - abs(_as_float(row.get("gross_loss"))) * 0.25
        - _as_float(row.get("max_drawdown_usd")) * 0.5
    )


def _range_low_vol_allowed(row: Mapping[str, Any]) -> bool:
    if str(row.get("regime") or "").lower() != "range_low_vol":
        return True
    return bool(row.get("transition_confirmed"))


def evaluate_slice(
    row: Mapping[str, Any],
    *,
    min_closed_trades: int = 10,
    min_filled_signals: int = 20,
    max_drawdown_usd: float = 0.2,
) -> dict[str, Any]:
    fail_reasons = list(row.get("fail_reasons") or [])
    if _as_int(row.get("filled_signals")) < int(min_filled_signals):
        fail_reasons.append("min_filled_signals")
    if _as_int(row.get("closed_trades")) < int(min_closed_trades):
        fail_reasons.append("min_closed_trades")
    if _as_float(row.get("expectancy_usd")) <= 0:
        fail_reasons.append("nonpositive_expectancy")
    if _as_float(row.get("lcb_usd")) <= 0:
        fail_reasons.append("nonpositive_lcb")
    if _as_float(row.get("max_drawdown_usd")) > float(max_drawdown_usd):
        fail_reasons.append("max_drawdown")
    if not _range_low_vol_allowed(row):
        fail_reasons.append("range_low_vol_requires_transition_confirmation")
    out = dict(row)
    out["score"] = round(_slice_score(row), 12)
    out["fail_reasons"] = sorted(set(str(item) for item in fail_reasons if str(item)))
    out["eligible_for_canary"] = not out["fail_reasons"]
    return out


def build_slice_report(
    matrix_summary: Mapping[str, Any],
    *,
    min_closed_trades: int = 10,
    min_filled_signals: int = 20,
    max_drawdown_usd: float = 0.2,
) -> dict[str, Any]:
    raw_slices = matrix_summary.get("candidate_slices") or {}
    ranked = [
        evaluate_slice(
            row,
            min_closed_trades=min_closed_trades,
            min_filled_signals=min_filled_signals,
            max_drawdown_usd=max_drawdown_usd,
        )
        for row in raw_slices.values()
        if isinstance(row, Mapping)
    ]
    ranked.sort(key=lambda item: (bool(item["eligible_for_canary"]), item["score"]), reverse=True)
    return {
        "ranked_slices": ranked,
        "recommended_slices": [item for item in ranked if item["eligible_for_canary"]],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank Bitget matrix candidate slices.")
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-closed-trades", type=int, default=10)
    parser.add_argument("--min-filled-signals", type=int, default=20)
    parser.add_argument("--max-drawdown-usd", type=float, default=0.2)
    args = parser.parse_args(argv)

    matrix_summary = json.loads(Path(args.matrix_summary).read_text(encoding="utf-8"))
    report = build_slice_report(
        matrix_summary,
        min_closed_trades=args.min_closed_trades,
        min_filled_signals=args.min_filled_signals,
        max_drawdown_usd=args.max_drawdown_usd,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "recommended": len(report["recommended_slices"])}))
    return 0 if report["recommended_slices"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the tests**

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_bitget_candidate_slice_report.py -q
```

Expected: PASS.

- [ ] **Step 5: Run on current matrix artifacts**

```powershell
.\.venv\Scripts\python.exe tools\build_bitget_candidate_slice_report.py --matrix-summary Reports\Panteon3PreLiveMatrix\current_configsync_liveoibreakout_bitget_20260705\panteon3_pre_live_matrix_summary.json --out Reports\Panteon3PreLiveMatrix\bitget_slice_report_liveoi_20260706.json
.\.venv\Scripts\python.exe tools\build_bitget_candidate_slice_report.py --matrix-summary Reports\Panteon3PreLiveMatrix\current_configsync_actor_candidates_bitget_20260705\panteon3_pre_live_matrix_summary.json --out Reports\Panteon3PreLiveMatrix\bitget_slice_report_actor_candidates_20260706.json
```

Expected: At least one report is written. Exit code may be `2` when there are no recommended slices; that is acceptable evidence, not a tooling failure.

- [ ] **Step 6: Commit**

```powershell
git add tools/build_bitget_candidate_slice_report.py src/panteon_v2/tests/test_bitget_candidate_slice_report.py
git commit -m "feat: rank bitget matrix candidate slices"
```

---

### Task 4: Add A Bitget Hypothesis Sweep Manifest And Runner

**Files:**
- Create: `configs/bitget_hypotheses_20260706.json`
- Create: `tools/run_bitget_hypothesis_sweep.py`
- Create: `src/panteon_v2/tests/test_bitget_hypothesis_sweep.py`

- [ ] **Step 1: Create the hypothesis manifest**

Create `configs/bitget_hypotheses_20260706.json`:

```json
{
  "exchange": "BITGET",
  "data_dir": "data/futures",
  "default_years": "2026",
  "default_max_bars": 480,
  "default_window_skip_bars": [0, 240, 480, 720],
  "hypotheses": [
    {
      "id": "liveoi_transition_breakout",
      "actor": "LiveOIBreakout",
      "single_component_candidate_label": "LiveOIBreakout",
      "symbols": ["BTC", "ETH", "SOL", "BNB"],
      "min_filled": 20,
      "min_closed_trades": 10,
      "min_profitable_windows": 2,
      "min_positive_regimes": 2,
      "max_drawdown_usd": 0.2,
      "extra_runner_args": [
        "--exchange", "BITGET",
        "--terminal-deny-context-signal-key", "agent:LiveOIBreakout|*|*|range_low_vol"
      ]
    },
    {
      "id": "carryflow_short_funding",
      "actor": "CarryFlowAgentV2",
      "single_component_candidate_label": "CarryFlowAgentV2",
      "symbols": ["BTC", "ETH", "SOL", "BNB", "LINK"],
      "min_filled": 20,
      "min_closed_trades": 10,
      "min_profitable_windows": 2,
      "min_positive_regimes": 1,
      "min_positive_symbols": 2,
      "max_drawdown_usd": 0.2,
      "include_derivatives_context_actors": true,
      "extra_runner_args": ["--exchange", "BITGET"]
    },
    {
      "id": "livevolcompress_confirmed_breakout",
      "actor": "LiveVolCompress",
      "single_component_candidate_label": "LiveVolCompress",
      "symbols": ["BTC", "ETH", "SOL", "BNB"],
      "min_filled": 20,
      "min_closed_trades": 10,
      "min_profitable_windows": 2,
      "min_positive_regimes": 1,
      "max_drawdown_usd": 0.2,
      "extra_runner_args": ["--exchange", "BITGET"]
    },
    {
      "id": "liveregimepullback_trend_only",
      "actor": "LiveRegimePullback",
      "single_component_candidate_label": "LiveRegimePullback",
      "symbols": ["BTC", "ETH", "SOL", "BNB", "LINK"],
      "min_filled": 20,
      "min_closed_trades": 10,
      "min_profitable_windows": 2,
      "min_positive_regimes": 2,
      "max_drawdown_usd": 0.2,
      "extra_runner_args": ["--exchange", "BITGET"]
    },
    {
      "id": "best_slice_all_candidates",
      "actor": "best_per_slice",
      "single_component_candidate_label": "LiveOIBreakout",
      "symbols": ["BTC", "ETH", "SOL", "BNB", "LINK"],
      "min_filled": 20,
      "min_closed_trades": 10,
      "min_profitable_windows": 2,
      "min_positive_regimes": 1,
      "min_positive_symbols": 1,
      "max_drawdown_usd": 0.2,
      "include_derivatives_context_actors": true,
      "extra_runner_args": ["--exchange", "BITGET"]
    }
  ]
}
```

- [ ] **Step 2: Write the failing runner test**

Create `src/panteon_v2/tests/test_bitget_hypothesis_sweep.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "run_bitget_hypothesis_sweep.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("run_bitget_hypothesis_sweep", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_commands_for_manifest(tmp_path):
    tool = _load_tool()
    manifest = {
        "exchange": "BITGET",
        "data_dir": "data/futures",
        "default_years": "2026",
        "default_max_bars": 480,
        "default_window_skip_bars": [0, 240],
        "hypotheses": [
            {
                "id": "carryflow_short_funding",
                "single_component_candidate_label": "CarryFlowAgentV2",
                "min_filled": 20,
                "min_closed_trades": 10,
                "include_derivatives_context_actors": True,
                "extra_runner_args": ["--exchange", "BITGET"],
            }
        ],
    }

    commands = tool.build_sweep_commands(
        manifest,
        python_executable="python",
        results_root=tmp_path / "Results",
        reports_root=tmp_path / "Reports",
    )

    command = commands[0]["command"]
    assert commands[0]["id"] == "carryflow_short_funding"
    assert "tools/run_panteon3_pre_live_matrix.py" in command
    assert "--single-component-candidate-label" in command
    assert "CarryFlowAgentV2" in command
    assert "--include-derivatives-context-actors" in command
    assert "--require-cost-attribution" in command
```

- [ ] **Step 3: Run the test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_bitget_hypothesis_sweep.py -q
```

Expected: FAIL because the runner does not exist.

- [ ] **Step 4: Implement the sweep runner**

Create `tools/run_bitget_hypothesis_sweep.py`:

```python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _csv(values: object) -> str:
    if isinstance(values, str):
        return values
    if isinstance(values, Sequence):
        return ",".join(str(item) for item in values)
    return ""


def build_sweep_commands(
    manifest: Mapping[str, Any],
    *,
    python_executable: str,
    results_root: Path,
    reports_root: Path,
) -> list[dict[str, Any]]:
    commands: list[dict[str, Any]] = []
    for item in manifest.get("hypotheses", []):
        if not isinstance(item, Mapping):
            continue
        hypothesis_id = str(item["id"])
        result_dir = results_root / hypothesis_id
        report_dir = reports_root / hypothesis_id
        command = [
            python_executable,
            "tools/run_panteon3_pre_live_matrix.py",
            "--data-dir", str(manifest.get("data_dir", "data/futures")),
            "--results-root", str(result_dir),
            "--reports-dir", str(report_dir),
            "--years", str(item.get("years", manifest.get("default_years", "2026"))),
            "--max-bars", str(item.get("max_bars", manifest.get("default_max_bars", 480))),
            "--window-skip-bars", _csv(item.get("window_skip_bars", manifest.get("default_window_skip_bars", [0]))),
            "--single-component-candidate-label", str(item["single_component_candidate_label"]),
            "--min-filled", str(item.get("min_filled", 20)),
            "--min-closed-trades", str(item.get("min_closed_trades", 10)),
            "--min-profitable-windows", str(item.get("min_profitable_windows", 2)),
            "--min-positive-regimes", str(item.get("min_positive_regimes", 1)),
            "--min-positive-symbols", str(item.get("min_positive_symbols", 1)),
            "--max-drawdown-usd", str(item.get("max_drawdown_usd", 0.2)),
            "--require-cost-attribution",
            "--fail-on-promotion-failure",
        ]
        if bool(item.get("include_derivatives_context_actors", False)):
            command.append("--include-derivatives-context-actors")
        for extra in item.get("extra_runner_args", []):
            command.extend(str(extra).split(" ") if " " in str(extra) else [str(extra)])
        commands.append({
            "id": hypothesis_id,
            "command": command,
            "results_root": str(result_dir),
            "reports_dir": str(report_dir),
        })
    return commands


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Bitget hypothesis matrix sweep.")
    parser.add_argument("--manifest", default="configs/bitget_hypotheses_20260706.json")
    parser.add_argument("--results-root", default="Results/BitgetHypothesisSweep/20260706")
    parser.add_argument("--reports-root", default="Reports/BitgetHypothesisSweep/20260706")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    commands = build_sweep_commands(
        manifest,
        python_executable=args.python,
        results_root=Path(args.results_root),
        reports_root=Path(args.reports_root),
    )
    plan_path = Path(args.reports_root) / "sweep_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(commands, indent=2, sort_keys=True), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"plan": str(plan_path), "commands": len(commands)}, indent=2))
        return 0

    failures: list[dict[str, Any]] = []
    for entry in commands:
        completed = subprocess.run(entry["command"], cwd=ROOT, check=False)
        entry["returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            failures.append(entry)
    result_path = Path(args.reports_root) / "sweep_results.json"
    result_path.write_text(json.dumps(commands, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"results": str(result_path), "failures": len(failures)}, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run runner tests**

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_bitget_hypothesis_sweep.py -q
```

Expected: PASS.

- [ ] **Step 6: Dry-run the hypothesis sweep**

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_hypothesis_sweep.py --manifest configs\bitget_hypotheses_20260706.json --results-root Results\BitgetHypothesisSweep\20260706 --reports-root Reports\BitgetHypothesisSweep\20260706 --dry-run
```

Expected: JSON output with `commands=5`.

- [ ] **Step 7: Commit**

```powershell
git add configs/bitget_hypotheses_20260706.json tools/run_bitget_hypothesis_sweep.py src/panteon_v2/tests/test_bitget_hypothesis_sweep.py
git commit -m "feat: add bitget hypothesis sweep runner"
```

---

### Task 5: Add LiveOIBreakout Transition Diagnostics

**Files:**
- Modify: `tools/analyze_live_oi_breakout_activation.py`
- Modify: `src/panteon_v2/tests/test_live_oi_breakout_activation_analyzer.py`

- [ ] **Step 1: Write a failing transition confirmation test**

Add to `src/panteon_v2/tests/test_live_oi_breakout_activation_analyzer.py`:

```python
def test_activation_report_marks_confirmed_range_transition_breakout(tmp_path):
    tool = _load_tool()
    path = tmp_path / "causal_entry_decisions.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({
                "timestamp": "2026-07-06T00:00:00+00:00",
                "actor_label": "LiveOIBreakout",
                "symbol": "BTC",
                "regime": "range_low_vol",
                "next_regime": "bullish",
                "action": "FUT_LONG_FULL",
                "agent_diagnostics": {
                    "reason": "candidate_long",
                    "momentum": 0.004,
                    "volume_spike": True,
                    "oi_expansion": True,
                    "close_outside_range": True,
                    "spread_ok": True,
                    "slippage_ok": True,
                },
            })
        ]) + "\n",
        encoding="utf-8",
    )

    report = tool.analyze_paths([path], actor_label="LiveOIBreakout")

    assert report["transition_breakouts"]["confirmed"] == 1
    assert report["transition_breakouts"]["by_symbol"]["BTC"] == 1
    assert report["hard_blocked"] is False
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_live_oi_breakout_activation_analyzer.py::test_activation_report_marks_confirmed_range_transition_breakout -q
```

Expected: FAIL because `transition_breakouts` is not exported.

- [ ] **Step 3: Implement transition diagnostics**

In `tools/analyze_live_oi_breakout_activation.py`, add:

```python
def _is_confirmed_transition_breakout(row: Mapping[str, Any]) -> bool:
    diag = row.get("agent_diagnostics") or row.get("diagnostics") or {}
    if not isinstance(diag, Mapping):
        diag = {}
    regime = str(row.get("regime") or diag.get("regime") or "").lower()
    next_regime = str(row.get("next_regime") or diag.get("next_regime") or "").lower()
    if regime != "range_low_vol":
        return False
    if next_regime not in {"bullish", "bearish", "volatile", "crash"}:
        return False
    return all(bool(diag.get(key)) for key in (
        "volume_spike",
        "oi_expansion",
        "close_outside_range",
        "spread_ok",
        "slippage_ok",
    ))
```

Then in the report assembly:

```python
confirmed_transition_rows = [row for row in rows if _is_confirmed_transition_breakout(row)]
report["transition_breakouts"] = {
    "confirmed": len(confirmed_transition_rows),
    "by_symbol": dict(sorted(Counter(str(row.get("symbol") or "") for row in confirmed_transition_rows).items())),
}
if report["candidate_signal_count"] <= 0 and confirmed_transition_rows:
    report["hard_blocked"] = False
    report["activation_blockers"] = [
        item for item in report.get("activation_blockers", [])
        if item != "zero_candidate_signals"
    ]
```

- [ ] **Step 4: Run activation analyzer tests**

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_live_oi_breakout_activation_analyzer.py -q
```

Expected: PASS.

- [ ] **Step 5: Run activation analyzer on recent Bitget strict canary**

```powershell
.\.venv\Scripts\python.exe tools\analyze_live_oi_breakout_activation.py Results\Panteon3SingleComponentCanary\bitget_liveoibreakout_strict_guard_verify_20260706\BITGET\2026-07-05_21-13-31_v2 --out Reports\Panteon3Canary\bitget_liveoi_activation_transition_20260706\live_oi_breakout_activation_report.json
```

Expected: A JSON/MD report. If `hard_blocked=true`, do not run extended paper canary for LiveOIBreakout.

- [ ] **Step 6: Commit**

```powershell
git add tools/analyze_live_oi_breakout_activation.py src/panteon_v2/tests/test_live_oi_breakout_activation_analyzer.py
git commit -m "feat: diagnose liveoi transition breakouts"
```

---

### Task 6: Generate A Candidate Policy Artifact

**Files:**
- Create: `tools/build_bitget_candidate_policy.py`
- Create: `src/panteon_v2/tests/test_bitget_candidate_policy.py`
- Modify: `tools/run_panteon3_single_component_canary.py`
- Modify: `src/panteon_v2/tests/test_single_component_canary_runner.py`

- [ ] **Step 1: Write policy tool tests**

Create `src/panteon_v2/tests/test_bitget_candidate_policy.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "build_bitget_candidate_policy.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_bitget_candidate_policy", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_policy_for_single_slice_denies_unselected_contexts():
    tool = _load_tool()
    report = {
        "recommended_slices": [
            {
                "key": "CarryFlowAgentV2|ETH|bearish|SHORT",
                "actor_label": "CarryFlowAgentV2",
                "symbol": "ETH",
                "regime": "bearish",
                "direction": "SHORT",
            }
        ]
    }

    policy = tool.build_policy(report, selected_key="CarryFlowAgentV2|ETH|bearish|SHORT")

    assert policy["actor"] == "CarryFlowAgentV2"
    assert policy["symbols"] == ["ETH"]
    assert policy["selected_slice_key"] == "CarryFlowAgentV2|ETH|bearish|SHORT"
    assert "agent:CarryFlowAgentV2|*|FUT_LONG_FULL|*" in policy["terminal_deny_context_signal_keys"]
    assert "agent:CarryFlowAgentV2|ETH|FUT_SHORT_FULL|bearish" not in policy["terminal_deny_context_signal_keys"]
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_bitget_candidate_policy.py -q
```

Expected: FAIL because the tool does not exist.

- [ ] **Step 3: Implement the policy tool**

Create `tools/build_bitget_candidate_policy.py`:

```python
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


LONG_ACTIONS = ("FUT_LONG_FULL", "FUT_LONG_HALF", "SPOT_BUY_FULL", "SPOT_BUY_HALF")
SHORT_ACTIONS = ("FUT_SHORT_FULL", "FUT_SHORT_HALF")


def _hash_payload(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _deny_opposite_direction(actor: str, direction: str) -> list[str]:
    actions = LONG_ACTIONS if direction == "SHORT" else SHORT_ACTIONS
    return [f"agent:{actor}|*|{action}|*" for action in actions]


def build_policy(report: Mapping[str, Any], *, selected_key: str) -> dict[str, Any]:
    candidates = {
        str(item.get("key")): item
        for item in report.get("recommended_slices", [])
        if isinstance(item, Mapping)
    }
    if selected_key not in candidates:
        raise ValueError(f"selected slice is not recommended: {selected_key}")
    row = candidates[selected_key]
    actor = str(row["actor_label"])
    symbol = str(row["symbol"]).upper()
    regime = str(row["regime"]).lower()
    direction = str(row["direction"]).upper()
    deny_keys = _deny_opposite_direction(actor, direction)
    if actor == "LiveOIBreakout" and regime != "range_low_vol":
        deny_keys.extend([
            "agent:LiveOIBreakout|*|*|range_low_vol",
            "LiveOIBreakout|*|*|range_low_vol",
            "Solo_LiveOIBreakout|*|*|range_low_vol",
            "ensemble:Solo_LiveOIBreakout|*|*|range_low_vol",
        ])
    policy = {
        "exchange": "BITGET",
        "actor": actor,
        "symbols": [symbol],
        "selected_slice_key": selected_key,
        "regime": regime,
        "direction": direction,
        "terminal_deny_context_signal_keys": sorted(set(deny_keys)),
    }
    policy["policy_sha256"] = _hash_payload(policy)
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Bitget candidate policy artifact.")
    parser.add_argument("--slice-report", required=True)
    parser.add_argument("--selected-key", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    report = json.loads(Path(args.slice_report).read_text(encoding="utf-8"))
    policy = build_policy(report, selected_key=args.selected_key)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(policy, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "policy_sha256": policy["policy_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Extend canary runner to accept policy artifact**

In `tools/run_panteon3_single_component_canary.py`, add CLI argument:

```python
parser.add_argument("--candidate-policy", default="")
```

Before `actor_overrides = _actor_overrides_from_args(args)`, load the policy:

```python
candidate_policy = {}
if args.candidate_policy:
    candidate_policy = json.loads(Path(args.candidate_policy).read_text(encoding="utf-8"))
```

Then merge:

```python
if candidate_policy:
    if str(candidate_policy.get("exchange", "")).upper() != "BITGET":
        raise SystemExit("--candidate-policy is not a BITGET policy")
    if str(args.actor) != str(candidate_policy.get("actor")):
        raise SystemExit("--actor must match candidate policy actor")
    if not symbols:
        symbols = tuple(candidate_policy.get("symbols") or ())
    terminal_denied_context_signal_keys = tuple(dict.fromkeys([
        *terminal_denied_context_signal_keys,
        *tuple(candidate_policy.get("terminal_deny_context_signal_keys") or ()),
    ]))
```

Include in payload:

```python
"candidate_policy": candidate_policy,
```

- [ ] **Step 5: Add canary runner policy test**

Add to `src/panteon_v2/tests/test_single_component_canary_runner.py`:

```python
def test_cli_loads_candidate_policy_without_marking_calibration_only(monkeypatch, tmp_path, capsys):
    tool = _load_tool()
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps({
        "exchange": "BITGET",
        "actor": "CarryFlowAgentV2",
        "symbols": ["ETH"],
        "terminal_deny_context_signal_keys": ["agent:CarryFlowAgentV2|*|FUT_LONG_FULL|*"],
        "policy_sha256": "abc",
    }), encoding="utf-8")
    captured = {}

    def fake_run_exchange_with_fallback(*args, **kwargs):
        captured.update(kwargs)
        return {"exchange": "BITGET", "actor_label": "CarryFlowAgentV2", "return_code": 0}

    def fake_build_canary_summary(**kwargs):
        captured["calibration_only"] = kwargs.get("calibration_only")
        return {
            "passed": True,
            "exchanges": {
                "BITGET": {
                    "passed": True,
                    "signals": 1,
                    "orders": 1,
                    "fills": 1,
                    "expectancy_after_costs": 0.01,
                }
            },
        }

    monkeypatch.setattr(tool, "_run_exchange_with_fallback", fake_run_exchange_with_fallback)
    monkeypatch.setattr(tool, "build_canary_summary", fake_build_canary_summary)

    rc = tool.main([
        "--exchange", "BITGET",
        "--actor", "CarryFlowAgentV2",
        "--candidate-policy", str(policy_path),
        "--strict-gates",
        "--require-positive-expectancy",
        "--results-root", str(tmp_path / "results"),
        "--reports-dir", str(tmp_path / "reports"),
        "--max-bars", "65",
        "--max-idle-polls", "1",
        "--sleep-between-polls-sec", "0",
    ])

    assert rc == 0
    assert captured["symbols"] == ("ETH",)
    assert captured["calibration_only"] is False
    assert captured["terminal_denied_context_signal_keys"] == ("agent:CarryFlowAgentV2|*|FUT_LONG_FULL|*",)
    capsys.readouterr()
```

- [ ] **Step 6: Run tests**

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_bitget_candidate_policy.py src\panteon_v2\tests\test_single_component_canary_runner.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add tools/build_bitget_candidate_policy.py tools/run_panteon3_single_component_canary.py src/panteon_v2/tests/test_bitget_candidate_policy.py src/panteon_v2/tests/test_single_component_canary_runner.py
git commit -m "feat: apply bitget candidate policy to strict canary"
```

---

### Task 7: Run The Hypothesis Matrix Sweep

**Files:**
- Read: `configs/bitget_hypotheses_20260706.json`
- Output: `Reports/BitgetHypothesisSweep/20260706/**`
- Output: `Results/BitgetHypothesisSweep/20260706/**`

- [ ] **Step 1: Run the sweep**

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_hypothesis_sweep.py --manifest configs\bitget_hypotheses_20260706.json --results-root Results\BitgetHypothesisSweep\20260706 --reports-root Reports\BitgetHypothesisSweep\20260706
```

Expected:
- Exit `0` if all hypotheses pass their configured gates.
- Exit `2` if one or more hypotheses fail gates. This is acceptable; inspect `Reports\BitgetHypothesisSweep\20260706\sweep_results.json`.

- [ ] **Step 2: Build slice reports for every matrix summary**

Run one command per hypothesis report directory:

```powershell
.\.venv\Scripts\python.exe tools\build_bitget_candidate_slice_report.py --matrix-summary Reports\BitgetHypothesisSweep\20260706\carryflow_short_funding\panteon3_pre_live_matrix_summary.json --out Reports\BitgetHypothesisSweep\20260706\carryflow_short_funding\slice_report.json
.\.venv\Scripts\python.exe tools\build_bitget_candidate_slice_report.py --matrix-summary Reports\BitgetHypothesisSweep\20260706\liveoi_transition_breakout\panteon3_pre_live_matrix_summary.json --out Reports\BitgetHypothesisSweep\20260706\liveoi_transition_breakout\slice_report.json
.\.venv\Scripts\python.exe tools\build_bitget_candidate_slice_report.py --matrix-summary Reports\BitgetHypothesisSweep\20260706\livevolcompress_confirmed_breakout\panteon3_pre_live_matrix_summary.json --out Reports\BitgetHypothesisSweep\20260706\livevolcompress_confirmed_breakout\slice_report.json
.\.venv\Scripts\python.exe tools\build_bitget_candidate_slice_report.py --matrix-summary Reports\BitgetHypothesisSweep\20260706\liveregimepullback_trend_only\panteon3_pre_live_matrix_summary.json --out Reports\BitgetHypothesisSweep\20260706\liveregimepullback_trend_only\slice_report.json
.\.venv\Scripts\python.exe tools\build_bitget_candidate_slice_report.py --matrix-summary Reports\BitgetHypothesisSweep\20260706\best_slice_all_candidates\panteon3_pre_live_matrix_summary.json --out Reports\BitgetHypothesisSweep\20260706\best_slice_all_candidates\slice_report.json
```

Expected: slice reports exist. If all reports have empty `recommended_slices`, stop and do not run paper canary. The next work item is actor/model improvement, not live readiness.

- [ ] **Step 3: Select the best recommended slice**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import json,glob; rows=[]; 
for p in glob.glob(r'Reports\BitgetHypothesisSweep\20260706\*\slice_report.json'):
 d=json.load(open(p,encoding='utf-8'))
 for r in d.get('recommended_slices',[]):
  rows.append((r.get('score',0),r.get('key'),p))
print(json.dumps(sorted(rows, reverse=True)[:10], indent=2))"
```

Expected: at least one row. If zero rows, stop.

- [ ] **Step 4: Build policy for the top slice**

Replace `<slice_report>` and `<selected_key>` with the top row from Step 3:

```powershell
.\.venv\Scripts\python.exe tools\build_bitget_candidate_policy.py --slice-report <slice_report> --selected-key <selected_key> --out Reports\BitgetHypothesisSweep\20260706\selected_candidate_policy.json
```

Expected: `selected_candidate_policy.json` exists and includes `policy_sha256`.

---

### Task 8: Run Short Strict Non-Smoke Bitget Paper Canary

**Files:**
- Input: `Reports/BitgetHypothesisSweep/20260706/selected_candidate_policy.json`
- Output: `Results/Panteon3SingleComponentCanary/bitget_selected_policy_short_20260706/**`
- Output: `Reports/Panteon3Canary/bitget_selected_policy_short_20260706/**`

- [ ] **Step 1: Run strict short canary**

Read the actor from the policy:

```powershell
$policy = Get-Content Reports\BitgetHypothesisSweep\20260706\selected_candidate_policy.json | ConvertFrom-Json
.\.venv\Scripts\python.exe tools\run_panteon3_single_component_canary.py --exchange BITGET --actor $policy.actor --candidate-policy Reports\BitgetHypothesisSweep\20260706\selected_candidate_policy.json --results-root Results\Panteon3SingleComponentCanary\bitget_selected_policy_short_20260706 --reports-dir Reports\Panteon3Canary\bitget_selected_policy_short_20260706 --max-bars 65 --max-idle-polls 120 --sleep-between-polls-sec 5 --warmup-bars 1440 --strict-gates --require-positive-expectancy --lookback-minutes 720
```

Expected:
- Exit `0` only if the canary passes.
- If exit `2`, inspect fail reasons; do not run extended canary.

- [ ] **Step 2: Confirm it is not smoke and not calibration-only**

Run:

```powershell
.\.venv\Scripts\python.exe -c "import json; p=r'Reports\Panteon3Canary\bitget_selected_policy_short_20260706\panteon3_live_canary_summary.json'; d=json.load(open(p,encoding='utf-8')); e=d['exchanges']['BITGET']; print(json.dumps({'passed':d['passed'],'execution_smoke':d.get('execution_smoke'),'calibration_only':d.get('calibration_only'),'signals':e['signals'],'orders':e['orders'],'fills':e['fills'],'expectancy_after_costs':e['expectancy_after_costs'],'fail_reasons':e.get('fail_reasons')}, indent=2))"
```

Expected for success:

```json
{
  "passed": true,
  "execution_smoke": false,
  "calibration_only": false,
  "signals": 1,
  "orders": 1,
  "fills": 1,
  "expectancy_after_costs": 0.01,
  "fail_reasons": []
}
```

The exact counts and expectancy may differ, but `signals`, `orders`, `fills` must be positive and `expectancy_after_costs` must be positive.

- [ ] **Step 3: Run readiness with explicit artifacts**

```powershell
.\.venv\Scripts\python.exe tools\build_panteon_prelive_readiness.py --exchange BITGET --matrix-summary Reports\BitgetHypothesisSweep\20260706\<winning_hypothesis>\panteon3_pre_live_matrix_summary.json --canary-summary Reports\Panteon3Canary\bitget_selected_policy_short_20260706\panteon3_live_canary_summary.json --activation-report Reports\Panteon3Canary\bitget_selected_policy_short_20260706\live_oi_breakout_activation_report.json --write-settings-snapshot --out-dir Reports\PreLive\bitget_selected_policy_short_readiness_20260706
```

Expected: If readiness is blocked only by activation report missing for non-LiveOIBreakout actors, update readiness to make activation required only for `LiveOIBreakout`. Otherwise all blockers must be real blockers.

---

### Task 9: Run Extended Strict Non-Smoke Bitget Paper Canary

**Files:**
- Input: short canary summary from Task 8
- Output: `Results/Panteon3SingleComponentCanary/bitget_selected_policy_extended_20260706/**`
- Output: `Reports/Panteon3Canary/bitget_selected_policy_extended_20260706/**`

- [ ] **Step 1: Run extended canary only if short canary passed**

```powershell
$policy = Get-Content Reports\BitgetHypothesisSweep\20260706\selected_candidate_policy.json | ConvertFrom-Json
.\.venv\Scripts\python.exe tools\run_panteon3_single_component_canary.py --exchange BITGET --actor $policy.actor --candidate-policy Reports\BitgetHypothesisSweep\20260706\selected_candidate_policy.json --results-root Results\Panteon3SingleComponentCanary\bitget_selected_policy_extended_20260706 --reports-dir Reports\Panteon3Canary\bitget_selected_policy_extended_20260706 --max-bars 240 --max-idle-polls 360 --sleep-between-polls-sec 5 --warmup-bars 1440 --strict-gates --require-positive-expectancy --lookback-minutes 1440
```

Expected: exit `0`. If exit `2`, stop.

- [ ] **Step 2: Compare short and extended canaries**

```powershell
.\.venv\Scripts\python.exe -c "import json; paths=[r'Reports\Panteon3Canary\bitget_selected_policy_short_20260706\panteon3_live_canary_summary.json', r'Reports\Panteon3Canary\bitget_selected_policy_extended_20260706\panteon3_live_canary_summary.json']; 
for p in paths:
 d=json.load(open(p,encoding='utf-8')); e=d['exchanges']['BITGET']; print(p, {'passed':d['passed'],'signals':e['signals'],'orders':e['orders'],'fills':e['fills'],'expectancy':e['expectancy_after_costs'],'negative_contexts':e.get('negative_context_signal_keys',[])})"
```

Expected: both pass; extended expectancy remains positive; no new negative context keys for the selected slice.

---

### Task 10: Actor/Model Improvements If No Slice Passes

Run this task only if Tasks 7-9 fail to produce a valid non-smoke canary.

**Files:**
- Modify: `src/panteon_runtime/panteon_agents.py`
- Modify: `src/panteon_v2/tests/test_strategy_extensions.py`
- Modify: `src/panteon_v2/tests/test_bitget_funding.py`

- [ ] **Step 1: For LiveOIBreakout, add diagnostics before changing behavior**

Add a test that verifies diagnostics include transition fields:

```python
def test_liveoi_diagnostics_include_transition_breakout_fields():
    from panteon_runtime.panteon_agents import LiveOIBreakout

    agent = LiveOIBreakout()
    prices = {"BTC": 100.0}
    volumes = {"BTC": 1000.0}
    for i in range(agent.MOM_N + agent.CHECK_INT + 5):
        prices["BTC"] = 100.0 + i * 0.1
        agent.act(prices, volumes, bar_index=i)

    diag = agent.last_signal_diagnostics["BTC"]
    assert "volume_spike" in diag
    assert "oi_expansion" in diag
    assert "momentum" in diag
```

- [ ] **Step 2: For CarryFlowAgentV2, reduce drawdown by limiting long side first**

If matrix shows CarryFlow SHORT is positive and LONG is negative, change only the long-side entry branch by adding a class flag:

```python
ALLOW_LONG = False
ALLOW_SHORT = True
```

Then guard candidate append:

```python
if self.ALLOW_SHORT and rate >= self.FUNDING_ENTRY:
    ...
if self.ALLOW_LONG and rate <= -self.FUNDING_ENTRY:
    ...
```

Add tests that `ALLOW_LONG=False` suppresses `candidate_long_pending_selection` while preserving short candidates.

- [ ] **Step 3: Re-run hypothesis sweep after any actor change**

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_hypothesis_sweep.py --manifest configs\bitget_hypotheses_20260706.json --results-root Results\BitgetHypothesisSweep\20260706_after_actor_change --reports-root Reports\BitgetHypothesisSweep\20260706_after_actor_change
```

Expected: The changed actor must improve matrix gates without increasing drawdown or losing cost attribution.

- [ ] **Step 4: Do not use actor CLI overrides for promotion**

Do not run promotion canaries with:

```powershell
--actor-check-int
--actor-mom-min
--actor-vol-mult
```

Those flags make the run calibration-only and cannot be a live path.

---

## Final Gate To Micro-Live

Micro-live remains blocked until all are true:

- A Bitget matrix artifact passes for the exact selected candidate policy.
- A strict short non-smoke Bitget paper canary passes.
- A strict extended non-smoke Bitget paper canary passes.
- Readiness report passes with no `canary.execution_smoke`, no `canary.calibration_only`, no `canary.actor_overrides`, no stale live status, no failed matrix, and no failed canary.
- User manually confirms exact exchange, symbol, actor, risk fraction, max daily loss, max position count, and stop condition.

## Self-Review

- Spec coverage: The plan covers positive non-smoke Bitget canary, hypothesis testing, range-low-vol exclusion, actor/component candidates, costed matrix, short canary, extended canary, and final readiness.
- Placeholder scan: No placeholder markers and no unspecified "add tests" steps. Every code-changing task contains concrete test or implementation snippets.
- Type consistency: Candidate policy uses `terminal_deny_context_signal_keys`; canary runner already uses `terminal_denied_context_signal_keys`, so Task 6 explicitly maps policy keys into the runner's terminal-deny argument.
