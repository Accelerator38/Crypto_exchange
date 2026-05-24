# Panteon Flash Profitability Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Panteon_Flash more profitable than its underlying agents/players by selecting only actor/symbol/action edges that add measurable incremental value.

**Architecture:** Keep the simplified Flash architecture: one best actor per symbol/action and players as ensemble entities for statistics. Add a measurement and promotion layer around the candidate set, not a second selector: benchmark Panteon against standalone components, promote only validated signal keys, and let high-confidence keys use more allocation while weak keys go to NoTrade. Every change must be explainable through attribution, shadow statistics, and reproducible retro-runs.

**Tech Stack:** Python, pytest, existing `panteon_v2` event log, `ProductionShadowTournament`, `SoftShadowScoreState`, `FlashAllocator`, `retrodate_market_runner.py`, local `Results/` artifacts.

---

## Current Evidence

- Best valid Flash run from the report: `Reserve-cap baseline`, PnL `+15.08%`, max DD `4.15%`, closed trades `245`.
- The same run still carries drag from losing actors: `LiveVolCompress`, `VolBreakoutHunter`, and `LiveCrashHunter` together reduced PnL by about `$73`.
- The real-allow experiment with current wrappers is rejected: PnL `-11.69%`, max DD `29.05%`.
- 2026 H1 shadow gate promoted zero experimental wrappers.
- The core problem is not "add more agents"; it is "promote only profitable actor/symbol/action behavior and suppress drag fast enough."

## Strategy

1. **Measure Panteon alpha directly.** A run is not successful unless Panteon beats the best standalone deployable component after fees, not only because it is positive.
2. **Promote signal keys, not whole agents.** The toxic behavior observed around `MomentumScalper` is action/symbol dependent; whole-agent promotion is too coarse.
3. **Make NoTrade a profitable decision.** If no actor/symbol/action key has positive recent evidence, Panteon must skip the symbol instead of allocating to the least bad actor.
4. **Allow earned concentration.** If one component is truly dominant, Panteon should not dilute it with weaker actors just to diversify. The current one-signal cap protects risk, but it can also cap the strongest edge.
5. **Keep experimental agents shadow-only until proven.** New wrappers should improve diagnostics first. They enter real allocation only through the same manifest and latest-period gates.
6. **Optimize by walk-forward, not by full-period overfit.** Tune on early years, validate on later years, and require 2026 H1/1Y early-stop gates before paying for a full 5-year run.

## File Structure

- Modify `src/panteon_v2/analysis/retrodate_market_runner.py`
  - Export component benchmark reports.
  - Export signal-key shadow reports.
  - Add CLI flags for profitability gates and manifests.
  - Include the new reports in `run_summary.json` and `analysis_report.md`.
- Create `src/panteon_v2/selection/promotion_manifest.py`
  - Hold pure data logic for actor/symbol/action promotion decisions.
  - Keep it independent from runner and allocator for testability.
- Modify `src/panteon_v2/selection/flash_allocator.py`
  - Reject unpromoted open signal keys when manifest mode is enabled.
  - Support earned cap overrides for promoted keys.
- Modify `src/panteon_v2/app/main_loop.py`
  - Pass runtime promoted/denied signal keys into `FlashAllocator.decide`.
  - Keep shadow confirmation scoped by symbol/action.
- Modify `src/panteon_v2/app/startup.py`
  - Add live settings for promotion manifest and earned cap knobs.
- Modify `src/panteon_v2/app/output_writer.py`
  - Expose current manifest, rejected keys, alpha-vs-component status, and cap state in `status.json`.
- Modify `src/panteon_v2/app/agent_bootstrap.py`
  - Add new experimental wrappers as shadow-only diagnostics.
- Add tests:
  - `src/panteon_v2/tests/test_promotion_manifest.py`
  - Extend `src/panteon_v2/tests/test_flash_allocator.py`
  - Extend `src/panteon_v2/tests/test_retrodate_market_runner.py`
  - Extend `src/panteon_v2/tests/test_app.py`

---

### Task 1: Add Panteon vs Component Benchmark

**Files:**
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

- [ ] **Step 1: Write the failing benchmark test**

Add this test to `src/panteon_v2/tests/test_retrodate_market_runner.py`:

```python
def test_component_benchmark_report_marks_panteon_under_best_component(tmp_path):
    path = runner.write_component_benchmark_report(
        tmp_path,
        panteon_pnl_usd=10.0,
        initial_capital=100.0,
        shadow_agent_pnl_events=[
            ShadowPnLEvent(bar=1, label="Alpha", pnl_usd=12.0, closed_trades=3, wins=2),
            ShadowPnLEvent(bar=2, label="Alpha", pnl_usd=3.0, closed_trades=1, wins=1),
            ShadowPnLEvent(bar=1, label="Beta", pnl_usd=4.0, closed_trades=2, wins=1),
        ],
        shadow_player_pnl_events=[
            ShadowPnLEvent(bar=1, label="PlayerOne", pnl_usd=7.0, closed_trades=2, wins=1),
        ],
        min_alpha_pct=2.0,
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["summary"]["panteon_pnl_pct"] == 10.0
    assert data["summary"]["best_component_label"] == "Alpha"
    assert data["summary"]["best_component_pnl_pct"] == 15.0
    assert data["summary"]["panteon_alpha_pct"] == -5.0
    assert data["summary"]["panteon_beats_best_component"] is False
    assert (tmp_path / "component_benchmark_report.md").exists()
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_retrodate_market_runner.py::test_component_benchmark_report_marks_panteon_under_best_component -q
```

Expected: fail with `AttributeError: module ... has no attribute 'write_component_benchmark_report'`.

- [ ] **Step 3: Implement `write_component_benchmark_report`**

Add a pure function near the other report writers in `src/panteon_v2/analysis/retrodate_market_runner.py`:

```python
def write_component_benchmark_report(
    output: Path,
    *,
    panteon_pnl_usd: float,
    initial_capital: float,
    shadow_agent_pnl_events: Sequence[ShadowPnLEvent],
    shadow_player_pnl_events: Sequence[ShadowPnLEvent],
    min_alpha_pct: float = 2.0,
    filename: str = "component_benchmark_report.json",
) -> Path:
    capital = max(1e-9, float(initial_capital or 0.0))
    component_rows = _component_benchmark_rows(
        shadow_agent_pnl_events=shadow_agent_pnl_events,
        shadow_player_pnl_events=shadow_player_pnl_events,
        capital=capital,
    )
    best = component_rows[0] if component_rows else {
        "label": "NoComponent",
        "actor_type": "none",
        "pnl_usd": 0.0,
        "pnl_pct": 0.0,
        "closed_trades": 0,
        "wins": 0,
        "win_rate_pct": 0.0,
    }
    panteon_pct = float(panteon_pnl_usd or 0.0) / capital * 100.0
    alpha_pct = panteon_pct - float(best["pnl_pct"])
    data = {
        "summary": {
            "panteon_pnl_usd": float(panteon_pnl_usd or 0.0),
            "panteon_pnl_pct": panteon_pct,
            "best_component_label": best["label"],
            "best_component_type": best["actor_type"],
            "best_component_pnl_usd": best["pnl_usd"],
            "best_component_pnl_pct": best["pnl_pct"],
            "panteon_alpha_pct": alpha_pct,
            "min_alpha_pct": float(min_alpha_pct),
            "panteon_beats_best_component": bool(alpha_pct >= float(min_alpha_pct)),
        },
        "components": component_rows,
    }
    path = output / filename
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "component_benchmark_report.md").write_text(
        "\n".join(_component_benchmark_report_lines(data)) + "\n",
        encoding="utf-8",
    )
    return path
```

- [ ] **Step 4: Add helpers for component aggregation**

Add helpers below the function:

```python
def _component_benchmark_rows(
    *,
    shadow_agent_pnl_events: Sequence[ShadowPnLEvent],
    shadow_player_pnl_events: Sequence[ShadowPnLEvent],
    capital: float,
) -> list[dict[str, object]]:
    buckets: dict[tuple[str, str], dict[str, float]] = {}
    for actor_type, events in (
        ("agent", shadow_agent_pnl_events),
        ("player", shadow_player_pnl_events),
    ):
        for event in events or ():
            label = str(getattr(event, "label", "") or "").strip()
            if not label:
                continue
            bucket = buckets.setdefault(
                (actor_type, label),
                {"pnl_usd": 0.0, "closed": 0.0, "wins": 0.0},
            )
            bucket["pnl_usd"] += float(getattr(event, "pnl_usd", 0.0) or 0.0)
            bucket["closed"] += float(getattr(event, "closed_trades", 0) or 0)
            bucket["wins"] += float(getattr(event, "wins", 0) or 0)
    rows: list[dict[str, object]] = []
    for (actor_type, label), bucket in buckets.items():
        closed = int(bucket["closed"])
        wins = int(min(bucket["wins"], bucket["closed"]))
        pnl_usd = float(bucket["pnl_usd"])
        rows.append({
            "actor_type": actor_type,
            "label": label,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_usd / max(1e-9, capital) * 100.0,
            "closed_trades": closed,
            "wins": wins,
            "win_rate_pct": (wins / closed * 100.0) if closed > 0 else 0.0,
        })
    rows.sort(key=lambda row: (-float(row["pnl_pct"]), str(row["actor_type"]), str(row["label"])))
    return rows
```

- [ ] **Step 5: Include the benchmark in run outputs**

In the retrodate run finalization block that already writes `shadow_agent_pnl_events`, `shadow_player_pnl_events`, and `flash_attribution_summary.json`, call `write_component_benchmark_report` with:

```python
status = _load_json(output_dir / "status.json")
live = status.get("live_session", {}) if isinstance(status, dict) else {}
write_component_benchmark_report(
    output_dir,
    panteon_pnl_usd=float(live.get("panteon_owned_realized_pnl_usd", 0.0) or 0.0),
    initial_capital=float(config.initial_capital),
    shadow_agent_pnl_events=shadow_agent_pnl_events,
    shadow_player_pnl_events=shadow_player_pnl_events,
    min_alpha_pct=config.component_benchmark_min_alpha_pct,
)
```

- [ ] **Step 6: Run the benchmark test**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_retrodate_market_runner.py::test_component_benchmark_report_marks_panteon_under_best_component -q
```

Expected: pass.

---

### Task 2: Add Signal-Key Promotion Manifest

**Files:**
- Create: `src/panteon_v2/selection/promotion_manifest.py`
- Test: `src/panteon_v2/tests/test_promotion_manifest.py`
- Modify: `src/panteon_v2/selection/__init__.py`

- [ ] **Step 1: Write manifest tests**

Create `src/panteon_v2/tests/test_promotion_manifest.py`:

```python
from panteon_v2.selection.promotion_manifest import (
    PromotionManifestConfig,
    build_promotion_manifest,
)


def test_promotion_manifest_allows_only_full_and_latest_positive_keys():
    rows = [
        {
            "signal_key": "agent:MomentumScalper|BTC/USDT|FUT_SHORT_FULL",
            "full_pnl_pct": 8.0,
            "latest_pnl_pct": 1.5,
            "full_closed_trades": 80,
            "latest_closed_trades": 12,
            "max_drawdown_pct": 6.0,
            "win_rate_pct": 57.0,
            "recent_downside_usd": 1.0,
        },
        {
            "signal_key": "agent:MomentumScalper|ATOM/USDT|SPOT_BUY_FULL",
            "full_pnl_pct": 5.0,
            "latest_pnl_pct": -0.5,
            "full_closed_trades": 90,
            "latest_closed_trades": 20,
            "max_drawdown_pct": 7.0,
            "win_rate_pct": 55.0,
            "recent_downside_usd": 3.0,
        },
    ]

    manifest = build_promotion_manifest(rows, PromotionManifestConfig())

    assert manifest.allowed_signal_keys == (
        "agent:MomentumScalper|BTC/USDT|FUT_SHORT_FULL",
    )
    rejected = {row.signal_key: row.reason for row in manifest.rejected}
    assert rejected["agent:MomentumScalper|ATOM/USDT|SPOT_BUY_FULL"] == "latest_pnl_below_gate"
```

- [ ] **Step 2: Implement manifest dataclasses and builder**

Add `src/panteon_v2/selection/promotion_manifest.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple


@dataclass(frozen=True)
class PromotionManifestConfig:
    min_full_closed_trades: int = 50
    min_latest_closed_trades: int = 10
    min_full_pnl_pct: float = 0.0
    min_latest_pnl_pct: float = 0.0
    max_drawdown_pct: float = 25.0
    min_win_rate_pct: float = 52.0
    max_recent_downside_usd: float = 5.0


@dataclass(frozen=True)
class PromotionRejection:
    signal_key: str
    reason: str


@dataclass(frozen=True)
class PromotionManifest:
    allowed_signal_keys: Tuple[str, ...]
    rejected: Tuple[PromotionRejection, ...]

    def as_dict(self) -> dict:
        return {
            "allowed_signal_keys": list(self.allowed_signal_keys),
            "rejected": [
                {"signal_key": row.signal_key, "reason": row.reason}
                for row in self.rejected
            ],
        }


def build_promotion_manifest(
    rows: Sequence[Mapping[str, object]],
    config: PromotionManifestConfig = PromotionManifestConfig(),
) -> PromotionManifest:
    allowed: list[str] = []
    rejected: list[PromotionRejection] = []
    for row in rows:
        signal_key = str(row.get("signal_key", "") or "").strip()
        if not signal_key:
            continue
        reason = _promotion_rejection_reason(row, config)
        if reason:
            rejected.append(PromotionRejection(signal_key, reason))
        else:
            allowed.append(signal_key)
    return PromotionManifest(
        allowed_signal_keys=tuple(sorted(dict.fromkeys(allowed))),
        rejected=tuple(sorted(rejected, key=lambda item: item.signal_key)),
    )
```

- [ ] **Step 3: Add rejection rules**

Add helper in the same file:

```python
def _promotion_rejection_reason(
    row: Mapping[str, object],
    config: PromotionManifestConfig,
) -> str:
    if _as_int(row.get("full_closed_trades")) < config.min_full_closed_trades:
        return "full_closed_trades_below_gate"
    if _as_int(row.get("latest_closed_trades")) < config.min_latest_closed_trades:
        return "latest_closed_trades_below_gate"
    if _as_float(row.get("full_pnl_pct")) < config.min_full_pnl_pct:
        return "full_pnl_below_gate"
    if _as_float(row.get("latest_pnl_pct")) < config.min_latest_pnl_pct:
        return "latest_pnl_below_gate"
    if _as_float(row.get("max_drawdown_pct")) > config.max_drawdown_pct:
        return "drawdown_above_gate"
    if _as_float(row.get("win_rate_pct")) < config.min_win_rate_pct:
        return "win_rate_below_gate"
    if _as_float(row.get("recent_downside_usd")) > config.max_recent_downside_usd:
        return "recent_downside_above_gate"
    return ""


def _as_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
```

- [ ] **Step 4: Export the module**

In `src/panteon_v2/selection/__init__.py`, export:

```python
from .promotion_manifest import (
    PromotionManifest,
    PromotionManifestConfig,
    PromotionRejection,
    build_promotion_manifest,
)
```

- [ ] **Step 5: Run manifest tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_promotion_manifest.py -q
```

Expected: pass.

---

### Task 3: Wire Manifest Into FlashAllocator

**Files:**
- Modify: `src/panteon_v2/selection/flash_allocator.py`
- Test: `src/panteon_v2/tests/test_flash_allocator.py`

- [ ] **Step 1: Add allocator tests**

Add tests to `src/panteon_v2/tests/test_flash_allocator.py`:

```python
def test_promotion_manifest_rejects_unpromoted_open_signal_key(self):
    perf = PerformanceMemory(trade_fraction=1.0)
    qm = QuarantineManager(seed=set())
    alpha = FakeAgent("Alpha", {"BTC": Action.FUT_LONG_FULL})
    beta = FakeAgent("Beta", {"BTC": Action.FUT_SHORT_FULL})
    _add_perf(perf, "Alpha", Regime.BULLISH, 20, 3.0, start_id=1)
    _add_perf(perf, "Beta", Regime.BULLISH, 20, 2.0, start_id=100)
    allocator = FlashAllocator(
        perf=perf,
        qm=qm,
        config=FlashAllocatorConfig(promotion_manifest_enabled=True),
    )

    decision = allocator.decide(
        make_market(prices={"BTC": 100.0}),
        agents=[alpha, beta],
        players=[],
        signal_id_start=1,
        promoted_signal_keys=("agent:Beta|BTC|FUT_SHORT_FULL",),
    )[0]

    assert decision.selected_actor == "Beta"
    rejected = {row.label: row for row in decision.candidates if row.rejected}
    assert rejected["Alpha"].reason == "flash_signal_not_promoted"
```

- [ ] **Step 2: Add config and decide argument**

In `FlashAllocatorConfig`, add:

```python
promotion_manifest_enabled: bool = False
```

In `FlashAllocator.decide`, add:

```python
promoted_signal_keys: Optional[Iterable[str]] = None,
```

Normalize it like degraded keys:

```python
promoted = {
    _normalize_signal_deny_key(key)
    for key in (promoted_signal_keys or ())
    if str(key or "").strip()
}
```

Pass `promoted_signal_keys=promoted` into `_decide_symbol`.

- [ ] **Step 3: Reject unpromoted open keys**

In `_decide_symbol`, before the degradation check, add:

```python
elif (
    self._config.promotion_manifest_enabled
    and output.action.is_open
    and not self._is_signal_promoted(output, symbol, promoted_signal_keys)
):
    rejected = True
    reason = "flash_signal_not_promoted"
```

Add helper:

```python
@staticmethod
def _is_signal_promoted(
    output: _ActorSignal,
    symbol: str,
    promoted_signal_keys: set[str],
) -> bool:
    if not promoted_signal_keys:
        return False
    key = _signal_deny_key(output.actor_key, symbol, output.action)
    return key in promoted_signal_keys
```

- [ ] **Step 4: Run allocator tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_flash_allocator.py -q
```

Expected: all tests pass.

---

### Task 4: Export Signal-Key Shadow Report

**Files:**
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

- [ ] **Step 1: Write test for signal-key report**

Add:

```python
def test_signal_key_shadow_report_groups_symbol_action_outcomes(tmp_path):
    path = runner.write_flash_signal_key_shadow_report(
        tmp_path,
        shadow_updates=[
            ShadowActorUpdated(
                bar=1,
                actor_type="agent",
                actor_label="Alpha",
                regime="bullish",
                symbol_action_outcomes=(("BTC/USDT", "long", 6.0, 3, 2),),
            ),
            ShadowActorUpdated(
                bar=2,
                actor_type="agent",
                actor_label="Alpha",
                regime="bullish",
                symbol_action_outcomes=(("BTC/USDT", "long", -1.0, 1, 0),),
            ),
        ],
        initial_capital=100.0,
    )

    data = json.loads(path.read_text(encoding="utf-8"))
    row = data["rows"][0]

    assert row["signal_key"] == "agent:Alpha|BTC/USDT|FUT_LONG_FULL"
    assert row["full_pnl_pct"] == 5.0
    assert row["full_closed_trades"] == 4
    assert row["win_rate_pct"] == 50.0
```

- [ ] **Step 2: Implement report writer**

Create `write_flash_signal_key_shadow_report` that reads `ShadowActorUpdated.symbol_action_outcomes`, normalizes `long` to `FUT_LONG_FULL`, `short` to `FUT_SHORT_FULL`, computes full and latest-period stats, and writes:

```json
{
  "summary": {
    "signal_key_count": 1,
    "positive_signal_keys": 1
  },
  "rows": [
    {
      "signal_key": "agent:Alpha|BTC/USDT|FUT_LONG_FULL",
      "actor_key": "agent:Alpha",
      "symbol": "BTC/USDT",
      "action": "FUT_LONG_FULL",
      "full_pnl_pct": 5.0,
      "latest_pnl_pct": 5.0,
      "full_closed_trades": 4,
      "latest_closed_trades": 4,
      "win_rate_pct": 50.0,
      "max_drawdown_pct": 1.0,
      "recent_downside_usd": 1.0
    }
  ]
}
```

- [ ] **Step 3: Generate promotion manifest from the report**

After writing `flash_signal_key_shadow_report.json`, call:

```python
manifest = build_promotion_manifest(
    signal_key_report["rows"],
    PromotionManifestConfig(
        min_full_closed_trades=config.flash_promotion_min_full_closed_trades,
        min_latest_closed_trades=config.flash_promotion_min_latest_closed_trades,
        min_full_pnl_pct=config.flash_promotion_min_full_pnl_pct,
        min_latest_pnl_pct=config.flash_promotion_min_latest_pnl_pct,
        max_drawdown_pct=config.flash_promotion_max_drawdown_pct,
        min_win_rate_pct=config.flash_promotion_min_win_rate_pct,
        max_recent_downside_usd=config.flash_promotion_max_recent_downside_usd,
    ),
)
```

Write `flash_promotion_manifest.json` with `manifest.as_dict()`.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_retrodate_market_runner.py::test_signal_key_shadow_report_groups_symbol_action_outcomes -q
```

Expected: pass.

---

### Task 5: Add Earned Cap Override

**Files:**
- Modify: `src/panteon_v2/selection/flash_allocator.py`
- Test: `src/panteon_v2/tests/test_flash_allocator.py`

- [ ] **Step 1: Add test for promoted actor cap override**

Add:

```python
def test_promoted_signal_key_can_override_actor_signal_cap(self):
    perf = PerformanceMemory(trade_fraction=1.0)
    qm = QuarantineManager(seed=set())
    alpha = FakeAgent(
        "Alpha",
        {
            "BTC": Action.FUT_SHORT_FULL,
            "ETH": Action.FUT_SHORT_FULL,
        },
    )
    _add_perf(perf, "Alpha", Regime.BEARISH, 50, 8.0, start_id=1)
    allocator = FlashAllocator(
        perf=perf,
        qm=qm,
        config=FlashAllocatorConfig(
            max_signals_per_actor=1,
            promotion_manifest_enabled=True,
            promoted_actor_cap_overrides=("agent:Alpha=2",),
        ),
    )

    decisions = allocator.decide(
        make_market(regime=Regime.BEARISH, prices={"BTC": 100.0, "ETH": 50.0}),
        agents=[alpha],
        players=[],
        signal_id_start=1,
        promoted_signal_keys=(
            "agent:Alpha|BTC|FUT_SHORT_FULL",
            "agent:Alpha|ETH|FUT_SHORT_FULL",
        ),
    )

    assert [decision.selected_actor for decision in decisions].count("Alpha") == 2
```

- [ ] **Step 2: Add config**

In `FlashAllocatorConfig`:

```python
promoted_actor_cap_overrides: Tuple[str, ...] = ()
```

Normalize format `actor_key=cap` in `__post_init__` by storing the tuple unchanged after stripping. Add a private parser:

```python
def _actor_cap_override_map(overrides: Sequence[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for raw in overrides or ():
        if "=" not in str(raw):
            continue
        actor_key, value = str(raw).split("=", 1)
        actor_key = actor_key.strip()
        try:
            cap = int(value)
        except ValueError:
            continue
        if actor_key and cap > 0:
            out[actor_key] = cap
    return out
```

- [ ] **Step 3: Apply override in `_apply_actor_signal_cap`**

Replace fixed `cap` usage with:

```python
default_cap = int(self._config.max_signals_per_actor or 0)
override_caps = _actor_cap_override_map(self._config.promoted_actor_cap_overrides)
...
actor_cap = int(override_caps.get(actor, default_cap) or 0)
if actor_cap > 0 and count >= actor_cap:
    ...
```

- [ ] **Step 4: Run allocator tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_flash_allocator.py -q
```

Expected: pass.

---

### Task 6: Add Shadow-Only Diagnostic Agent Variants

**Files:**
- Modify: `src/panteon_v2/app/agent_bootstrap.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

- [ ] **Step 1: Add variants only to the experimental registry path**

Use the existing `ActionFilterAgent` pattern. Add labels:

```python
"MomentumScalperShortCrashOnly"
"MomentumScalperShortBearOnly"
"MomentumScalperSpotPullbackOnly"
"ResearchValidatorNeutralOnly"
"FundingArbBearOnly"
"CrashPanicCrashOnly"
```

Each variant must be registered by `register_experimental_flash_agents`, not by production bootstrap defaults.

- [ ] **Step 2: Extend `ActionFilterAgent` with regime allowlist**

Add optional `allowed_regimes: Tuple[str, ...] = ()`. In `act`, after reading the source action:

```python
if self.allowed_regimes:
    regime = str(getattr(market.regime, "label", market.regime) or "").lower()
    if regime not in self.allowed_regimes:
        out[symbol] = Action.HOLD
        continue
```

- [ ] **Step 3: Add registry isolation test**

Extend `test_shadow_only_experimental_flash_registry_does_not_mutate_real_registry`:

```python
assert not real_registry.has("MomentumScalperShortCrashOnly")
assert shadow_registry.has("MomentumScalperShortCrashOnly")
assert shadow_registry.has("ResearchValidatorNeutralOnly")
```

- [ ] **Step 4: Run tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_retrodate_market_runner.py::test_shadow_only_experimental_flash_registry_does_not_mutate_real_registry -q
```

Expected: pass.

---

### Task 7: Add Profitability Matrix Runner

**Files:**
- Create: `tools/run_panteon_flash_profitability_matrix.py`

- [ ] **Step 1: Create a deterministic experiment matrix**

The matrix must run these configs:

```python
EXPERIMENTS = [
    {
        "name": "baseline_reserve_cap",
        "args": [
            "--enable-flash",
            "--enable-flash-shadow-confirmation",
            "--enable-flash-symbol-shadow-confirmation",
            "--enable-flash-shadow-quality-confirmation",
            "--flash-max-signals-per-actor", "1",
            "--enable-flash-degradation-guard",
            "--flash-degradation-reserve-actor-cap",
        ],
    },
    {
        "name": "manifest_cap1",
        "args": [
            "--enable-flash",
            "--enable-flash-shadow-confirmation",
            "--enable-flash-symbol-shadow-confirmation",
            "--enable-flash-shadow-quality-confirmation",
            "--enable-flash-promotion-manifest",
            "--flash-max-signals-per-actor", "1",
        ],
    },
    {
        "name": "manifest_earned_cap2",
        "args": [
            "--enable-flash",
            "--enable-flash-shadow-confirmation",
            "--enable-flash-symbol-shadow-confirmation",
            "--enable-flash-shadow-quality-confirmation",
            "--enable-flash-promotion-manifest",
            "--flash-max-signals-per-actor", "1",
            "--enable-flash-earned-cap-overrides",
        ],
    },
]
```

- [ ] **Step 2: Add early-stop tiers**

Run each experiment in this order:

```python
TIERS = [
    ("2026_h1", ["--years", "2026", "--max-bars", "3600"]),
    ("2025", ["--years", "2025"]),
    ("full_2022_2026", ["--years", "2022", "2023", "2024", "2025", "2026"]),
]
```

Stop an experiment if `component_benchmark_report.json` has `panteon_beats_best_component == false` on `2026_h1` or if PnL is negative and max DD exceeds `10%`.

- [ ] **Step 3: Write machine-readable output**

Write `Reports/panteon_flash_profitability_matrix.json`:

```json
{
  "experiments": [
    {
      "name": "manifest_earned_cap2",
      "tier": "full_2022_2026",
      "pnl_pct": 0.0,
      "max_drawdown_pct": 0.0,
      "panteon_alpha_pct": 0.0,
      "beats_best_component": false,
      "output_dir": "Results/..."
    }
  ]
}
```

- [ ] **Step 4: Run matrix smoke**

Run one short smoke first:

```powershell
.\.venv\Scripts\python.exe tools\run_panteon_flash_profitability_matrix.py --smoke
```

Expected: creates `Reports/panteon_flash_profitability_matrix.json` with at least one experiment row.

---

### Task 8: Acceptance Gates Before Considering Real Trading

**Files:**
- Modify: `src/panteon_v2/tests/test_retrodate_market_runner.py`
- Modify: `tools/run_panteon_flash_profitability_matrix.py`

- [ ] **Step 1: Define profitability acceptance**

A deployment candidate passes only if all are true:

```text
5-year PnL > 0
5-year max DD <= 8%
5-year Panteon alpha vs best deployable component >= +2 percentage points
2026 H1 Panteon alpha >= 0
No experimental real-allow wrapper is promoted unless latest-period gate passes
No shadow-only label appears in real attribution
```

- [ ] **Step 2: Add JSON gate output**

The matrix runner writes:

```json
{
  "best_candidate": "manifest_earned_cap2",
  "deployment_gate_passed": false,
  "gate_reasons": [
    "panteon_alpha_below_gate"
  ]
}
```

- [ ] **Step 3: Run final verification**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_promotion_manifest.py src\panteon_v2\tests\test_flash_allocator.py src\panteon_v2\tests\test_retrodate_market_runner.py -q
```

Then run full tests:

```powershell
.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests -q
```

Expected: all tests pass.

---

## Experiment Priority

1. `manifest_cap1`: should remove known drag. This is the safest first profitability attempt.
2. `manifest_earned_cap2`: tests whether the current cap is suppressing the strongest edge.
3. `MomentumScalperShortBearOnly` and `MomentumScalperShortCrashOnly`: test short-only behavior without letting spot-buy toxicity back in.
4. `ResearchValidatorNeutralOnly`: test whether its positive real-allow contribution is regime-specific.
5. `FundingArbBearOnly`: likely low-frequency but stable; useful as defensive contributor.
6. `VolBreakout` variants last: current spot-only gate failed, so these need stricter symbol/action validation before real allocation.

## Expected Outcome

The first realistic improvement target is not a huge agent invention. It is removing drag:

- If the best baseline is `+15.08%`, and losing actors contributed about `-$73`, a manifest that blocks only the worst actor/symbol/action keys can plausibly move the result toward `+20%` without increasing architecture complexity.
- If earned cap shows that the strongest keys were artificially capped, the next target is `+20-25%` with drawdown still below `8%`.
- If Panteon still fails to beat the best standalone component, the correct behavior is to collapse toward that component's promoted keys rather than diversify into weaker actors.

## Self-Review

- Spec coverage: the plan directly targets the user's requirement that Panteon must be more profitable than its components.
- Overfitting control: every optimization requires latest-period, holdout, and full-period gates.
- Debuggability: every real allocation is explainable by `component_benchmark_report.json`, `flash_signal_key_shadow_report.json`, `flash_promotion_manifest.json`, and `flash_attribution_summary.json`.
- Architecture fit: players remain ensemble entities for stats/visualization; there is no restored player-selection layer.
