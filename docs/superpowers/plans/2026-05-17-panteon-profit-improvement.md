# Panteon Profit Improvement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move real Panteon allocation from the current best `+10.42%` baseline toward the non-lookahead soft allocator result `+33.62%`, while preserving drawdown near or below `10%`.

**Architecture:** Keep the current Panteon v2 pipeline, but separate three concerns that are currently coupled: decision score, actionability, and position ownership. First add diagnostics, then route real execution only to actionable candidates, then port the proven `soft_regime_top1_24_confirmed` policy into Strategist as an executable allocator, and finally add entry-causal scoring that measures opportunities from fresh opens rather than closed shadow PnL alone.

**Tech Stack:** Python 3.12, pytest, existing `src/panteon_v2` modules, local `Results/RetrodateMarket` artifacts.

---

## Evidence Summary

Latest complete benchmark artifacts:
- Baseline run: `Results/RetrodateMarket/RETRODATE_MARKET/2026-05-17_17-33-13_retrodate_market_v2`
- Hard shadow rolling run: `Results/RetrodateMarket/RETRODATE_MARKET/2026-05-17_18-17-19_retrodate_market_v2`
- Strict shadow position gate run: `Results/RetrodateMarket/RETRODATE_MARKET/2026-05-17_18-54-40_retrodate_market_v2`
- Flat handoff run: `Results/RetrodateMarket/RETRODATE_MARKET/2026-05-17_19-27-19_retrodate_market_v2`
- Fresh handoff run: `Results/RetrodateMarket/RETRODATE_MARKET/2026-05-17_19-55-00_retrodate_market_v2`

Key results:
- Best real Panteon so far: `+10.42%`, `$104.18`, DD `5.02%`, `1180` trades.
- Hard shadow rolling: `-18.35%`, DD `18.73%`, `810` trades.
- Strict shadow position gate: `-2.55%`, DD `5.20%`, `346` trades.
- Flat handoff: `-9.82%`, DD `13.32%`, `897` trades.
- Fresh handoff: `-10.25%`, DD `10.25%`, `458` trades.
- Soft allocator oracle without monthly lookahead: `soft_regime_top1_24_confirmed`, `+33.62%`, DD `10.29%`.
- Perfect monthly hindsight oracle: `+367.13%`, DD `4.18%`. This is not executable directly because it knows the best month in hindsight.

Code facts:
- `src/panteon_v2/app/main_loop.py:586` currently records `actionable fallback disabled: selected leader emitted no raw signals`.
- `src/panteon_v2/app/main_loop.py:826` already has `_find_actionable_candidate_vote`, but the normal loop does not use it for execution.
- `src/panteon_v2/app/main_loop.py:53` limits solo candidates to `3`.
- `src/panteon_v2/selection/strategist.py:1626` scores v3 shadow rolling from closed shadow PnL, not from fresh actionable opens.
- `src/panteon_v2/analysis/soft_allocator.py:133` defines the best observed no-lookahead policy shape.

Primary diagnosis:
- The allocator often selects a statistically good leader that has no real signal on the current bar.
- Shadow rolling and soft/perfect reports can credit closed PnL from positions opened before the real allocator selected that actor.
- Position handoff experiments confirm that entering into an already-running shadow book is not enough; it increases stale/late fills.

---

### Task 1: Allocation Diagnostics Export

**Files:**
- Create: `src/panteon_v2/analysis/allocation_diagnostics.py`
- Modify: `src/panteon_v2/analysis/__init__.py`
- Test: `src/panteon_v2/tests/test_allocation_diagnostics.py`

- [ ] **Step 1: Write failing tests for log-derived diagnostics**

Add `src/panteon_v2/tests/test_allocation_diagnostics.py`:

```python
from pathlib import Path

from panteon_v2.analysis.allocation_diagnostics import analyze_trading_log


def test_analyze_trading_log_counts_no_trade_and_actionability(tmp_path: Path):
    log = tmp_path / "trading.log"
    log.write_text(
        "\n".join([
            "2026-01-01 bar=1 regime=neutral leader=NoTrade selected_leader=NoTrade executed_leader=NoTrade raw_signals=0 signals=0 filled=0",
            "2026-01-01 bar=2 regime=neutral leader=Alpha selected_leader=Alpha executed_leader=Alpha raw_signals=0 signals=0 filled=0",
            "2026-01-01 bar=3 regime=neutral leader=Alpha selected_leader=Alpha executed_leader=Alpha raw_signals=2 signals=1 filled=1",
        ]),
        encoding="utf-8",
    )

    report = analyze_trading_log(log)

    assert report.bars == 3
    assert report.no_trade_bars == 1
    assert report.no_trade_share_pct == 33.333333333333336
    assert report.raw_zero_share_pct == 66.66666666666667
    assert report.by_leader["Alpha"].bars == 2
    assert report.by_leader["Alpha"].raw_zero_bars == 1
    assert report.by_leader["Alpha"].filled == 1
```

- [ ] **Step 2: Run the failing diagnostic test**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_allocation_diagnostics.py
```

Expected: `ModuleNotFoundError: No module named 'panteon_v2.analysis.allocation_diagnostics'`.

- [ ] **Step 3: Implement diagnostics**

Create `src/panteon_v2/analysis/allocation_diagnostics.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


_FIELD_RE = re.compile(r"\b([A-Za-z_]+)=([^\s]+)")


@dataclass(frozen=True)
class LeaderActionability:
    label: str
    bars: int = 0
    raw_zero_bars: int = 0
    filled_zero_bars: int = 0
    raw_signals: int = 0
    signals: int = 0
    filled: int = 0


@dataclass(frozen=True)
class AllocationDiagnostics:
    bars: int
    no_trade_bars: int
    raw_zero_bars: int
    filled_zero_bars: int
    by_leader: dict[str, LeaderActionability] = field(default_factory=dict)

    @property
    def no_trade_share_pct(self) -> float:
        return 100.0 * self.no_trade_bars / self.bars if self.bars else 0.0

    @property
    def raw_zero_share_pct(self) -> float:
        return 100.0 * self.raw_zero_bars / self.bars if self.bars else 0.0

    @property
    def filled_zero_share_pct(self) -> float:
        return 100.0 * self.filled_zero_bars / self.bars if self.bars else 0.0


def analyze_trading_log(path: str | Path) -> AllocationDiagnostics:
    leader_rows: dict[str, dict[str, int]] = {}
    bars = no_trade = raw_zero = filled_zero = 0
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if " bar=" not in line:
            continue
        fields = dict(_FIELD_RE.findall(line))
        leader = fields.get("leader")
        if not leader:
            continue
        bars += 1
        raw = _int(fields.get("raw_signals"))
        signals = _int(fields.get("signals"))
        filled = _int(fields.get("filled"))
        if leader == "NoTrade":
            no_trade += 1
        if raw == 0:
            raw_zero += 1
        if filled == 0:
            filled_zero += 1
        row = leader_rows.setdefault(
            leader,
            {"bars": 0, "raw_zero_bars": 0, "filled_zero_bars": 0, "raw_signals": 0, "signals": 0, "filled": 0},
        )
        row["bars"] += 1
        row["raw_zero_bars"] += int(raw == 0)
        row["filled_zero_bars"] += int(filled == 0)
        row["raw_signals"] += raw
        row["signals"] += signals
        row["filled"] += filled
    return AllocationDiagnostics(
        bars=bars,
        no_trade_bars=no_trade,
        raw_zero_bars=raw_zero,
        filled_zero_bars=filled_zero,
        by_leader={
            label: LeaderActionability(label=label, **values)
            for label, values in leader_rows.items()
        },
    )


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
```

- [ ] **Step 4: Export diagnostics in `analysis/__init__.py`**

Add imports:

```python
from .allocation_diagnostics import AllocationDiagnostics, LeaderActionability, analyze_trading_log
```

Add names to `__all__`:

```python
"AllocationDiagnostics",
"LeaderActionability",
"analyze_trading_log",
```

- [ ] **Step 5: Run diagnostics tests**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_allocation_diagnostics.py
```

Expected: `1 passed`.

---

### Task 2: Opt-In Actionable Fallback Execution

**Files:**
- Modify: `src/panteon_v2/app/main_loop.py`
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_app.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

- [ ] **Step 1: Write failing app test for enabled fallback**

Add a test next to `test_actionable_fallback_does_not_override_selected_leader_for_real_execution`:

```python
def test_actionable_fallback_executes_when_explicitly_enabled(self):
    pipeline = make_pipeline_with_selected_no_signal_and_actionable_candidate()
    pipeline.actionable_fallback_enabled = True

    steps = run_one_bar(pipeline)

    self.assertTrue(steps[0].fallback_used)
    self.assertEqual(steps[0].fallback_candidate, "FallbackPlayer")
    self.assertEqual(steps[0].executed_leader, "FallbackPlayer")
    self.assertGreater(steps[0].n_raw_signals, 0)
```

Use the same fake agents and pipeline pattern from the existing disabled fallback test; the selected leader must emit zero raw signals and the fallback player must emit one open signal.

- [ ] **Step 2: Run the failing app test**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_app.py::TestMainLoop::test_actionable_fallback_executes_when_explicitly_enabled
```

Expected: FAIL because `pipeline.actionable_fallback_enabled` is ignored.

- [ ] **Step 3: Wire fallback into main loop**

In `src/panteon_v2/app/main_loop.py`, replace the current skipped-only block with:

```python
    if vote_attempt["raw_signal_count"] == 0:
        fallback_enabled = bool(getattr(pipeline, "actionable_fallback_enabled", False))
        fallback_attempt = None
        if fallback_enabled:
            fallback_attempt = _find_actionable_candidate_vote(
                pipeline,
                market,
                candidates,
                decision,
                skip_label=selected_leader.label,
                signal_id_counter=signal_id_counter,
                trace_id=trace,
            )
        if fallback_attempt is not None:
            fallback_used = True
            executed_leader = fallback_attempt["leader"]
            fallback_candidate = executed_leader.label
            fallback_reason = "selected leader emitted no raw signals; actionable fallback used"
            vote_attempt = fallback_attempt
        else:
            fallback_skipped = any(
                candidate.label != selected_leader.label for candidate in candidates
            )
            if fallback_skipped:
                fallback_reason = (
                    "actionable fallback disabled: selected leader emitted no raw signals"
                    if not fallback_enabled
                    else "no actionable fallback candidate emitted raw signals"
                )
```

- [ ] **Step 4: Add retro CLI flag**

In `RetrodateMarketConfig`, add:

```python
actionable_fallback_enabled: bool = False
```

In `_build_arg_parser()` add:

```python
parser.add_argument("--enable-actionable-fallback", action="store_true")
```

In `_parse_cli_config()` pass:

```python
actionable_fallback_enabled=args.enable_actionable_fallback
```

When building the pipeline in the runner, set:

```python
setattr(pipeline, "actionable_fallback_enabled", bool(config.actionable_fallback_enabled))
```

Add the value into `run_summary.json`.

- [ ] **Step 5: Run targeted tests**

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_app.py src\panteon_v2\tests\test_retrodate_market_runner.py
```

Expected: all selected tests pass.

- [ ] **Step 6: Run 5-year A/B retrotests**

Run baseline plus fallback:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m panteon_v2.analysis.retrodate_market_runner --years 2022,2023,2024,2025,2026 --stride-minutes 60 --include-optional-agents --optional-agent-labels GeneticsGenomeEnsemble --real-promotion-gate --real-promotion-min-closed-trades 20 --real-promotion-min-pnl-pct 0 --real-promotion-loss-budget-pct -1.0 --real-promotion-probation-min-score 0.0 --use-v3-rolling-score --enable-actionable-fallback --write-every-bars 2000 --full-snapshot-every 2000 --progress-every-bars 1000
```

Acceptance target:
- `pnl_pct > 10.42`
- `max_drawdown_pct <= 10.0`
- `fallback_used > 0`
- `raw_zero_pct < 95.0`

---

### Task 3: Executable Soft Regime Top1 Score

**Files:**
- Create: `src/panteon_v2/selection/soft_shadow_score.py`
- Modify: `src/panteon_v2/selection/strategist.py`
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_soft_shadow_score.py`
- Test: `src/panteon_v2/tests/test_strategist.py`

- [ ] **Step 1: Add score state tests**

Create `src/panteon_v2/tests/test_soft_shadow_score.py`:

```python
from panteon_v2.selection.soft_shadow_score import SoftShadowScoreState


def test_soft_shadow_score_uses_previous_events_only_in_same_regime():
    state = SoftShadowScoreState(window_bars=24, min_closed_trades=2)
    state.update(bar=10, label="Alpha", regime="bullish", pnl_usd=5.0, closed_trades=1)
    state.update(bar=11, label="Alpha", regime="bullish", pnl_usd=7.0, closed_trades=1)
    state.update(bar=11, label="Beta", regime="neutral", pnl_usd=99.0, closed_trades=10)

    assert state.score(label="Alpha", regime="bullish", current_bar=12) == 12.0
    assert state.score(label="Beta", regime="bullish", current_bar=12) == 0.0


def test_soft_shadow_score_drops_old_bars_and_requires_trade_count():
    state = SoftShadowScoreState(window_bars=3, min_closed_trades=2)
    state.update(bar=1, label="Alpha", regime="bullish", pnl_usd=10.0, closed_trades=10)
    state.update(bar=4, label="Alpha", regime="bullish", pnl_usd=2.0, closed_trades=1)

    assert state.score(label="Alpha", regime="bullish", current_bar=5) == 0.0
```

- [ ] **Step 2: Implement the score state**

Create `src/panteon_v2/selection/soft_shadow_score.py`:

```python
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class _Event:
    bar: int
    pnl_usd: float
    closed_trades: int


class SoftShadowScoreState:
    def __init__(self, *, window_bars: int = 24, min_closed_trades: int = 50) -> None:
        self.window_bars = max(1, int(window_bars))
        self.min_closed_trades = max(0, int(min_closed_trades))
        self._events: dict[tuple[str, str], deque[_Event]] = defaultdict(deque)

    def update(self, *, bar: int, label: str, regime: str, pnl_usd: float, closed_trades: int) -> None:
        if not label:
            return
        key = (str(label), str(regime or "all").lower())
        self._events[key].append(_Event(int(bar), float(pnl_usd), max(0, int(closed_trades))))

    def score(self, *, label: str, regime: str, current_bar: int) -> float:
        key = (str(label), str(regime or "all").lower())
        events = self._events.get(key)
        if not events:
            return 0.0
        cutoff = int(current_bar) - self.window_bars
        while events and events[0].bar <= cutoff:
            events.popleft()
        trades = sum(event.closed_trades for event in events)
        if trades < self.min_closed_trades:
            return 0.0
        return sum(event.pnl_usd for event in events)
```

- [ ] **Step 3: Integrate into Strategist behind a new config flag**

Add to `StrategistConfig`:

```python
use_v3_soft_shadow_score: bool = False
```

In `Strategist.__init__`, create:

```python
self._soft_shadow_score = SoftShadowScoreState(
    window_bars=self._config.v3_shadow_rolling_window_bars,
    min_closed_trades=self._config.v3_shadow_rolling_min_closed_trades,
)
```

In `update_shadow_actor_updates`, call:

```python
self._soft_shadow_score.update(
    bar=bar,
    label=actor_label,
    regime=regime,
    pnl_usd=realized_pnl_usd,
    closed_trades=closed_trades,
)
```

In `_score_player_detail_v3`, before `use_v3_shadow_rolling_score`, use:

```python
if self._config.use_v3_soft_shadow_score:
    score = self._soft_shadow_score.score(
        label=player.label,
        regime=regime.label,
        current_bar=current_bar,
    )
    return CandidateScore(
        label=player.label,
        score=score,
        rank=0,
        has_data=score > 0.0,
        closed_trades=0,
        signals=0,
        execution_failures=0,
        score_source="v3_soft_shadow",
        agent_labels=agent_labels,
        memory_keys_read=memory_keys,
        session_score_delta=score,
        session_pnl_pct=score / max(1e-9, float(self._realized_initial_capital or 0.0)) * 100.0,
    )
```

- [ ] **Step 4: Add CLI flag and tests**

Add `--use-v3-soft-shadow-score` to `retrodate_market_runner.py` and pass it into `StrategistConfig`.

Run:

```powershell
$env:PYTHONPATH='src'; .\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_soft_shadow_score.py src\panteon_v2\tests\test_strategist.py src\panteon_v2\tests\test_retrodate_market_runner.py
```

Acceptance target in 5-year retest with actionable fallback:
- `pnl_pct >= 20.0`
- `max_drawdown_pct <= 12.0`
- beats `MeanRevResearch` best single shadow on real execution, or improves baseline by at least `+5%`.

---

### Task 4: Entry-Causal Shadow Episodes

**Files:**
- Modify: `src/panteon_v2/app/shadow_tournament.py`
- Create: `src/panteon_v2/analysis/entry_episode_score.py`
- Test: `src/panteon_v2/tests/test_entry_episode_score.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [ ] **Step 1: Add episode event contract**

Create test:

```python
from panteon_v2.analysis.entry_episode_score import EntryEpisode, score_recent_entry_edge


def test_score_recent_entry_edge_uses_only_fresh_opened_positions():
    events = [
        EntryEpisode(open_bar=10, close_bar=12, label="Alpha", regime="bullish", pnl_usd=5.0, closed_trades=1),
        EntryEpisode(open_bar=1, close_bar=12, label="Alpha", regime="bullish", pnl_usd=50.0, closed_trades=1),
    ]

    assert score_recent_entry_edge(events, label="Alpha", regime="bullish", current_bar=13, max_entry_age_bars=4) == 5.0
```

- [ ] **Step 2: Implement entry score helper**

Create `src/panteon_v2/analysis/entry_episode_score.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EntryEpisode:
    open_bar: int
    close_bar: int
    label: str
    regime: str
    pnl_usd: float
    closed_trades: int = 1


def score_recent_entry_edge(
    events: list[EntryEpisode],
    *,
    label: str,
    regime: str,
    current_bar: int,
    max_entry_age_bars: int,
) -> float:
    cutoff = int(current_bar) - max(0, int(max_entry_age_bars))
    return sum(
        float(event.pnl_usd)
        for event in events
        if event.label == label
        and event.regime == regime
        and int(event.open_bar) >= cutoff
        and int(event.close_bar) < int(current_bar)
    )
```

- [ ] **Step 3: Export shadow position open metadata**

Extend shadow tournament payloads with `opened_bar`, `opened_regime`, `opened_signal_action`, `opened_symbol`, and close outcome when available. Do not replace existing `shadow_player_pnl_events.jsonl`; add a new `shadow_entry_episode_events.jsonl` so existing reports remain stable.

- [ ] **Step 4: Retest whether entry-causal score predicts profit**

Run a reporting-only 5-year benchmark and compare:
- closed PnL rolling score
- entry-causal score
- soft allocator score
- perfect monthly labels

Acceptance target:
- entry-causal policy must beat strict gate `-2.55%` before being wired into real selection.
- if it does not, keep it as diagnostics only.

---

### Task 5: Real Loss Kill Defaults and Drawdown-Aware Demotion

**Files:**
- Modify: `src/panteon_v2/selection/strategist.py`
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_strategist.py`

- [ ] **Step 1: Add a test for recent real-loss demotion**

Add:

```python
def test_v3_real_loss_kill_demotes_recent_loser_even_with_positive_virtual_score(self):
    # Build a candidate with positive shadow score and realized pnl <= -1.0% after 20 trades.
    # Expected: candidate is rejected with "v3 real-loss kill".
```

Use the existing patterns around `test_v3_real_loss_kill_blocks_candidate_before_slow_promotion_gate`.

- [ ] **Step 2: Enable conservative defaults only in experiment CLI**

Do not change live defaults yet. In the 5-year experiment command, run:

```powershell
--v3-persistent-loss-kill-min-closed-trades 20 --v3-persistent-loss-kill-pnl-pct -1.0 --v3-persistent-loss-kill-win-rate-pct 42 --v3-persistent-loss-ignore-virtual-quality
```

Acceptance target:
- removes `Solo_LiveAfterShock`, `Solo_MomentumScalper`, and `Solo_VolBreakoutHunter` after their real loss budget is breached.
- improves fresh/strict gate DD without reducing profitable leader share below `25%`.

---

### Task 6: Candidate Universe Expansion

**Files:**
- Modify: `src/panteon_v2/app/main_loop.py`
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_app.py`

- [ ] **Step 1: Parameterize solo candidate limit**

Replace:

```python
_SOLO_AGENT_CANDIDATE_LIMIT = 3
```

with:

```python
_DEFAULT_SOLO_AGENT_CANDIDATE_LIMIT = 3
```

Read runtime value with:

```python
limit = int(getattr(pipeline, "solo_agent_candidate_limit", _DEFAULT_SOLO_AGENT_CANDIDATE_LIMIT) or _DEFAULT_SOLO_AGENT_CANDIDATE_LIMIT)
```

Use `limit` for `selector.select()` and `select_with_fallback()`.

- [ ] **Step 2: Add retro CLI flag**

Add:

```python
parser.add_argument("--solo-agent-candidate-limit", type=int, default=3)
```

Set:

```python
setattr(pipeline, "solo_agent_candidate_limit", int(config.solo_agent_candidate_limit))
```

- [ ] **Step 3: Test expanded candidates**

Run a matrix:

```powershell
--solo-agent-candidate-limit 3
--solo-agent-candidate-limit 6
--solo-agent-candidate-limit 10
```

Acceptance target:
- expanded limit improves PnL or profitable leader share without increasing DD above `12%`.

---

### Task 7: Evaluation Matrix and Acceptance Gate

**Files:**
- Create: `tools/run_panteon_allocator_matrix.ps1`
- Create: `src/panteon_v2/analysis/compare_retro_runs.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

- [ ] **Step 1: Add matrix runner**

Create `tools/run_panteon_allocator_matrix.ps1` with named experiments:

```powershell
$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$base = @(
  "-m", "panteon_v2.analysis.retrodate_market_runner",
  "--years", "2022,2023,2024,2025,2026",
  "--stride-minutes", "60",
  "--include-optional-agents",
  "--optional-agent-labels", "GeneticsGenomeEnsemble",
  "--real-promotion-gate",
  "--real-promotion-min-closed-trades", "20",
  "--real-promotion-min-pnl-pct", "0",
  "--real-promotion-loss-budget-pct", "-1.0",
  "--real-promotion-probation-min-score", "0.0",
  "--write-every-bars", "2000",
  "--full-snapshot-every", "2000",
  "--progress-every-bars", "1000"
)

$experiments = @(
  @("--use-v3-rolling-score"),
  @("--use-v3-rolling-score", "--enable-actionable-fallback"),
  @("--use-v3-rolling-score", "--use-v3-soft-shadow-score", "--enable-actionable-fallback"),
  @("--use-v3-rolling-score", "--use-v3-soft-shadow-score", "--enable-actionable-fallback", "--solo-agent-candidate-limit", "6"),
  @("--use-v3-rolling-score", "--use-v3-soft-shadow-score", "--enable-actionable-fallback", "--v3-persistent-loss-kill-min-closed-trades", "20", "--v3-persistent-loss-kill-pnl-pct", "-1.0", "--v3-persistent-loss-kill-win-rate-pct", "42", "--v3-persistent-loss-ignore-virtual-quality")
)

foreach ($exp in $experiments) {
  .\.venv\Scripts\python.exe @base @exp
}
```

- [ ] **Step 2: Add comparison script**

Create `src/panteon_v2/analysis/compare_retro_runs.py` that reads each `run_summary.json` and `status.json`, prints CSV columns:

```text
run,pnl_pct,pnl_usd,dd_pct,trades,notrade_pct,raw_zero_pct,fallback_used,soft_best,perfect
```

- [ ] **Step 3: Final acceptance**

The first version worth promoting must satisfy all:
- PnL beats current real baseline `+10.42%`.
- PnL is at least `50%` of soft allocator no-lookahead target, so `>= +16.8%`.
- Max drawdown `<= 12%`.
- NoTrade share below strict gate `62%`.
- Raw zero share below latest fresh handoff `97.65%`.
- Profitable real leaders share above latest `25%`.

---

## Execution Order

1. Task 1: diagnostics. Commit.
2. Task 2: actionable fallback. Retest. Commit only if A/B result is not worse than strict gate.
3. Task 3: executable soft score. Retest. Commit.
4. Task 5: real loss kill matrix. Commit only the best default-safe configuration.
5. Task 6: candidate limit matrix. Commit if it improves PnL/DD.
6. Task 4: entry-causal score. Do this after diagnostics prove current soft score still has execution gap.
7. Task 7: keep matrix runner and comparison script once two or more experiments exist.

## Risks

- `Perfect_Panteon` is a hindsight oracle. Do not optimize directly to monthly labels without walk-forward validation.
- Soft allocator reports are no-lookahead by close event order, but they still assume shadow position ownership. Treat `+33.62%` as the near-term engineering target, not a guaranteed live result.
- Enabling actionable fallback can increase churn. Use DD and real-loss kill gates from the start.

## Current Recommendation

Start with Task 2, not another handoff variant. The highest-signal defect in logs is not stale shadow position age; it is that selected leaders usually emit no current raw signals and fallback is explicitly disabled.
