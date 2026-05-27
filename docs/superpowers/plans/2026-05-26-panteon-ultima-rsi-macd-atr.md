# Panteon Ultima RSI MACD ATR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add RSI, MACD, and ATR to `Panteon_Ultima` as auditable market context, not as standalone buy/sell magic.

**Architecture:** Compute normalized OHLC technical indicators before strategy decisions, attach them to `MarketSnapshot`, and let `FlashAllocator` consume them through opt-in diagnostics, score overlays, hard gates, and ATR-based sizing. Keep all indicator math pure/tested and require retrodate proof before any live profile enables trading impact.

**Tech Stack:** Python dataclasses, pytest, existing `panteon_v2` domain snapshots, Flash allocator, Retrodate market runner, candidate/audit reports.

---

## Trading Thesis

RSI, MACD, and ATR should be used by role:

- Best trader: RSI is entry timing, MACD is trend/impulse confirmation, ATR is position risk and volatility regime.
- Senior analyst: every rule must be measurable by signal key, symbol, regime, and out-of-sample window.
- Senior programmer: indicators are data features; allocation, risk, and execution remain separate components.

Initial production stance:

- Default behavior is diagnostic only.
- Score overlay is opt-in and bounded.
- Hard gate is opt-in and disabled until retrodate evidence beats current `Panteon_Ultima`.
- ATR sizing is safer than direction gating and can be evaluated earlier.

## File Structure

- Create `src/panteon_v2/analysis/technical_indicators.py`
  - Pure technical indicator state and batch/update helpers.
  - Owns RSI(14), MACD(12,26,9), ATR(14), percent normalization.
- Modify `src/panteon_v2/domain/types.py`
  - Add immutable `TechnicalIndicators`.
  - Add `technicals_by_symbol` to `MarketSnapshot`.
- Modify `src/panteon_v2/analysis/retrodate_market_runner.py`
  - Preserve OHLC rows, update indicator state per closed candle, populate `MarketSnapshot.technicals_by_symbol`.
  - Add config/CLI flags for Flash technical overlay.
- Modify `src/panteon_v2/selection/flash_allocator.py`
  - Add opt-in technical overlay config.
  - Add candidate audit fields.
  - Apply score adjustment, optional hard gate, and optional ATR risk sizing.
- Modify `src/panteon_v2/dashboards/png_renderer.py`
  - Aggregate technical alignment counts in Flash candidate rows.
- Modify `docs/PANTEON_FLASH_MECHANISMS.md`
  - Document technical overlay semantics and safe rollout.
- Test `src/panteon_v2/tests/test_technical_indicators.py`
- Test `src/panteon_v2/tests/test_retrodate_market_runner.py`
- Test `src/panteon_v2/tests/test_flash_allocator.py`

---

### Task 1: Add Pure Technical Indicator Engine

**Files:**
- Create: `src/panteon_v2/analysis/technical_indicators.py`
- Modify: `src/panteon_v2/domain/types.py`
- Test: `src/panteon_v2/tests/test_technical_indicators.py`

- [ ] **Step 1: Write failing tests for RSI, MACD, ATR**

Add `src/panteon_v2/tests/test_technical_indicators.py`:

```python
from __future__ import annotations

import pytest

from panteon_v2.analysis.technical_indicators import (
    TechnicalIndicatorConfig,
    TechnicalIndicatorState,
)


def test_rsi_becomes_available_after_period_plus_one_closes():
    state = TechnicalIndicatorState(TechnicalIndicatorConfig(rsi_period=14))

    last = None
    for idx in range(15):
        price = 100.0 + idx
        last = state.update_symbol("BTC/USDT", high=price, low=price, close=price)

    assert last is not None
    assert last.rsi_14 == pytest.approx(100.0)


def test_rsi_handles_mixed_gains_and_losses():
    state = TechnicalIndicatorState(TechnicalIndicatorConfig(rsi_period=14))
    prices = [
        100.0, 102.0, 101.0, 103.0, 104.0,
        102.0, 105.0, 107.0, 106.0, 108.0,
        109.0, 107.0, 110.0, 112.0, 111.0,
    ]

    last = None
    for price in prices:
        last = state.update_symbol("ETH/USDT", high=price + 1.0, low=price - 1.0, close=price)

    assert last is not None
    assert 60.0 < last.rsi_14 < 75.0


def test_macd_is_normalized_as_percent_of_close():
    state = TechnicalIndicatorState()

    last = None
    for idx in range(40):
        price = 100.0 + idx * 0.5
        last = state.update_symbol("BTC/USDT", high=price, low=price, close=price)

    assert last is not None
    assert last.macd_line_pct is not None
    assert last.macd_signal_pct is not None
    assert last.macd_histogram_pct is not None
    assert last.macd_line_pct > 0.0
    assert abs(last.macd_histogram_pct) < abs(last.macd_line_pct)


def test_atr_percent_uses_true_range_and_previous_close_gap():
    state = TechnicalIndicatorState(TechnicalIndicatorConfig(atr_period=3))
    candles = [
        (101.0, 99.0, 100.0),
        (103.0, 99.0, 102.0),
        (110.0, 108.0, 109.0),
        (111.0, 107.0, 108.0),
    ]

    last = None
    for high, low, close in candles:
        last = state.update_symbol("SOL/USDT", high=high, low=low, close=close)

    assert last is not None
    assert last.atr_14_pct is not None
    assert last.atr_14_pct == pytest.approx(((4.0 + 8.0 + 4.0) / 3.0) / 108.0 * 100.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_technical_indicators.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'panteon_v2.analysis.technical_indicators'`.

- [ ] **Step 3: Add immutable indicator value type**

In `src/panteon_v2/domain/types.py`, add `import math` near the top and add this dataclass before `MarketSnapshot`:

```python
@dataclass(frozen=True)
class TechnicalIndicators:
    """Per-symbol normalized technical context for a closed market bar."""

    rsi_14: Optional[float] = None
    macd_line_pct: Optional[float] = None
    macd_signal_pct: Optional[float] = None
    macd_histogram_pct: Optional[float] = None
    atr_14_pct: Optional[float] = None

    def __post_init__(self) -> None:
        for name in (
            "rsi_14",
            "macd_line_pct",
            "macd_signal_pct",
            "macd_histogram_pct",
            "atr_14_pct",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"TechnicalIndicators.{name} must be finite")
            object.__setattr__(self, name, parsed)
        if self.rsi_14 is not None and not 0.0 <= self.rsi_14 <= 100.0:
            raise ValueError("TechnicalIndicators.rsi_14 must be in [0, 100]")
        if self.atr_14_pct is not None and self.atr_14_pct < 0.0:
            raise ValueError("TechnicalIndicators.atr_14_pct must be >= 0")
```

- [ ] **Step 4: Implement indicator engine**

Create `src/panteon_v2/analysis/technical_indicators.py`:

```python
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from panteon_v2.domain.types import TechnicalIndicators


@dataclass(frozen=True)
class TechnicalIndicatorConfig:
    rsi_period: int = 14
    macd_fast_period: int = 12
    macd_slow_period: int = 26
    macd_signal_period: int = 9
    atr_period: int = 14

    def __post_init__(self) -> None:
        if self.rsi_period <= 1:
            raise ValueError("rsi_period must be > 1")
        if self.macd_fast_period <= 1:
            raise ValueError("macd_fast_period must be > 1")
        if self.macd_slow_period <= self.macd_fast_period:
            raise ValueError("macd_slow_period must be greater than macd_fast_period")
        if self.macd_signal_period <= 1:
            raise ValueError("macd_signal_period must be > 1")
        if self.atr_period <= 1:
            raise ValueError("atr_period must be > 1")


@dataclass
class _SymbolState:
    closes: Deque[float]
    true_ranges: Deque[float]
    prev_close: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    macd_signal: Optional[float] = None


@dataclass
class TechnicalIndicatorState:
    config: TechnicalIndicatorConfig = field(default_factory=TechnicalIndicatorConfig)
    _symbols: dict[str, _SymbolState] = field(default_factory=dict)

    def update_symbol(
        self,
        symbol: str,
        *,
        high: float,
        low: float,
        close: float,
    ) -> TechnicalIndicators:
        clean = str(symbol or "").upper()
        if not clean:
            raise ValueError("symbol must be non-empty")
        high_f = _finite_positive(high, "high")
        low_f = _finite_positive(low, "low")
        close_f = _finite_positive(close, "close")
        if high_f < low_f:
            raise ValueError("high must be >= low")

        maxlen = max(self.config.rsi_period + 1, self.config.atr_period, self.config.macd_slow_period + self.config.macd_signal_period + 1)
        state = self._symbols.get(clean)
        if state is None:
            state = _SymbolState(closes=deque(maxlen=maxlen), true_ranges=deque(maxlen=self.config.atr_period))
            self._symbols[clean] = state

        tr = _true_range(high_f, low_f, state.prev_close)
        state.true_ranges.append(tr)
        state.closes.append(close_f)
        state.ema_fast = _ema_update(state.ema_fast, close_f, self.config.macd_fast_period)
        state.ema_slow = _ema_update(state.ema_slow, close_f, self.config.macd_slow_period)

        macd_line_pct = None
        macd_signal_pct = None
        macd_histogram_pct = None
        if state.ema_fast is not None and state.ema_slow is not None:
            macd_line = state.ema_fast - state.ema_slow
            state.macd_signal = _ema_update(state.macd_signal, macd_line, self.config.macd_signal_period)
            macd_line_pct = macd_line / close_f * 100.0
            if state.macd_signal is not None:
                macd_signal_pct = state.macd_signal / close_f * 100.0
                macd_histogram_pct = (macd_line - state.macd_signal) / close_f * 100.0

        state.prev_close = close_f
        return TechnicalIndicators(
            rsi_14=_rsi(list(state.closes), self.config.rsi_period),
            macd_line_pct=macd_line_pct,
            macd_signal_pct=macd_signal_pct,
            macd_histogram_pct=macd_histogram_pct,
            atr_14_pct=_atr_pct(state.true_ranges, close_f, self.config.atr_period),
        )


def _finite_positive(value: float, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def _ema_update(previous: Optional[float], value: float, period: int) -> float:
    if previous is None:
        return float(value)
    alpha = 2.0 / (float(period) + 1.0)
    return float(previous) + alpha * (float(value) - float(previous))


def _rsi(closes: list[float], period: int) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    window = closes[-(period + 1):]
    gains = []
    losses = []
    for previous, current in zip(window, window[1:]):
        delta = current - previous
        gains.append(max(delta, 0.0))
        losses.append(max(-delta, 0.0))
    avg_gain = sum(gains) / float(period)
    avg_loss = sum(losses) / float(period)
    if avg_loss <= 0.0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(high: float, low: float, prev_close: Optional[float]) -> float:
    if prev_close is None:
        return high - low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def _atr_pct(true_ranges: Deque[float], close: float, period: int) -> Optional[float]:
    if len(true_ranges) < period:
        return None
    atr = sum(true_ranges) / float(period)
    return atr / close * 100.0
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_technical_indicators.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src\panteon_v2\domain\types.py src\panteon_v2\analysis\technical_indicators.py src\panteon_v2\tests\test_technical_indicators.py
git commit -m "feat: add technical indicator engine"
```

---

### Task 2: Add Technical Indicators To MarketSnapshot

**Files:**
- Modify: `src/panteon_v2/domain/types.py`
- Modify: `src/panteon_v2/tests/_helpers.py`
- Test: `src/panteon_v2/tests/test_types.py`

- [ ] **Step 1: Write failing domain test**

Add this test to `src/panteon_v2/tests/test_types.py`:

```python
def test_market_snapshot_accepts_immutable_technical_indicators():
    from datetime import datetime, timezone

    from panteon_v2.domain.types import MarketSnapshot, Regime, TechnicalIndicators

    snapshot = MarketSnapshot(
        bar=1,
        timestamp=datetime.now(timezone.utc),
        regime=Regime.BULLISH,
        prices={"BTC/USDT": 100.0},
        volumes={"BTC/USDT": 1000.0},
        technicals_by_symbol={
            "BTC/USDT": TechnicalIndicators(
                rsi_14=61.0,
                macd_line_pct=0.25,
                macd_signal_pct=0.15,
                macd_histogram_pct=0.10,
                atr_14_pct=1.75,
            )
        },
    )

    assert snapshot.technicals_by_symbol["BTC/USDT"].rsi_14 == 61.0
    assert snapshot.technicals_by_symbol["BTC/USDT"].atr_14_pct == 1.75
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_types.py::test_market_snapshot_accepts_immutable_technical_indicators -q
```

Expected: FAIL because `MarketSnapshot` does not accept `technicals_by_symbol`.

- [ ] **Step 3: Modify domain types**

In `src/panteon_v2/domain/types.py`, add this field to `MarketSnapshot`:

```python
    technicals_by_symbol: Dict[str, TechnicalIndicators] = field(default_factory=dict)
```

In `MarketSnapshot.__post_init__`, keep existing regime confidence validation and add:

```python
        for symbol in self.technicals_by_symbol:
            if str(symbol).upper() not in {str(item).upper() for item in self.prices}:
                raise ValueError(
                    f"MarketSnapshot.technicals_by_symbol contains unknown symbol {symbol!r}"
                )
```

- [ ] **Step 4: Run domain tests**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_types.py src\panteon_v2\tests\test_technical_indicators.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src\panteon_v2\domain\types.py src\panteon_v2\tests\test_types.py src\panteon_v2\tests\_helpers.py
git commit -m "feat: attach technical indicators to market snapshots"
```

---

### Task 3: Populate Indicators In Retrodate Snapshots

**Files:**
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

- [ ] **Step 1: Write failing retrodate loader test**

Add this test near the existing snapshot loader tests:

```python
def test_load_retrodate_year_snapshots_populates_technical_indicators(tmp_path):
    csv_path = tmp_path / "crypto_1m_2025_all_symbols.csv"
    start_ts = 1735689600000
    rows = []
    for index in range(40):
        close = 100.0 + index
        rows.append({
            "timestamp": start_ts + index * 60 * 60 * 1000,
            "open": close - 0.5,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": close * 10,
            "symbol": "BTC/USDT",
            "datetime": "",
        })
    _write_rows(csv_path, rows)

    snapshots = load_retrodate_year_snapshots(
        csv_path,
        stride_minutes=60,
        state=RetrodateSnapshotState(),
    )

    tech = snapshots[-1].technicals_by_symbol["BTC/USDT"]
    assert tech.rsi_14 == pytest.approx(100.0)
    assert tech.macd_line_pct is not None
    assert tech.macd_signal_pct is not None
    assert tech.macd_histogram_pct is not None
    assert tech.atr_14_pct is not None
    assert tech.atr_14_pct > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_retrodate_market_runner.py::test_load_retrodate_year_snapshots_populates_technical_indicators -q
```

Expected: FAIL because `technicals_by_symbol` is empty.

- [ ] **Step 3: Modify retrodate snapshot state and loader**

In `src/panteon_v2/analysis/retrodate_market_runner.py`, import:

```python
from .technical_indicators import TechnicalIndicatorState
```

Add this field to `RetrodateSnapshotState`:

```python
    technicals: TechnicalIndicatorState = field(default_factory=TechnicalIndicatorState)
```

Inside `load_retrodate_year_snapshots`, replace separate close-only storage with OHLC storage:

```python
    candles_by_ts: dict[int, dict[str, tuple[float, float, float]]] = {}
```

When reading each row, parse:

```python
            high = _coerce_float(row.get("high"))
            low = _coerce_float(row.get("low"))
            if high <= 0 or low <= 0 or high < low:
                high = close
                low = close
            candles_by_ts.setdefault(timestamp, {})[symbol] = (high, low, close)
```

When building each snapshot:

```python
        technicals_by_symbol = {
            symbol: state.technicals.update_symbol(
                symbol,
                high=candles_by_ts[timestamp][symbol][0],
                low=candles_by_ts[timestamp][symbol][1],
                close=candles_by_ts[timestamp][symbol][2],
            )
            for symbol in prices
        }
```

Pass it into `MarketSnapshot`:

```python
                technicals_by_symbol=technicals_by_symbol,
```

Keep `_update_symbol_close_history(prices, state)` after snapshot creation so existing lookback returns continue to represent prior history plus current bar behavior exactly as current tests expect.

- [ ] **Step 4: Run focused retrodate tests**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_retrodate_market_runner.py::test_load_retrodate_year_snapshots_populates_technical_indicators src\panteon_v2\tests\test_retrodate_market_runner.py::test_load_retrodate_year_snapshots_populates_lookback_returns -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add src\panteon_v2\analysis\retrodate_market_runner.py src\panteon_v2\tests\test_retrodate_market_runner.py
git commit -m "feat: populate technical indicators in retrodate snapshots"
```

---

### Task 4: Add Flash Technical Overlay Config And Audit Fields

**Files:**
- Modify: `src/panteon_v2/selection/flash_allocator.py`
- Test: `src/panteon_v2/tests/test_flash_allocator.py`

- [ ] **Step 1: Write failing allocator audit test**

Add `import pytest` near the top of `src/panteon_v2/tests/test_flash_allocator.py`, then add this method inside `TestFlashAllocator`:

```python
def test_technical_overlay_emits_candidate_audit_fields(self):
    from panteon_v2.domain.types import TechnicalIndicators

    perf = PerformanceMemory(trade_fraction=1.0)
    qm = QuarantineManager(seed=set())
    active = FakeAgent("TechAgent", {"BTC": Action.FUT_LONG_FULL})
    _add_perf(perf, "TechAgent", Regime.BULLISH, 10, 2.0, start_id=1)
    market = replace(
        make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
        technicals_by_symbol={
            "BTC": TechnicalIndicators(
                rsi_14=55.0,
                macd_line_pct=0.20,
                macd_signal_pct=0.10,
                macd_histogram_pct=0.10,
                atr_14_pct=2.0,
            )
        },
    )

    allocator = FlashAllocator(
        perf=perf,
        qm=qm,
        config=FlashAllocatorConfig(technical_overlay_enabled=True),
    )

    decision = allocator.decide(market, agents=[active], players=[], signal_id_start=1)[0]
    row = next(item for item in decision.candidates if item.label == "TechAgent")

    assert row.technical_rsi_14 == pytest.approx(55.0)
    assert row.technical_macd_histogram_pct == pytest.approx(0.10)
    assert row.technical_atr_14_pct == pytest.approx(2.0)
    assert row.technical_alignment == "long_aligned"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_flash_allocator.py::TestFlashAllocator::test_technical_overlay_emits_candidate_audit_fields -q
```

Expected: FAIL because `technical_overlay_enabled` and audit fields do not exist.

- [ ] **Step 3: Add config and audit fields**

In `FlashAllocatorConfig`, add:

```python
    technical_overlay_enabled: bool = False
    technical_hard_gate_enabled: bool = False
    technical_score_bonus: float = 0.10
    technical_score_penalty: float = 0.25
    technical_rsi_long_min: float = 45.0
    technical_rsi_long_max: float = 72.0
    technical_rsi_short_min: float = 28.0
    technical_rsi_short_max: float = 55.0
    technical_macd_histogram_min_abs_pct: float = 0.0
    technical_atr_risk_sizing_enabled: bool = False
    technical_atr_target_pct: float = 2.0
    technical_atr_min_pct: float = 0.25
    technical_atr_max_mult: float = 1.5
```

In `FlashAllocatorConfig.__post_init__`, add validation:

```python
        if self.technical_score_bonus < 0.0:
            raise ValueError("technical_score_bonus must be >= 0")
        if self.technical_score_penalty < 0.0:
            raise ValueError("technical_score_penalty must be >= 0")
        if not 0.0 <= self.technical_rsi_long_min <= self.technical_rsi_long_max <= 100.0:
            raise ValueError("technical RSI long bounds must be ordered within [0, 100]")
        if not 0.0 <= self.technical_rsi_short_min <= self.technical_rsi_short_max <= 100.0:
            raise ValueError("technical RSI short bounds must be ordered within [0, 100]")
        if self.technical_macd_histogram_min_abs_pct < 0.0:
            raise ValueError("technical_macd_histogram_min_abs_pct must be >= 0")
        if self.technical_atr_target_pct <= 0.0:
            raise ValueError("technical_atr_target_pct must be > 0")
        if self.technical_atr_min_pct <= 0.0:
            raise ValueError("technical_atr_min_pct must be > 0")
        if self.technical_atr_max_mult <= 0.0:
            raise ValueError("technical_atr_max_mult must be > 0")
```

In `FlashCandidateAudit`, add:

```python
    technical_rsi_14: Optional[float] = None
    technical_macd_histogram_pct: Optional[float] = None
    technical_atr_14_pct: Optional[float] = None
    technical_alignment: str = ""
    technical_score_adjustment: float = 0.0
    technical_gate_reason: str = ""
```

Add the matching keys to `FlashCandidateAudit.as_dict()`.

- [ ] **Step 4: Add helper methods**

In `flash_allocator.py`, add:

```python
def _technical_indicators_for_symbol(market: MarketSnapshot, symbol: str):
    clean_symbol = str(symbol or "").upper()
    for raw_symbol, indicators in (getattr(market, "technicals_by_symbol", {}) or {}).items():
        if str(raw_symbol).upper() == clean_symbol:
            return indicators
    return None
```

Inside `FlashAllocator`, add:

```python
    def _technical_alignment(self, market: MarketSnapshot, symbol: str, action: Action) -> tuple[str, float, str]:
        if not self._config.technical_overlay_enabled or not action.is_open:
            return "", 0.0, ""
        tech = _technical_indicators_for_symbol(market, symbol)
        if tech is None:
            return "missing", 0.0, "technical_missing"
        rsi = tech.rsi_14
        hist = tech.macd_histogram_pct
        if rsi is None or hist is None:
            return "missing", 0.0, "technical_missing"
        min_hist = float(self._config.technical_macd_histogram_min_abs_pct)
        if action.is_long_open:
            rsi_ok = self._config.technical_rsi_long_min <= rsi <= self._config.technical_rsi_long_max
            macd_ok = hist > min_hist
            if rsi_ok and macd_ok:
                return "long_aligned", float(self._config.technical_score_bonus), ""
            return "long_misaligned", -float(self._config.technical_score_penalty), "technical_long_misaligned"
        if action.is_short_open:
            rsi_ok = self._config.technical_rsi_short_min <= rsi <= self._config.technical_rsi_short_max
            macd_ok = hist < -min_hist
            if rsi_ok and macd_ok:
                return "short_aligned", float(self._config.technical_score_bonus), ""
            return "short_misaligned", -float(self._config.technical_score_penalty), "technical_short_misaligned"
        return "", 0.0, ""
```

- [ ] **Step 5: Wire helper into candidate scoring**

In the candidate scoring block, after funding and genetics score adjustments but before `effective_score`, add:

```python
            technical_alignment, technical_score_adjustment, technical_gate_reason = (
                self._technical_alignment(market, symbol, output.action)
            )
            gate_score += technical_score_adjustment
            tech = _technical_indicators_for_symbol(market, symbol)
```

Before terminal deny checks, add the hard gate:

```python
            elif (
                self._config.technical_overlay_enabled
                and self._config.technical_hard_gate_enabled
                and output.action.is_open
                and technical_gate_reason
            ):
                rejected = True
                reason = technical_gate_reason
```

When constructing `FlashCandidateAudit`, populate:

```python
                technical_rsi_14=None if tech is None else tech.rsi_14,
                technical_macd_histogram_pct=None if tech is None else tech.macd_histogram_pct,
                technical_atr_14_pct=None if tech is None else tech.atr_14_pct,
                technical_alignment=technical_alignment,
                technical_score_adjustment=technical_score_adjustment,
                technical_gate_reason=technical_gate_reason,
```

- [ ] **Step 6: Run focused allocator tests**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_flash_allocator.py::TestFlashAllocator::test_technical_overlay_emits_candidate_audit_fields -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add src\panteon_v2\selection\flash_allocator.py src\panteon_v2\tests\test_flash_allocator.py
git commit -m "feat: add Flash technical overlay audit fields"
```

---

### Task 5: Add Technical Hard Gate And ATR Risk Sizing Tests

**Files:**
- Modify: `src/panteon_v2/selection/flash_allocator.py`
- Test: `src/panteon_v2/tests/test_flash_allocator.py`

- [ ] **Step 1: Write hard gate test**

Add this method inside `TestFlashAllocator`:

```python
def test_technical_hard_gate_blocks_long_when_rsi_or_macd_misaligned(self):
    perf = PerformanceMemory(trade_fraction=1.0)
    qm = QuarantineManager(seed=set())
    active = FakeAgent("TechLong", {"BTC": Action.FUT_LONG_FULL})
    _add_perf(perf, "TechLong", Regime.BULLISH, 10, 2.0, start_id=1)
    market = replace(
        make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
        technicals_by_symbol={
            "BTC": TechnicalIndicators(
                rsi_14=82.0,
                macd_histogram_pct=-0.05,
                atr_14_pct=2.0,
            )
        },
    )

    allocator = FlashAllocator(
        perf=perf,
        qm=qm,
        config=FlashAllocatorConfig(
            technical_overlay_enabled=True,
            technical_hard_gate_enabled=True,
        ),
    )

    decision = allocator.decide(market, agents=[active], players=[], signal_id_start=1)[0]

    assert decision.selected_actor == "NoTrade"
    row = next(item for item in decision.candidates if item.label == "TechLong")
    assert row.reason == "technical_long_misaligned"
```

- [ ] **Step 2: Write ATR risk sizing test**

Add this method inside `TestFlashAllocator`:

```python
def test_technical_atr_risk_sizing_reduces_size_when_atr_is_high(self):
    perf = PerformanceMemory(trade_fraction=1.0)
    qm = QuarantineManager(seed=set())
    active = FakeAgent("AtrAgent", {"BTC": Action.FUT_LONG_FULL})
    _add_perf(perf, "AtrAgent", Regime.BULLISH, 10, 2.0, start_id=1)
    market = replace(
        make_market(regime=Regime.BULLISH, prices={"BTC": 100.0}),
        technicals_by_symbol={
            "BTC": TechnicalIndicators(
                rsi_14=55.0,
                macd_histogram_pct=0.10,
                atr_14_pct=4.0,
            )
        },
    )

    allocator = FlashAllocator(
        perf=perf,
        qm=qm,
        config=FlashAllocatorConfig(
            technical_overlay_enabled=True,
            technical_atr_risk_sizing_enabled=True,
            technical_atr_target_pct=2.0,
            technical_atr_max_mult=1.5,
        ),
    )

    decision = allocator.decide(market, agents=[active], players=[], signal_id_start=1)[0]

    assert decision.selected_actor == "AtrAgent"
    assert decision.signal is not None
    assert decision.signal.risk_mult == pytest.approx(0.5)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_flash_allocator.py::TestFlashAllocator::test_technical_hard_gate_blocks_long_when_rsi_or_macd_misaligned src\panteon_v2\tests\test_flash_allocator.py::TestFlashAllocator::test_technical_atr_risk_sizing_reduces_size_when_atr_is_high -q
```

Expected: first test may pass if Task 4 hard gate is complete; second FAIL until ATR sizing is wired.

- [ ] **Step 4: Implement ATR risk sizing**

In `FlashAllocator._risk_mult_for_candidate`, after `volatility_risk_sizing_enabled`, add:

```python
        if self._config.technical_overlay_enabled and self._config.technical_atr_risk_sizing_enabled:
            risk_mult *= self._technical_atr_risk_mult(market, symbol)
```

Add method:

```python
    def _technical_atr_risk_mult(self, market: MarketSnapshot, symbol: str) -> float:
        tech = _technical_indicators_for_symbol(market, symbol)
        if tech is None or tech.atr_14_pct is None:
            return 1.0
        atr = max(float(self._config.technical_atr_min_pct), float(tech.atr_14_pct))
        raw = float(self._config.technical_atr_target_pct) / atr
        return _clamp(raw, 0.0, float(self._config.technical_atr_max_mult))
```

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_flash_allocator.py::TestFlashAllocator::test_technical_hard_gate_blocks_long_when_rsi_or_macd_misaligned src\panteon_v2\tests\test_flash_allocator.py::TestFlashAllocator::test_technical_atr_risk_sizing_reduces_size_when_atr_is_high -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src\panteon_v2\selection\flash_allocator.py src\panteon_v2\tests\test_flash_allocator.py
git commit -m "feat: apply technical gates and ATR risk sizing"
```

---

### Task 6: Add Retrodate CLI Flags And Config Propagation

**Files:**
- Modify: `src/panteon_v2/analysis/retrodate_market_runner.py`
- Test: `src/panteon_v2/tests/test_retrodate_market_runner.py`

- [ ] **Step 1: Add CLI config test**

Add:

```python
def test_cli_config_accepts_flash_technical_overlay_flags():
    config = runner._parse_cli_config([
        "--enable-flash",
        "--enable-flash-technical-overlay",
        "--enable-flash-technical-hard-gate",
        "--enable-flash-technical-atr-risk-sizing",
        "--flash-technical-score-bonus", "0.12",
        "--flash-technical-score-penalty", "0.30",
        "--flash-technical-atr-target-pct", "1.8",
    ])

    assert config.flash_technical_overlay_enabled is True
    assert config.flash_technical_hard_gate_enabled is True
    assert config.flash_technical_atr_risk_sizing_enabled is True
    assert config.flash_technical_score_bonus == pytest.approx(0.12)
    assert config.flash_technical_score_penalty == pytest.approx(0.30)
    assert config.flash_technical_atr_target_pct == pytest.approx(1.8)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_retrodate_market_runner.py::test_cli_config_accepts_flash_technical_overlay_flags -q
```

Expected: FAIL because CLI flags/config fields do not exist.

- [ ] **Step 3: Add config fields to RetrodateMarketConfig**

Add near Flash economic config:

```python
    flash_technical_overlay_enabled: bool = False
    flash_technical_hard_gate_enabled: bool = False
    flash_technical_score_bonus: float = 0.10
    flash_technical_score_penalty: float = 0.25
    flash_technical_rsi_long_min: float = 45.0
    flash_technical_rsi_long_max: float = 72.0
    flash_technical_rsi_short_min: float = 28.0
    flash_technical_rsi_short_max: float = 55.0
    flash_technical_macd_histogram_min_abs_pct: float = 0.0
    flash_technical_atr_risk_sizing_enabled: bool = False
    flash_technical_atr_target_pct: float = 2.0
    flash_technical_atr_min_pct: float = 0.25
    flash_technical_atr_max_mult: float = 1.5
```

Add matching validation in `RetrodateMarketConfig.__post_init__` using the same bounds as `FlashAllocatorConfig`.

- [ ] **Step 4: Add parser flags and config mapping**

Add parser arguments near existing Flash volatility flags:

```python
    parser.add_argument("--enable-flash-technical-overlay", action="store_true")
    parser.add_argument("--enable-flash-technical-hard-gate", action="store_true")
    parser.add_argument("--flash-technical-score-bonus", type=float, default=0.10)
    parser.add_argument("--flash-technical-score-penalty", type=float, default=0.25)
    parser.add_argument("--flash-technical-rsi-long-min", type=float, default=45.0)
    parser.add_argument("--flash-technical-rsi-long-max", type=float, default=72.0)
    parser.add_argument("--flash-technical-rsi-short-min", type=float, default=28.0)
    parser.add_argument("--flash-technical-rsi-short-max", type=float, default=55.0)
    parser.add_argument("--flash-technical-macd-histogram-min-abs-pct", type=float, default=0.0)
    parser.add_argument("--enable-flash-technical-atr-risk-sizing", action="store_true")
    parser.add_argument("--flash-technical-atr-target-pct", type=float, default=2.0)
    parser.add_argument("--flash-technical-atr-min-pct", type=float, default=0.25)
    parser.add_argument("--flash-technical-atr-max-mult", type=float, default=1.5)
```

In `config_from_args`, map each argument into `RetrodateMarketConfig`.

In the `FlashAllocatorConfig(...)` construction, pass:

```python
        technical_overlay_enabled=config.flash_technical_overlay_enabled,
        technical_hard_gate_enabled=config.flash_technical_hard_gate_enabled,
        technical_score_bonus=config.flash_technical_score_bonus,
        technical_score_penalty=config.flash_technical_score_penalty,
        technical_rsi_long_min=config.flash_technical_rsi_long_min,
        technical_rsi_long_max=config.flash_technical_rsi_long_max,
        technical_rsi_short_min=config.flash_technical_rsi_short_min,
        technical_rsi_short_max=config.flash_technical_rsi_short_max,
        technical_macd_histogram_min_abs_pct=(
            config.flash_technical_macd_histogram_min_abs_pct
        ),
        technical_atr_risk_sizing_enabled=(
            config.flash_technical_atr_risk_sizing_enabled
        ),
        technical_atr_target_pct=config.flash_technical_atr_target_pct,
        technical_atr_min_pct=config.flash_technical_atr_min_pct,
        technical_atr_max_mult=config.flash_technical_atr_max_mult,
```

Add fields to the run summary/config JSON payload:

```python
        "flash_technical_overlay_enabled": config.flash_technical_overlay_enabled,
        "flash_technical_hard_gate_enabled": config.flash_technical_hard_gate_enabled,
        "flash_technical_atr_risk_sizing_enabled": config.flash_technical_atr_risk_sizing_enabled,
```

- [ ] **Step 5: Run config test**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_retrodate_market_runner.py::test_cli_config_accepts_flash_technical_overlay_flags -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add src\panteon_v2\analysis\retrodate_market_runner.py src\panteon_v2\tests\test_retrodate_market_runner.py
git commit -m "feat: expose Flash technical overlay in retrodate config"
```

---

### Task 7: Add Reporting And Mechanism Documentation

**Files:**
- Modify: `docs/PANTEON_FLASH_MECHANISMS.md`
- Modify: `src/panteon_v2/dashboards/png_renderer.py`
- Test: `src/panteon_v2/tests/test_png_renderer.py`

- [ ] **Step 1: Document mechanism**

Append to `docs/PANTEON_FLASH_MECHANISMS.md`:

```markdown
## Technical overlay

`technical_overlay_enabled` adds RSI(14), MACD(12,26,9), and ATR(14) context
to Flash candidate audit rows. The overlay is diagnostic by default.

For open candidates:

- Long alignment requires RSI inside `technical_rsi_long_min/max` and a
  positive MACD histogram above `technical_macd_histogram_min_abs_pct`.
- Short alignment requires RSI inside `technical_rsi_short_min/max` and a
  negative MACD histogram below the negative histogram threshold.
- Aligned candidates receive `technical_score_bonus`.
- Misaligned candidates receive `technical_score_penalty`.
- `technical_hard_gate_enabled` converts misalignment into a hard NoTrade
  rejection. This remains disabled until retrodate evidence beats baseline.
- `technical_atr_risk_sizing_enabled` scales open risk toward constant ATR
  exposure with `technical_atr_target_pct / atr_14_pct`, capped by
  `technical_atr_max_mult`.

Technical indicators are computed from closed OHLC candles in Retrodate and
stored on `MarketSnapshot.technicals_by_symbol`. They are not allowed to read
future bars or execution outcomes.
```

- [ ] **Step 2: Add dashboard/report counters**

In `src/panteon_v2/dashboards/png_renderer.py`, update `_flash_candidate_rows` so technical adjustments are aggregated for the Flash Top Candidates panel. Extend the row default dict:

```python
                "technical_aligned_count": 0,
                "technical_misaligned_count": 0,
                "technical_missing_count": 0,
                "technical_score_adjustment": 0.0,
```

Inside the per-candidate loop, after `row["best_score"]` is updated, add:

```python
            alignment = str(candidate.get("technical_alignment") or "").strip()
            if alignment.endswith("_aligned"):
                row["technical_aligned_count"] += 1
            elif alignment.endswith("_misaligned"):
                row["technical_misaligned_count"] += 1
            elif alignment == "missing":
                row["technical_missing_count"] += 1
            row["technical_score_adjustment"] += _float_value(
                candidate.get("technical_score_adjustment", 0.0)
            )
```

In `_draw_flash_candidates_panel`, append a compact technical suffix when any technical field is non-zero:

```python
        tech_total = (
            int(row.get("technical_aligned_count", 0))
            + int(row.get("technical_misaligned_count", 0))
            + int(row.get("technical_missing_count", 0))
        )
        if tech_total:
            suffix += (
                f" tech {int(row.get('technical_aligned_count', 0))}/"
                f"{int(row.get('technical_misaligned_count', 0))}"
            )
```

Add test data to `src/panteon_v2/tests/test_png_renderer.py` using existing renderer fixtures:

```python
def test_flash_candidate_rows_aggregate_technical_overlay_counts(self):
    from panteon_v2.dashboards import png_renderer

    rows = png_renderer._flash_candidate_rows({
        "flash": {
            "decisions": [
                {
                    "symbol": "BTC",
                    "selected_actor": "TechAgent",
                    "candidates": [
                        {
                            "label": "TechAgent",
                            "actor_type": "agent",
                            "effective_score": 1.2,
                            "action": "FUT_LONG_FULL",
                            "rejected": False,
                            "technical_alignment": "long_aligned",
                            "technical_score_adjustment": 0.10,
                        },
                    ],
                },
                {
                    "symbol": "ETH",
                    "selected_actor": "NoTrade",
                    "candidates": [
                        {
                            "label": "TechAgent",
                            "actor_type": "agent",
                            "effective_score": 0.7,
                            "action": "FUT_LONG_FULL",
                            "rejected": True,
                            "reason": "technical_long_misaligned",
                            "technical_alignment": "long_misaligned",
                            "technical_score_adjustment": -0.25,
                        },
                    ],
                },
            ],
        },
    })

    row = next(item for item in rows if item["label"] == "TechAgent")

    self.assertEqual(row["technical_aligned_count"], 1)
    self.assertEqual(row["technical_misaligned_count"], 1)
    self.assertAlmostEqual(row["technical_score_adjustment"], -0.15)
```

- [ ] **Step 3: Run docs/renderer tests**

Run:

```powershell
python -m pytest src\panteon_v2\tests\test_png_renderer.py -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```powershell
git add docs\PANTEON_FLASH_MECHANISMS.md src\panteon_v2\dashboards\png_renderer.py src\panteon_v2\tests\test_png_renderer.py
git commit -m "docs: describe Flash technical overlay"
```

---

### Task 8: Run Validation Matrix Before Any Live Enablement

**Files:**
- No source code changes expected.
- Output: new `Results/RetrodateMarket/...` runs and report summaries.

- [ ] **Step 1: Run baseline**

Run:

```powershell
python -m panteon_v2.analysis.retrodate_market_runner --enable-flash --years 2022 2023 2024 2025 2026 --stride-minutes 60
```

Expected: run completes and writes a `Results/RetrodateMarket/.../summary.json`.

- [ ] **Step 2: Run diagnostic-only technical overlay**

Run:

```powershell
python -m panteon_v2.analysis.retrodate_market_runner --enable-flash --enable-flash-technical-overlay --years 2022 2023 2024 2025 2026 --stride-minutes 60
```

Expected: total trade selection should be close to baseline except score penalty/bonus effects if `technical_score_bonus` or `technical_score_penalty` are non-zero. For pure diagnostic pass, repeat with both set to zero:

```powershell
python -m panteon_v2.analysis.retrodate_market_runner --enable-flash --enable-flash-technical-overlay --flash-technical-score-bonus 0 --flash-technical-score-penalty 0 --years 2022 2023 2024 2025 2026 --stride-minutes 60
```

- [ ] **Step 3: Run ATR sizing candidate**

Run:

```powershell
python -m panteon_v2.analysis.retrodate_market_runner --enable-flash --enable-flash-technical-overlay --enable-flash-technical-atr-risk-sizing --flash-technical-score-bonus 0 --flash-technical-score-penalty 0 --years 2022 2023 2024 2025 2026 --stride-minutes 60
```

Promotion rule: ATR sizing can advance only if it reduces max drawdown or realized loss without reducing net PnL more than the existing project threshold for selected-subset candidates.

- [ ] **Step 4: Run soft score overlay candidate**

Run:

```powershell
python -m panteon_v2.analysis.retrodate_market_runner --enable-flash --enable-flash-technical-overlay --flash-technical-score-bonus 0.10 --flash-technical-score-penalty 0.25 --years 2022 2023 2024 2025 2026 --stride-minutes 60
```

Promotion rule: soft overlay can advance only if it improves full-window PnL, latest-window PnL, and max drawdown versus baseline, and does not increase rejected actionable signals into a lower-quality NoTrade mix.

- [ ] **Step 5: Run hard gate candidate only after soft overlay wins**

Run:

```powershell
python -m panteon_v2.analysis.retrodate_market_runner --enable-flash --enable-flash-technical-overlay --enable-flash-technical-hard-gate --flash-technical-score-bonus 0.10 --flash-technical-score-penalty 0.25 --years 2022 2023 2024 2025 2026 --stride-minutes 60
```

Promotion rule: hard gate is rejected unless it beats baseline on total PnL, latest-period PnL, max drawdown, and realized downside. A hard gate that only reduces trades is not sufficient.

- [ ] **Step 6: Commit only source changes, not generated bulky results**

```powershell
git status --short
git add src\panteon_v2 docs\PANTEON_FLASH_MECHANISMS.md
git commit -m "feat: integrate technical indicators into Panteon Ultima Flash"
```

---

## Self-Review

Spec coverage:

- RSI is included as bounded entry timing context for long and short opens.
- MACD is included as direction/impulse confirmation via normalized histogram.
- ATR is included as volatility/risk sizing, not direction prediction.
- All features are attached to `MarketSnapshot`, audited per candidate, and gated by retrodate evidence before live use.

Placeholder scan:

- No task relies on undefined future modules except modules created in earlier tasks.
- Every code-facing task includes concrete file paths, test code, and commands.

Type consistency:

- `TechnicalIndicators` is the domain type used by indicator engine, snapshots, tests, and allocator.
- `technicals_by_symbol` is the single field name across loader and allocator.
- Candidate audit field names use `technical_*` prefix.

## Execution Options

Plan complete and saved to `docs/superpowers/plans/2026-05-26-panteon-ultima-rsi-macd-atr.md`.

1. Subagent-Driven (recommended) - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. Inline Execution - execute tasks in this session using executing-plans, batch execution with checkpoints.
