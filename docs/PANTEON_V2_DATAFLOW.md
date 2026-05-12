# Panteon v2 — Data Flow (визуальная схема)

Сопровождает `PANTEON_V2_ARCHITECTURE.md`.

## 1. Слои системы

```
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 6 — Dashboards (read-only views)                               │
│   DashboardRenderer ← AttributionLedger, PerformanceMemory, QM       │
└──────────────────────────────────────────────────────────────────────┘
                                ▲
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 5 — Attribution (immutable journal projection)                 │
│   AttributionLedger ← EventLog                                       │
└──────────────────────────────────────────────────────────────────────┘
                                ▲
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 4 — Execution                                                  │
│   TradeExecutor → Exchange  → SymbolHealthMonitor                    │
│                                ▲ records pending failures            │
└──────────────────────────────────────────────────────────────────────┘
                                ▲
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 3 — Decision (one decision per bar)                            │
│   Strategist → PlayerComposer → AgentSelector                        │
│      uses ↓               uses ↓              uses ↓                 │
│   PerformanceMemory   QuarantineManager     ScoringConfig            │
└──────────────────────────────────────────────────────────────────────┘
                                ▲
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 2 — Memory & Quarantine (single sources of truth)              │
│   PerformanceMemory                                                  │
│   QuarantineManager                                                  │
└──────────────────────────────────────────────────────────────────────┘
                                ▲
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — Pure scoring (no state)                                    │
│   regime_score(), confidence_from_sample()                           │
└──────────────────────────────────────────────────────────────────────┘
                                ▲
┌──────────────────────────────────────────────────────────────────────┐
│ LAYER 0 — Core types (immutable)                                     │
│   Action, Regime, Signal, Trade, Metrics                             │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│ EVENT LOG (cross-cutting, append-only)                               │
│   Все компоненты эмиттят events. EventLog read by AttributionLedger. │
└──────────────────────────────────────────────────────────────────────┘
```

## 2. Один bar — последовательность

```
Bar N начинается
    │
    ├─[1] BarLoop.start_bar(N) → EventLog.emit(BarStarted)
    │
    ├─[2] RegimeDetector.detect(market) → regime
    │       └─→ EventLog.emit(RegimeDetected if changed)
    │
    ├─[3] (every 10 bars) QuarantineManager.recompute(perf)
    │       ├─→ added/removed
    │       ├─→ EventLog.emit(QuarantineRecomputed)
    │       └─→ notify subscribers (Selector, PlayerComposer, Strategist)
    │
    ├─[4] (every K bars) Strategist.consider_switch(regime, N)
    │       ├─ candidates = [
    │       │      PlayerComposer.compose_ensemble(regime, default_voting),
    │       │      PlayerComposer.compose_from_profile("PanteonTrendResearch"),
    │       │      PlayerComposer.compose_from_profile("PanteonMeanRev"),
    │       │      ...
    │       │  ]
    │       ├─ для каждого: убедиться что Player.agents ∩ QM.all = ∅
    │       ├─ score каждого через PerformanceMemory.aggregate_for_regime
    │       └─ pick best, switch with cooldown/urgent rules
    │       └─→ EventLog.emit(LeaderSelected)
    │
    ├─[5] leader = Strategist.current_leader()
    │       │     (это RetainedPlayer, переживший с прошлого bar-а
    │       │      или новый из шага 4)
    │       │
    │       └─ leader.vote(market) → [Signal, Signal, ...]
    │           для каждого: EventLog.emit(SignalEmitted)
    │
    ├─[6] для каждого signal:
    │       ├─ TradeExecutor.execute(signal)
    │       │   ├─ if SymbolHealthMonitor.is_blocked(sym) → skip
    │       │   ├─ if RiskLimits.deny(signal) → skip
    │       │   ├─ Exchange.send_order
    │       │   ├─→ EventLog.emit(OrderSent)
    │       │   ├─ wait/poll for fill or timeout
    │       │   ├─→ EventLog.emit(OrderFilled or OrderRejected)
    │       │   ├─ PositionTracker.update
    │       │   │   └─→ EventLog.emit(PositionOpened or PositionClosed)
    │       │   └─ PerformanceMemory.update_from_trade(trade, signal)
    │       └─ if order failed: SymbolHealthMonitor.record_pending_failure(sym)
    │
    ├─[7] (async, every M bars) DashboardRenderer.render_*()
    │       читает из AttributionLedger, PerformanceMemory, QM
    │
    └─[8] EventLog.emit(BarEnded)
```

## 3. Источники истины (SSOT)

```
QUARANTINE STATUS    →  QuarantineManager._dynamic   (никаких копий)
PER-AGENT METRICS    →  PerformanceMemory._data
CURRENT REGIME       →  BarLoop.current_regime       (передаётся как параметр)
CURRENT LEADER       →  Strategist._current
HISTORICAL DECISIONS →  EventLog
REAL ATTRIBUTION     →  AttributionLedger  (projection of EventLog)
SHADOW METRICS       →  PerformanceMemory  (тот же — virtual это просто отдельный регим/контекст)
SYMBOL HEALTH        →  SymbolHealthMonitor
```

## 4. Critical invariants

```
INV-1 (no quarantined in player.agents):
    ∀ bar N, ∀ player ∈ Strategist.candidates(N):
        QM.all_quarantined ∩ {a.label for a in player.agents} = ∅

INV-2 (attribution sums to real):
    ∀ time window W:
        |Σ AttributionLedger.player_pnl(W) - sum(Trade.realized for W)| < ε
            где ε = fees + funding adjustments

INV-3 (single leader):
    ∀ bar N: |{p : p emits signals on bar N}| = 1
        (только Strategist.current_leader делает vote())

INV-4 (regime monotonic decisions):
    Если regime изменился на бар N, то на бар N+1 все подсистемы
    (Selector, Strategist, Dashboards) уже видят новый regime.

INV-5 (deterministic replay):
    Дано тот же EventLog, AttributionLedger.replay_from_event_log()
    даёт идентичный результат при повторных запусках.
```

## 5. Что упрощается

| v1 проблема | v2 решение |
|-------------|-----------|
| `LIVE_AGENT_BLOCKLIST` копировался в N мест | Один `QM.is_quarantined()` |
| 5 разных скоринговых функций | Одна `regime_score()` |
| `_active_weights` ⊕ `_selected_shadow_player` | Один `Strategist.current_leader` |
| `sub_agent_pvs` фейк | `AttributionLedger.player_pnl()` |
| Куча наследников Panteon | Один `EnsemblePlayer` + profiles |
| Запутанный ID между signal/trade | `Trade.signal_id: int` обязателен |

## 6. Расширяемость

Новый тип агента → register в `AgentRegistry`. Всё.
Новый VotingPolicy → имплементировать Protocol. Всё.
Новый дашборд → читать из существующих SSOT. Всё.
Новый exchange → имплементировать `Exchange` Protocol. Всё.

Ни одна доработка не требует менять core code (только regions).
