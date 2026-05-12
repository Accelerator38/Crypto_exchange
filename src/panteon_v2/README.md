# Panteon v2

**Status:** 🚧 Phase 0/1 (skeleton + core types) — under construction.

Чистая переработка Panteon. Решает архитектурные проблемы v1
(`panteon_runtime/`), описанные в `docs/PANTEON_V2_ARCHITECTURE.md`.

## Принципы

1. **Single Source of Truth** — каждое знание имеет ровно одного владельца.
2. **Composition over Inheritance** — `EnsemblePlayer` + profiles вместо
   иерархии `Panteon → PanteonResearch → ...`.
3. **Event-Sourcing** — все решения логируются в `EventLog`,
   `AttributionLedger` строит проекцию для дашбордов.
4. **Pure Functions** — скоринг без состояния, тривиальная testability.
5. **Explicit Synchronization** — точки обновления state расписаны явно.
6. **Observability First** — каждое решение имеет `trace_id`.

## Структура

```
panteon_v2/
├── domain/         # Core types: Action, Regime, Signal, Trade, Metrics
├── scoring/        # Pure scoring functions (regime_score, ...)
├── memory/         # PerformanceMemory, QuarantineManager (single owners)
├── selection/      # AgentSelector, EnsemblePlayer, Strategist
├── execution/      # TradeExecutor, SymbolHealthMonitor
├── attribution/    # EventLog, AttributionLedger
├── dashboards/     # DashboardRenderer (читает только из ledger/perf/qm)
└── tests/          # pytest, property-based via hypothesis
```

## План реализации (см. PANTEON_V2_ARCHITECTURE.md)

- [x] Phase 0 — изоляция, скелет пакета
- [🚧] Phase 1 — core types, pure scoring, EventLog skeleton
- [ ] Phase 2 — Memory & Quarantine
- [ ] Phase 3 — Selector, Player, Strategist
- [ ] Phase 4 — Executor & Symbol health
- [ ] Phase 5 — AttributionLedger
- [ ] Phase 6 — Dashboards
- [ ] Phase 7 — Replay-validation
- [ ] Phase 8 — Shadow run
- [ ] Phase 9 — Cutover

## Важно

**Этот пакет не должен импортироваться из `panteon_runtime/` v1.** v1
работает в production, v2 проектируется без backward-compat ограничений.

После Фазы 7 (replay-validation) и Фазы 8 (shadow run) сделаем cutover.
