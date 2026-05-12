# Panteon v2 — Cutover Runbook

**Назначение.** Пошаговая операционная инструкция для перевода
production-торговли с `panteon_runtime/` (v1) на `panteon_v2/`.

**Аудитория.** Оператор, отвечающий за безаварийность боевых торгов.

---

## Pre-flight checklist (за 24 часа до cutover)

- [ ] Все unit-тесты v2 проходят: `cd src && python -m panteon_v2.tests.run_all`
- [ ] Phase 7 replay-validation на последних 5 v1-сессиях: `python -m panteon_v2.replay.cli --latest MEXC BITGET`
  - Проверка: Q1, Q2, Q4 PASS на всех сессиях
- [ ] Phase 8 shadow-run параллельно с боевым v1 минимум 24 часа
- [ ] AttributionLedger v2 ≈ stats.pnl v1 (расхождение < 1%)
- [x] Подготовлены реальные exchange-адаптеры:
  - [x] `src/panteon_v2/app/bitget_adapter.py` (v1 futures client -> v2 Exchange)
  - [x] `src/panteon_v2/app/mexc_adapter.py` (v1 futures client -> v2 Exchange)
- [ ] Карантинный seed-список одобрен (см. `Panteon._SEED_AGENT_QUARANTINE` v1)
- [ ] Все 5 PlayerProfile-ов одобрены (PRODUCTION_PROFILES в `bootstrap.py`)
- [ ] Backup persisted state v1: `cp panteon_runtime/state/memory.json /backup/v1_pre_cutover.json`
- [ ] Notification сделан: «cutover окно X-Y часов»
- [ ] Stop-loss уровни на бирже выставлены (защита на случай v2-бага)

---

## Cutover (T = 0)

### Шаг 1 — Закрыть все open-позиции v1 (T-15 минут)

```bash
# Через v1 CLI (или вручную на бирже)
python -m panteon_runtime.cli --close-all
# Проверка
python -m panteon_runtime.cli --status   # n_positions=0
```

**Acceptance:** `status.json[n_positions] == 0` для обоих exchange.

### Шаг 2 — Stop v1

```bash
# Грейсфул shutdown:
pkill -SIGTERM -f Start_BITGET.py
pkill -SIGTERM -f Start_MEXC.py
# Подождать 30 сек, проверить:
ps aux | grep Start_  # должно быть пусто
```

### Шаг 3 — Migrate state

```bash
mkdir -p panteon_v2_state
python -m panteon_v2.app.cli \
  --migrate-v1 panteon_runtime/state/memory.json \
  --to panteon_v2_state/perf_snapshot.json
```

**Acceptance:** в выводе `Migrated N labels, M pairs` где N ≥ 15 (число
известных агентов), M ≥ 50 (label × regime пар).

### Шаг 4 — Verify v2 bootstrap dry-run

```bash
python -m panteon_v2.app.cli --replay Results/MEXC/$(ls Results/MEXC | tail -1)
```

**Acceptance:**
- `Σ AttributionLedger.player_pnl ≈ status.pnl_pct` v1 (diff < 1%)
- `Q1/Q2/Q4 PASS` в выводе

### Шаг 5 — Start v2

```bash
# Стартуем production loop (с реальным exchange-adapter)
nohup python -m panteon_v2.app.cli --production \
  --bitget --mexc \
  --load-snapshot panteon_v2_state/perf_snapshot.json \
  > logs/panteon_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**Acceptance в течение 10 минут:**
- В лог-файле есть `BarStarted` events
- `n_signals` начинает расти (v2 принимает решения)
- Никаких `ERROR` или `Traceback`

### Шаг 6 — First-hour monitoring

| Метрика | Norm | Action если выходит за норму |
|---------|------|------------------------------|
| AttributionLedger.consistency_check | True | **STOP v2, rollback** (см. ROLLBACK.md) |
| failed_orders за час | ≤ 5 | Investigate symbol_health, возможно tune blocklist |
| open_positions count | ≤ MAX_OPEN_POSITIONS | RiskLimits сработает, но прверить настройки |
| EventLog.size | растёт каждый бар | Если не растёт — feed/network проблема |
| current_balance | в коридоре ±2% от initial | Если упал > 2%, **проверить leader-выбор** |

### Шаг 7 — Сравнение через час

```bash
# Сгенерировать дашборд:
python -c "
from panteon_v2.app.bootstrap import build_production_pipeline
from panteon_v2.dashboards import TextRenderer
# ... (загрузить current state)
print(TextRenderer().render_main(pipeline.renderer.build_main()))
"
```

Сверить с панелью v1 (если она запущена в read-only) на тех же 60 минутах:
- Σ realized PnL должен быть в одном порядке
- Карантинные не должны появляться в leader

---

## Post-cutover monitoring (24 часа)

### Часовая проверка

```bash
# Сводный дашборд + проверка консистентности:
python -m panteon_v2.app.cli --status
```

### Daily summary (через 24 часа)

Сравнить с baseline v1 за тот же период (если есть):

| Параметр | v1 baseline (~неделя) | v2 первые 24ч | OK если |
|----------|----------------------|---------------|---------|
| Real PnL% | средний x | y | y ≥ x − 0.5% |
| Win rate | 50–55% | z | z ≥ 50% |
| Failed orders / day | ~5 | w | w ≤ 5 |
| n trades / hour | ~3 | u | 1 ≤ u ≤ 6 |

Если все 4 пункта PASS — **v2 принят в production**.

---

## Acceptance criteria для подтверждения cutover

После 7 календарных дней непрерывной работы:

- [ ] Накопительный PnL v2 ≥ накопительного PnL v1 за тот же период истории
- [ ] Ни одного `Q1/Q2/Q4` violation в EventLog
- [ ] Ни одного crash main_loop (ловится через max_bars + on_error)
- [ ] AttributionLedger.consistency_check = True всегда
- [ ] Failed orders / total_orders ≤ 1%
- [ ] Карантин динамически обновлялся (есть события `QuarantineRecomputed` с непустыми added/removed)

Если ВСЕ ✓ — **cutover закрывается**, v1 архивируется.

Если хоть один ✗ — **rollback** (см. PANTEON_V2_ROLLBACK.md).

---

## Контакты в emergency

- Stop-trading hotline (биржевые позиции через UI)
- Backup operator
- v2 architect (для крит-багов)

---

## Логирование

В каждом шаге пишется JSON-line в `logs/cutover_YYYYMMDD.jsonl`:

```json
{"step": "1", "ts": "...", "result": "OK", "metric": {...}}
{"step": "2", "ts": "...", "result": "OK", "metric": {...}}
...
```

Это позволит post-mortem быстро восстановить картину.
