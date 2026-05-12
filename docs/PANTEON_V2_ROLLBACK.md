# Panteon v2 — Rollback Plan

**Назначение.** Безопасный возврат на `panteon_runtime/` (v1) если v2
проявил себя плохо.

**Trigger condition (любое из):**
- v2 произвёл losses > 2% капитала за 1 час
- ≥ 3 непрерывных bar-ов с `Q2 violation` (consistency_check = False)
- Карантинный лейбл попал в leader (`Q1 violation`)
- main_loop crash >= 2 раза за 10 минут
- Биржа отклонила ≥ 30% ордеров за час

---

## Rollback steps (выполнить ВСЕ, в порядке)

### Шаг R1 — Emergency stop v2 (T+0, сразу)

```bash
# Грейсфул:
pkill -SIGTERM -f panteon_v2.app.cli
sleep 30
# Если не помогло — kill:
pkill -SIGKILL -f panteon_v2.app.cli
```

**Acceptance:** `ps aux | grep panteon_v2` пусто.

### Шаг R2 — Закрыть все v2 open-позиции вручную (T+1 мин)

Через UI биржи или API скрипт:
```bash
python -c "
from panteon_runtime.exchange_api_runtime import MexcDirectClient
c = MexcDirectClient(...)
for p in c.get_all_positions():
    c.close_position(p['symbol'])
"
```

**Acceptance:** Биржа подтверждает 0 open-позиций по всем символам.

### Шаг R3 — Восстановить v1 state из backup (T+3 мин)

```bash
cp /backup/v1_pre_cutover.json panteon_runtime/state/memory.json
```

### Шаг R4 — Запустить v1 (T+5 мин)

```bash
nohup python panteon_runtime/Start_BITGET.py > logs/v1_rollback_bitget.log 2>&1 &
nohup python panteon_runtime/Start_MEXC.py > logs/v1_rollback_mexc.log 2>&1 &
```

**Acceptance в течение 5 минут:**
- v1 пишет `[REAL Panteon] портфолио_value=...` в trading.log
- Никаких `Traceback` в первые 100 строк лога

### Шаг R5 — Сохранить v2-логи для post-mortem (T+10 мин)

```bash
mkdir -p /backup/v2_postmortem_$(date +%Y%m%d_%H%M%S)
cp -r logs/panteon_v2_*.log /backup/v2_postmortem_*/
cp -r panteon_v2_state/ /backup/v2_postmortem_*/
# EventLog (если включён JSONL):
cp logs/v2_events.jsonl /backup/v2_postmortem_*/
```

### Шаг R6 — Notification

Сообщить:
- «Cutover ROLLBACK выполнен в HH:MM, причина: [X]»
- «v1 работает с T+5 мин»
- «v2 logs сохранены в /backup/v2_postmortem_*/»

---

## Post-rollback (в течение 24 часов)

### Анализ причины

1. Открыть `EventLog` v2 (последние 100 events перед сбоем):
   ```bash
   tail -100 logs/v2_events.jsonl | jq .
   ```

2. Прогнать через replay-validator:
   ```bash
   # подготовить «псевдо-сессию» из v2 events
   # ...
   python -m panteon_v2.replay.cli /backup/v2_postmortem_*/synth_session/
   ```

3. Проверить какой Q-чек упал и почему:
   - **Q1 violation:** `QuarantineManager.recompute` сработал но Selector
     не отфильтровал → проверить sequence событий с тем же `trace_id`
   - **Q2 violation:** `AttributionLedger` не парит open/close — возможно
     race condition между threads (но v2 single-thread, не должно быть)
   - **Q4 violation:** Strategist пропустил sanity check — проверить
     PlayerComposer корректность

### Tick-list для следующего cutover attempt

- [ ] Корневая причина выявлена
- [ ] Unit-тест воспроизводящий проблему добавлен
- [ ] Фикс реализован, тест проходит
- [ ] Все 310 + новые тесты PASS
- [ ] Phase 7 replay-validation проходит на постмортем-сессии
- [ ] Phase 8 shadow-run 48 часов без incidents
- [ ] Pre-flight checklist пройден заново

---

## Что НЕ ДЕЛАТЬ при rollback

- ❌ **Не запускать v2 повторно** «посмотреть»: это может усугубить ситуацию
- ❌ **Не отключать v1 до confirmation** что v1 действительно стартанул
- ❌ **Не редактировать `memory.json` вручную** — могут быть несогласованности
- ❌ **Не игнорировать post-mortem analysis** — корневая причина обязана
  быть найдена до повторного cutover

---

## Архивация

Если v1 успешно работает 7 дней после rollback и анализ закрыт:
- v2 logs архивируются в read-only хранилище
- v1 продолжает торговать
- Cutover откладывается до следующей итерации
