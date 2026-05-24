# Panteon Flash — повторный аудит и обновлённая стратегия

Дата: 2026-05-21 (re-audit относительно `PANTEON_FLASH_AUDIT_AND_IMPROVEMENTS.md` от той же даты).

Метод: построчное чтение текущего состояния `selection/flash_allocator.py`, `scoring/scoring.py`, `selection/voting.py`, `selection/player.py`, `selection/promotion_manifest.py`, `domain/types.py`, `selection/strategist.py`, `app/main_loop.py`, `app/agent_bootstrap.py`. Сравнение с 25 пунктами предыдущего отчёта. Дополнительно — поиск регрессий и новых проблем, возникших при правках.

Документ разбит на три блока:

1. сводка по каждому пункту прошлого отчёта (fixed / partial / open);
2. новые проблемы, появившиеся вместе с правками;
3. обновлённая стратегия и шаги дальнейших улучшений.

---

## 1. Сводка по 25 пунктам прошлого отчёта

| № | Проблема | Статус |
| --- | --- | --- |
| 1.1 | Разные «правды» в ranking и gate | Fixed |
| 1.2 | `regime_score` игнорирует параметр `regime` | Open |
| 1.3 | `regime_confidence` нигде не учитывается в Flash-скоринге | Partial |
| 1.4 | `_component_score` — арифметическое среднее без веса | Fixed |
| 1.5 | Inactivity penalty при `closed_trades > 0` | Fixed |
| 1.6 | WeightedConsensus: close-канал асимметричен open-каналу | Fixed |
| 1.7 | Tie-breaker смещён к `agent` против `ensemble` | Partial |
| 1.8 | `actor_signal_cap` режет по алфавиту, а не по score | Fixed |
| 1.9 | `actionable_bonus` не нормирован | Open |
| 1.10 | `NoTradePlayer` неэффективен как кандидат | Open |
| 1.11 | `EnsemblePlayer.last_vote_errors` мутирует не-frozen dataclass | Open |
| 1.12 | Overextension thresholds глобальны | Open |
| 1.13 | `Signal.price` может стать 0 | Fixed |
| 1.14 | Promotion gate только для `is_open` | По задумке, не баг |
| 1.15 | `actor_signal_cap` теряет оригинальный selected_actor в audit | Open |
| 1.16 | Разные ключи в shadow scores | Open |
| 1.17 | `regime_score` теряет монотонность при `closed_trades < 2` | Open |
| 1.18 | Solo-safety drop удаляет агента из любых ensemble | Open |
| 1.19 | Скоринг без вычета комиссий/funding/slippage | Open |
| 1.20 | Нет switch_margin / hysteresis в Flash | Open (есть только в legacy Strategist) |
| 1.21 | `AgentRegistry` не thread-safe | Open |
| 1.22 | `ActionFilterAgent` с пустым `allowed_actions` | Документировано, не критично |
| 1.23 | Promotion manifest с фиксированными порогами без статистики | Open |
| 1.24 | `main_loop.py` 3 657 строк → монолит | Open (сейчас 3 629) |
| 1.25 | Нет инварианта эквивалентности voting policy | Open |

Итог: 5 пунктов закрыты (1.1, 1.4, 1.5, 1.6, 1.8, 1.13), 2 частично (1.3, 1.7), 14 открыто. Семь из закрытых/частично закрытых — это P0/P1 в прежнем roadmap. Большинство P2-долгов остаётся.

### 1.1. Подробности по закрытым

**1.1 (gate ↔ ranking)**. `FlashCandidateAudit` теперь хранит `base_score`, `effective_score`, `gate_score` (см. `flash_allocator.py:141-143,166-168`). Гейт `min_score_to_trade` сравнивается с `gate_score` (`flash_allocator.py:499`). Ранжирование по `effective_score` (через поле `score` в audit, `flash_allocator.py:534`). Это унифицировано.

**1.4 (`_component_score`)**. Усреднение взвешенное по `closed_trades` (`flash_allocator.py:786-796`), `pnl_pct` суммируется. Самосогласованность с агрегированными метриками улучшилась.

**1.5 (inactivity penalty)**. Условие теперь: `closed_trades == 0 and signals == 0 and entries == 0` (`scoring.py:158`). Прибыльный неактивный актор больше не штрафуется.

**1.6 (close asymmetry)**. Введена `score_keep` (сумма весов агентов, не голосовавших за close), и close разрешён только при `score_close > score_keep` (`voting.py:130-149,172-173`). Симметрия восстановлена.

**1.8 (cap по алфавиту)**. `_apply_actor_signal_cap` сортирует `selectable` по `(actor, -score, symbol)` и режет лишние с конца (`flash_allocator.py:740-767`). Это устраняет произвольную потерю лучшего сигнала.

**1.13 (price=0)**. `_market_price_for_symbol` возвращает `None`, если цены нет или она не парсится. Когда цена недоступна, decision становится NoTrade с reason="missing_price" (`flash_allocator.py:556-567,1095-1103`). Тихая ошибка больше не проходит дальше.

### 1.2. Подробности по частично закрытым

**1.3 (`regime_confidence`)**. `StrategistConfig.min_regime_confidence` существует и блокирует решение legacy Strategist (`strategist.py:71,1131-1136`). В `FlashAllocator` и `regime_score` поле по-прежнему не учитывается. Поскольку production-путь — это Flash, проблема остаётся открытой на главном маршруте.

**1.7 (tie-breaker)**. Сортировка стала `(rejected, -score, -closed_trades, label, actor_type)` (`flash_allocator.py:531-538`). `actor_type` ушёл в конец, но всё равно даёт детерминированный bias: при полном равенстве всех первых 4 ключей побеждает «agent» как более ранний в алфавите. По факту такое равенство редкое, но если на пуле случаются ties между Solo-обёрткой и raw-агентом, исход теперь решается их `label`-ами. Стало менее заметно, но не нейтрально.

### 1.3. Что осталось не закрыто

P0-долг прошлого отчёта частично всё ещё актуален:

- **1.19 (net-PnL)** — скоринг использует gross PnL, fee/funding/slippage не вычитаются. Это самый важный неисправленный пункт. Любые улучшения ranking тонут в этом, если агент с большим turnover остаётся «лучшим» по gross.
- **1.20 (switch_margin в Flash)** — legacy Strategist имеет `switch_margin=0.30`, но FlashAllocator принимает решение каждый бар независимо. Churn-cost остаётся.
- **1.12 (overextension)** — пороги `±8%/12 баров` всё ещё глобальные.
- **1.23 (promotion manifest)** — пороги фиксированные, без Bayesian оценки.

P1/P2-долги (1.2, 1.10, 1.11, 1.15-1.18, 1.21, 1.24, 1.25) также открыты.

---

## 2. Новые проблемы, появившиеся вместе с правками

Расширение `FlashAllocatorConfig` принесло новые механизмы (актор-degradation guard, shadow handoff, base fallback, solo-wrapper preference). Каждый из них добавил свои дефекты.

### 2.1. `shadow_base_fallback_confirmation_enabled` обходит shadow-гейт через production-метрику

`flash_allocator.py:394-405`. Когда `shadow_actor_fallback_confirmation_enabled=True`, `shadow_base_fallback_confirmation_enabled=True`, `shadow_actor_fallback_min_base_score>0`:

```python
shadow = _ShadowConfirmation(
    score=base_score,
    closed_trades=metrics.closed_trades,
    winning_trades=metrics.wins,
    win_rate_pct=metrics.win_rate,
)
shadow_source = "base_fallback"
shadow_confirmed_by_base = True
```

После этого reject-проверка shadow_unconfirmed пропускается:

```python
elif (
    self._config.shadow_confirmation_enabled
    and not shadow_confirmed_by_base
    and shadow.closed_trades < self._config.shadow_confirmation_min_closed_trades
):
    rejected = True
    reason = "shadow_unconfirmed"
```

Это значит: достаточно иметь высокий `base_score` (production-метрика), и **сигнал проходит мимо shadow-валидации**. Идея, видимо, в том, чтобы зрелые agents не блокировались отсутствием shadow-данных. Но эффект тот же, что у удалённого гейта: новый агент с накопленной production-историей не обязан подтверждаться shadow’ом. Это противоречит документу проекта («экспериментальная идея должна сначала доказать полезность в shadow») и создаёт окно злоупотребления при добавлении нового агента сразу в production.

Что хуже, при `shadow_confirmed_by_base=True` фильтр `shadow_full_open_unconfirmed` и `shadow_quality_confirmation` всё ещё используют `shadow.closed_trades` и `shadow.win_rate_pct`, которые теперь равны production-метрикам. Audit покажет `shadow_score`, `shadow_closed_trades` равные `base_score`, `metrics.closed_trades` — поле названо `shadow_*`, а данные production. Это путаница в трассировке и потенциально опасное смешение источников.

### 2.2. Логическое противоречие в `actor_fallback_allowed`

`flash_allocator.py:366-374`:

```python
actor_fallback_allowed = (
    base_score >= fallback_min_base_score
    and shadow.score >= self._config.shadow_confirmation_min_score
    if fallback_min_base_score > 0.0
    else shadow.score > self._config.shadow_confirmation_min_score
)
```

Это тернар `(A and B) if cond else C`. Когда `fallback_min_base_score > 0`, условие включает `>= min_score`. В альтернативной ветке — строго `> min_score`. Граница при `shadow.score == 0` теперь зависит не от логики, а от того, выставлен ли `fallback_min_base_score`. Это асимметрия по unrelated параметру — источник неинтуитивного поведения.

Кроме того, dataclass валидирует `shadow_actor_fallback_min_base_score >= 0`, а не `>= 0` строго. При значении 0 ветка else тоже не запустит fallback (требует `shadow.score > 0`, что при пустом shadow всегда false). Эффективно `fallback_min_base_score=0` отключает fallback. Это не очевидно из API.

### 2.3. `shadow_signal_handoff_enabled` пускает shadow-сигналы в production без safety-фильтров

`flash_allocator.py:932-1000`. При включённом флаге `_shadow_signal_outputs` добавляет в `outputs` сигналы из `shadow_player_signals` и `shadow_agent_signals`. Эти outputs затем идут через тот же `_decide_symbol`, но:

- Нет проверки `live_trading_eligible` или `shadow_only` (агент мог быть зарегистрирован как shadow-only — но shadow signal от него теперь может попасть в production).
- `_score_actor` запросит метрики у `PerformanceMemory.get(label)`. Если для shadow-only метки ничего нет, `_component_score`/`no_data_score` вернёт 0 (или взвешенно по компонентам). При `min_score_to_trade=0` сигнал может пройти.
- Promoted gate проверяется по `_is_signal_promoted` — то есть нужен явный promotion. Это страхует. Но если promotion включён со слабыми порогами, shadow-handoff будет идти в production по той же цепочке.

Подытоживая: shadow handoff устраняет «mostly safe» principle документации. Нужны явные гарантии (метка `shadow_only`, отдельный `min_score_to_trade_for_handoff`, обязательный promotion).

### 2.4. `prefer_solo_player_wrappers_enabled` подавляет raw-агента безусловно

`flash_allocator.py:596-636`. При `prefer_solo_player_wrappers_enabled=True` любой raw-agent, у которого где-то существует `Solo_<label>` (в players, shadow_player_signals, actionable_labels), удаляется из `agent_rows`:

```python
if self._config.prefer_solo_player_wrappers_enabled:
    return set(solo_by_raw)
```

Это значит: достаточно один раз появиться Solo wrapper-у в shadow-канале — и raw агент исчезает из production без оглядки на его собственные метрики. Если raw агент сильнее своей Solo-обёртки (например, Solo сужает действия и теряет часть edge), он перестанет быть кандидатом. Контроль через `prefer_proven_solo_player_wrappers_enabled` опциональнее, но он включается отдельным флагом и не является default-ом.

### 2.5. `prefer_proven_solo_player_wrappers_enabled` тихо отключается при пустых Solo-метриках

`flash_allocator.py:638-660`. Если у Solo wrapper-а нет `has_data` или `closed_trades < min_closed_trades_to_trade`, функция возвращает `False` — wrapper не «outscores» raw, suppression не происходит. Это разумно по сути, но непрозрачно: один день wrapper мог подавлять raw, на следующий — нет. Это создаёт нестабильный пул кандидатов, который сложно объяснить ретроспективно. Нет audit-события «raw_agent_suppressed_by_solo».

### 2.6. `flash_actor_degraded` не имеет recovery

`main_loop.py:1369-1407,1435-1442`. `degraded_actors` пополняется, когда `actor_bucket` (последние `window` закрытий по актору) суммарно ≤ threshold. Из кода **не существует пути снятия actor с degraded set**. Однажды попавший туда актор останется заблокированным на весь run.

В случае signal-key degradation тот же баг присутствовал и раньше: degraded_signal_keys тоже не очищается. Сейчас он расширен ещё на уровень акторов. Реальные торговые системы должны давать акторам «реабилитацию» по окнам.

### 2.7. `degradation_actor_guard` режет в большем масштабе, чем индикация деградации

Actor degraded → блокированы **все** signal keys этого актора на **всех** символах. Если у актора 5 signal keys, и один из них стабильно убыточен, банится не он один, а весь актор. Это резкая реакция, особенно при отсутствии recovery (см. 2.6).

### 2.8. `_shadow_signal_outputs` не нормализует `Signal.id`

`flash_allocator.py:1029-1035`. `clean_signal` берёт исходный shadow signal через `replace(...)`, не сбрасывая `id`. Затем в `_decide_symbol` происходит `signal = replace(signal, id=signal_id, ...)`, что нормализует. То есть на данном этапе ID будет переписан. Это работает, но в audit для отклонённых shadow-сигналов остаётся непереписанный id из shadow канала — потенциальная коллизия с production-сигналами в attribution.

### 2.9. `_apply_actor_signal_cap` теперь дважды итерирует decisions, но сохраняет ранний side-effect

`flash_allocator.py:730-738`. Первый проход накапливает reservation counts для NoTrade/degraded; второй проход сортирует selectable и режет. В первом проходе `continue` — то есть NoTrade-решение оставляется как есть (`out = list(decisions)`). Это корректно, но при `degradation_reserve_actor_cap=True` reservation count может «съесть» cap для актора, который не имеет права на сигналы (deny по degraded_signal). В итоге cap-сценарий выглядит как «cap исчерпан», но на деле «исчерпан» degraded reservation. Очень тонкая семантика, документированная только кодом.

### 2.10. `actionable_bonus` теперь напрямую добавляется к `gate_score`

`flash_allocator.py:408-410`. `gate_score = effective_score`. Это значит, если actor в `actionable`, гейт `min_score_to_trade` понижается на `actionable_bonus` (по факту). Документация ожидает, что `actionable_bonus` — это ranking-бонус, не гейт-бонус. Сейчас обе функции склеены. Если задумано — фиксировать в spec; если нет — выделить отдельно `gate_actionable_bonus`.

### 2.11. Конфигурационная сложность взлетела

`FlashAllocatorConfig` теперь имеет 30+ полей с тонкими взаимозависимостями: `shadow_confirmation_enabled` × `shadow_symbol_confirmation_enabled` × `shadow_actor_fallback_confirmation_enabled` × `shadow_base_fallback_confirmation_enabled` × `shadow_actor_fallback_min_base_score` × `shadow_quality_confirmation_enabled` × `prefer_solo_player_wrappers_enabled` × `prefer_proven_solo_player_wrappers_enabled` × `proven_solo_min_score_advantage` × `degradation_guard_enabled` × `degradation_actor_guard_enabled` × `degradation_reserve_actor_cap` × `shadow_signal_handoff_enabled` × `promotion_manifest_enabled`.

Это конфигурационный взрыв. Каждая комбинация — это отдельное поведение, и тестов на пересечения нет. Высока вероятность, что в production будет жить непреднамеренная комбинация (по типу: handoff включён, promotion отключён, base_fallback включён → новый shadow-only актор уходит в production).

---

## 3. Обновлённая стратегия и шаги дальнейших улучшений

Текущая правка закрыла часть базовых дефектов прошлого отчёта, но добавила новые рычаги, которые в плохой комбинации усиливают риск (shadow-handoff, base-fallback, solo-suppression). Дальнейшая стратегия должна сначала завершить P0-долг прошлого отчёта, потом упорядочить новые механизмы, и только потом строить вертикаль улучшений (LCB-ranking, Bayesian promotion, soft voting).

### 3.1. Фиксы первого приоритета (P0)

1. **Net-PnL в скоринге**. `PerformanceMemory` должен возвращать `pnl_net_pct = pnl_gross_pct - fees - funding - slippage`. `regime_score` использует net. Без этого все остальные правки усреднены fee-bleed-ом.
2. **Switch margin в FlashAllocator**. По образу `StrategistConfig.switch_margin=0.30` ввести `FlashAllocatorConfig.actor_switch_margin`. Прошлогодний выбор актора по символу удерживается, пока новый кандидат не лучше на margin.
3. **Recovery для degraded actors/signals**. Добавить `recovery_window_bars` и `recovery_pnl_threshold`. После N баров без сделок или после положительной серии — выходить из degraded set. Без этого degraded → permanent ban.
4. **Per-symbol overextension**. Вместо глобальных `±8%/12 баров` — z-score по ATR_lookback. Параметры калибровать walk-forward.
5. **Закрытие shadow-handoff байпасса**. Запретить `_shadow_signal_outputs` для агентов с `live_trading_eligible=False` или `shadow_only=True`. Добавить отдельный `min_handoff_score_to_trade`, превышающий обычный гейт.
6. **Сделать `shadow_base_fallback` опциональным per-actor**. Базовый fallback должен применяться только к меткам в списке `allow_base_fallback_actor_keys`. Иначе любой агент с высоким production-score обходит shadow gate.
7. **Снять асимметрию `>=` vs `>` в `actor_fallback_allowed`**. Унифицировать на `>= min_score` или строго `> min_score`.

### 3.2. Фиксы второго приоритета (P1)

1. **Audit «raw_suppressed_by_solo»**. Каждый раз, когда `_suppressed_agent_labels_for_solo_players` исключает raw-агента, emit-ить событие с причиной.
2. **Original selected actor в audit**. При срабатывании `actor_signal_cap` или `missing_price` сохранять `original_selected_actor` в FlashDecision (новое поле). Сейчас оно перетирается на NoTrade.
3. **Регламент конфигурации**. Заморозить минимум 3 опубликованных preset’а: `safe`, `default`, `aggressive`. Сделать invariant-тесты на каждый preset.
4. **`regime_confidence` в скоринге Flash**. Умножить `effective_score` на `min(1.0, regime_confidence / floor)` или повысить `min_score_to_trade` на низкой уверенности.
5. **`actionable_bonus` отделить от gate_score**. `gate_score = base_score` (или blend); `effective_score = gate_score + actionable_bonus`.
6. **Bayesian promotion gate**. Заменить пороги по win_rate/closed_trades на LCB-границу из Beta-апостериора. На малых выборках алгоритм будет блокировать promotion даже при высоких эмпирических win_rate.
7. **Concentration penalty / Herfindahl**. На уровне `_apply_actor_signal_cap` после greedy выбора применять штраф score за уже занятые слоты одним актором.

### 3.3. Фиксы третьего приоритета (P2)

1. **LCB-ranking**. Score = `mean − k * std / √n` per signal key. Снимает selection bias.
2. **Soft-voting**. Расширить контракт агента до `(action, confidence, expected_holding_bars, expected_cost_bps)`.
3. **Decomposition `main_loop.py`**. Вынести real/shadow/flash/fallback в отдельные модули с тестами.
4. **Thread safety**. AgentRegistry / `_flash_degraded_*` под lock либо immutable per bar.
5. **`regime` в `regime_score`**. Per-regime коэффициенты (`dict[Regime, ScoringConfig]`).
6. **Test invariants на voting policies**. `WeightedConsensus == RiskParity(equal vols)` при равных весах и т.п.

### 3.4. Регламент дальнейших правок

1. **Любая новая опция в `FlashAllocatorConfig`** должна сопровождаться:
   - default=False/0 (opt-in);
   - unit-тестом на каждую новую ветку фильтрации;
   - integration-тестом на пересечение с уже существующими опциями;
   - записью в audit с явным reason при срабатывании.
2. **Любой shortcut, обходящий gate** (shadow handoff, base fallback) — только через explicit allowlist меток.
3. **Любое подавление кандидатов** (solo suppression, degraded set) — с audit-событием и recovery-механизмом.

### 3.5. Контракт «Пантеон не хуже своих составляющих»

Из прошлого отчёта остаётся актуальным: на каждом walk-forward слое должно выполняться

```
equity(Panteon) >= equity(best_standalone) * 0.95   (после net-pnl)
churn(Panteon) <= 2 * churn(best_standalone)
PanteonAdvantage = (pnl_panteon_net - pnl_best_standalone_net) / max_dd_panteon > 0
```

Без net-PnL и без switch_margin эти инварианты сейчас не достижимы методически. Поэтому P0-долг (3.1.1 и 3.1.2) — необходимое условие для любого дальнейшего разговора о «не хуже». Остальные улучшения (LCB, Bayesian gate, soft voting) — это уже шаги к «лучше».

### 3.6. Приоритизированная очередь работ

| Очередь | Задача | Размер | Ожидаемый эффект |
| --- | --- | --- | --- |
| 1 | Net-PnL в скоринге | M | Снимает иллюзию edge на high-turnover |
| 2 | Switch margin в FlashAllocator | S | Уменьшает churn cost |
| 3 | Recovery для degraded actors/signals | S | Снимает permanent ban риск |
| 4 | Закрытие shadow handoff байпасса (live_trading_eligible) | S | Перекрывает основной риск 2.3 |
| 5 | `shadow_base_fallback` через allowlist | S | Перекрывает риск 2.1 |
| 6 | Унификация `>=` vs `>` в actor_fallback_allowed | XS | Убирает hidden inconsistency |
| 7 | Original actor в audit при cap/missing_price | XS | Улучшает диагностику |
| 8 | Per-symbol overextension через ATR | M | Меньше ложных блокировок |
| 9 | Bayesian promotion | M | Корректная статистическая значимость |
| 10 | `regime_confidence` в Flash scoring | S | Меньше торговли на неуверенных переходах |
| 11 | LCB-ranking | M | Устраняет selection bias |
| 12 | Concentration penalty | M | Снижает correlated drawdown |
| 13 | Soft voting (agent confidence) | L | Унификация ensemble и Flash |
| 14 | Decomposition main_loop.py | L | Тестируемость |

---

## 4. Краткое резюме

- Из 25 пунктов прошлого отчёта закрыты 5, частично — 2, открыто — 18.
- Все закрытые пункты относятся к локальным фиксам (gate vs ranking, weighted component score, close asymmetry, cap by score, price=0, inactivity penalty).
- P0-долг сохранён (net-PnL, switch_margin, overextension, promotion gate).
- Новые опции (`shadow_handoff`, `base_fallback`, `prefer_solo_wrappers`, `degradation_actor_guard`) принесли свои дефекты: возможный обход shadow-валидации, подавление raw-агентов без audit, permanent ban деградировавших акторов.
- Конфигурационная сложность взлетела до уровня, на котором необходим explicit preset-режим, иначе production-комбинации станут источником тихих ошибок.
- Без net-PnL и switch_margin контракт «Пантеон не хуже своих составляющих» методически недостижим — это блокирующие задачи перед любыми вертикальными улучшениями.

Следующий разумный шаг — закрыть пункты 1-7 очереди из 3.6 (всё это XS/S/M, кроме net-PnL) и зафиксировать `safe` preset со всеми experimental opt-in флагами в false. Это вернёт системе предсказуемое поведение до того, как стартует второй виток улучшений.
