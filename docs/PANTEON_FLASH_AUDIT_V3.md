# Panteon Flash — третий аудит и план следующего витка

Дата: 2026-05-21 (re-audit относительно `PANTEON_FLASH_AUDIT_V2.md`).

Метод: построчное чтение свежих версий `selection/flash_allocator.py` (769 строк, было 789), `selection/voting.py`, `selection/player.py`, `selection/promotion_manifest.py`, `scoring/scoring.py`, `domain/types.py` (361 строк, было 353), `memory/performance.py`, `memory/degradation.py`, `selection/strategist.py`, `selection/selector.py`, `selection/composer.py`, `app/main_loop.py` (3 618 строк, было 3 629), `app/agent_bootstrap.py`, тестового подкаталога `panteon_v2/tests/`. Сверка с 25 пунктами первого отчёта и со списком новых рисков из V2.

Документ разделён на четыре блока: что закрылось за последний цикл, что осталось, какие новые проблемы возникли вместе с правками, и обновлённая стратегия дальнейших улучшений.

---

## 1. Что закрылось за период между V2 и сейчас

Хорошие новости: семь P0/P1-пунктов из V2 закрыты осмысленно и в нужных местах.

### 1.1. Net-PnL интегрирован сквозь PerformanceMemory и scoring

`memory/performance.py:457-473`:

```python
gross_pnl = return_pct * self._trade_fraction
fee_total_pct = (...) * self._trade_fraction
funding_total_pct = (...) * self._trade_fraction
net_pnl = gross_pnl - fee_total_pct - funding_total_pct
state.pnl_pct += net_pnl_pct           # cumulative NET
state.pnl_gross_pct += gross_pnl * 100.0
state.fee_pct += fee_total_pct * 100.0
state.funding_pct += funding_total_pct * 100.0
```

`Metrics` теперь хранит `pnl_gross_pct`, `fee_pct`, `funding_pct` (`domain/types.py:297-299`), а `pnl_net_pct` и `trading_cost_pct` — read-only computed properties (`types.py:336-343`). Поле `Metrics.pnl_pct` де-факто является net — это согласовано с `_record_close`. `regime_score` использует `metrics.pnl_pct`, то есть net автоматически. FlashCandidateAudit публикует все четыре числа (`flash_allocator.py:192-196, 225-229, 611-615`). Это закрывает 1.19.

### 1.2. Switch margin в FlashAllocator

Добавлены: поле `actor_switch_margin` в config (`flash_allocator.py:60`), параметр `previous_actor_by_symbol` в `decide()` (`flash_allocator.py:332,399`), логика удержания предыдущего актора (`flash_allocator.py:652-675`):

```python
if self._config.actor_switch_margin > 0.0 and previous_actor_key:
    previous = next((row for row in ranked if not row.rejected and ...), None)
    if (previous is not None
        and previous.actor_key != selected.actor_key
        and selected.score - previous.score < self._config.actor_switch_margin):
        selected = previous
        selected_reason = "switch_margin_hold"
```

Интеграция в main_loop (`main_loop.py:907-944`) корректно поддерживает мапу «symbol → previous actor_key» и обновляет её после исполнения. Поле сохраняется только когда сигнал реально вышел и `selected_actor != "NoTrade"`. Это закрывает 1.20.

### 1.3. Recovery для degraded set

`flash_allocator.py:86-88,167-173`: новые конфиг-поля `degradation_recovery_enabled`, `degradation_recovery_min_closed_trades`, `degradation_recovery_min_recent_pnl_usd`. Валидация: recovery_min_closed ≤ window.

В `main_loop.py:1383-1451`:

```python
elif (
    recovery_enabled
    and key in degraded
    and _flash_degradation_recent_pnl(bucket, recovery_min_closed) >= recovery_threshold
):
    degraded.discard(key)
```

То же для actor-level (`degraded_actors.discard(actor_key)`). Это снимает V2-риск 2.6 (permanent ban) — деградировавший актор/сигнал может выйти из карантина при накоплении положительных результатов.

### 1.4. Per-symbol overextension через z-score

Добавлены: `overextension_volatility_normalized_enabled`, `short_overextension_z_floor`, `long_overextension_z_ceiling`, `overextension_min_volatility_pct` (`flash_allocator.py:75-78`), `lookback_volatility_pct` в `MarketSnapshot` (`domain/types.py:179`). Логика (`flash_allocator.py:832-867`) выбирает z-score только если включено и volatility доступна; иначе fallback на старые абсолютные пороги. Валидация знаков порогов в `__post_init__`. Audit хранит `lookback_volatility_pct` и `lookback_return_z`. Это закрывает 1.12.

### 1.5. `shadow_base_fallback` через allowlist

Новое поле `shadow_base_fallback_actor_keys` (`flash_allocator.py:59`). Метод-фильтр (`flash_allocator.py:795-799`):

```python
def _is_base_fallback_allowed(self, output: _ActorSignal) -> bool:
    allowed = set(self._config.shadow_base_fallback_actor_keys)
    if not allowed:
        return False
    return output.actor_key in allowed or output.label in allowed
```

Используется в _decide_symbol (`flash_allocator.py:471-482`) — base_fallback срабатывает только если actor_key/label явно перечислены. Безопасный default (пустой → fallback off даже при flag=true). Это закрывает основной V2-риск 2.1.

### 1.6. `regime_confidence` в скоринге Flash

`flash_allocator.py:484-486`:

```python
gate_score = shadow.score if self._config.shadow_confirmation_enabled else base_score
gate_score *= _regime_confidence_scale(market)
effective_score = gate_score
```

`_regime_confidence_scale` clamps в [0, 1] (`flash_allocator.py:1261-1271`). Это даёт пропорциональное снижение score на неуверенных регулярных переходах. Закрывает 1.3 на главном маршруте.

### 1.7. `shadow_signal_handoff` ограничен известными actor_keys

`flash_allocator.py:1093-1174`. Введён `known_actor_keys` = объединение всех actor_keys из реальных agent/player rows. Shadow signal проходит только если его `actor_key in known_actor_keys` (`flash_allocator.py:1124,1157`). Для Solo-обёрток дополнительно проверяется родительский raw agent (`_known_solo_wrapper_label`). Закрывает V2-риск 2.3 (shadow-only сигналы в production).

### 1.8. `original_selected_actor` сохраняется в audit

`flash_allocator.py:260-261,690-712,913-918`. `FlashDecision` теперь имеет два дополнительных поля; при срабатывании `actor_signal_cap` или `missing_price` оригинал не теряется. Это закрывает V2-риск 2.x и 1.15.

### 1.9. `actor_fallback_allowed` упрощён

`flash_allocator.py:440-448`:

```python
actor_fallback_base_allowed = (
    base_score >= fallback_min_base_score
    if fallback_min_base_score > 0.0
    else True
)
symbol_shadow_not_bad = (
    shadow.closed_trades <= 0
    or shadow.score >= self._config.shadow_confirmation_min_score
)
```

Прежняя асимметрия `>=` vs `>` и спутанный тернар устранены. Закрывает V2-2.2.

Итого: из 18 пунктов, остававшихся открытыми после V2, закрыто 7 (1.3, 1.12, 1.15, 1.19, 1.20, плюс V2-2.1, V2-2.2, V2-2.3, V2-2.6). Это самый продуктивный цикл с начала аудита.

---

## 2. Что осталось открытым

| № | Проблема | Источник | Статус |
| --- | --- | --- | --- |
| 1.2 | `regime` не используется в `regime_score` | первый отчёт | Open |
| 1.7 | Tie-breaker по `label`/`actor_type` | V2 уже частично | Open (label-driven, малозаметно) |
| 1.9 | `actionable_bonus` не нормирован | первый отчёт | Open |
| 1.10 | `NoTradePlayer` не первоклассный кандидат | первый отчёт | Open |
| 1.11 | `EnsemblePlayer.last_vote_errors` мутирует | первый отчёт | Open |
| 1.16 | Разные ключи в shadow scores | первый отчёт | Open |
| 1.17 | `regime_score` теряет монотонность при `closed<2` | первый отчёт | Open |
| 1.18 | Solo-safety drop удаляет агента из ensemble | первый отчёт | Open |
| 1.21 | `AgentRegistry` без блокировки | первый отчёт | Open |
| 1.22 | `ActionFilterAgent` с пустым `allowed_actions` | первый отчёт | Документировано, не критично |
| 1.23 | Promotion manifest без Bayesian-гейта | первый отчёт | Open |
| 1.24 | `main_loop.py` монолит | первый отчёт | Open (3618 строк) |
| 1.25 | Нет invariant-тестов voting policies | первый отчёт | Open |
| V2-2.4 | `prefer_solo_player_wrappers_enabled` без proven по умолчанию подавляет raw | V2 | Open (есть `prefer_proven`, но это opt-in) |
| V2-2.5 | Нет audit-события «raw_suppressed_by_solo» | V2 | Open |
| V2-2.7 | `degradation_actor_guard` режет всего актора на всех символах | V2 | Partial (есть recovery, но granularity та же) |
| V2-2.8 | Shadow signal ID нормализация | V2 | Fixed косвенно: clean_signal теперь явно `replace(..., id=0, ...)` (`flash_allocator.py:1218-1225`) |
| V2-2.10 | `actionable_bonus` склеен с gate | V2 | Partial — теперь gate использует только base/shadow score, но actionable добавляется к effective; новая проблема ниже |
| V2-2.11 | Конфигурационный взрыв | V2 | Hardened: 40+ полей |

Несколько комментариев к статусам.

**1.7 (tie-breaker)**. Сортировка `(rejected, -score, -closed_trades, label, actor_type)` — `actor_type` стал последним, но при равенстве остальных bias остаётся: `agent < ensemble` алфавитно. Для прод это нюансово, потому что после введения net-PnL и confidence-масштабирования полные ties редки.

**1.10 (NoTradePlayer)**. `NoTradePlayer.vote()` возвращает пустой список. `_player_outputs` собирает Action.HOLD для всех символов, и `_decide_symbol` отбрасывает row как `reason="inactive"`. То есть как кандидат NoTradePlayer декоративный. Решение «не торговать» по-прежнему берётся косвенно — когда никто не прошёл фильтры. Можно сделать NoTrade полноценным кандидатом с собственным positive score (например, `expected_cost_savings_pct` исходя из ожидаемых fees при сделке), но это пока не сделано.

**1.11 (EnsemblePlayer.last_vote_errors)**. Поле `last_vote_errors: List[VoteError]` мутируется внутри `vote()`. Класс `@dataclass` (не frozen). Эта мутация делает Player не pure и не thread-safe. Если кто-то запустит vote() в двух потоках на одном экземпляре, состояние гонка. Сейчас main_loop вызывает одиночно — баг латентный.

**1.23 (Promotion manifest)**. `PromotionManifestConfig` фиксированные пороги (`min_full_closed_trades=50`, `min_win_rate_pct=52.0`). Никакой статистической значимости (LCB/Beta-posterior). Этот фронт остался непатченным.

**1.24 (main_loop)**. Размер уменьшился на 39 строк за два цикла. Архитектурного декомпозинга не было.

---

## 3. Новые проблемы, возникшие вместе с правками

Семь новых рисков от V2 цикла. Они мягче, чем те, что закрыты, но при некомфортных конфигурациях могут стрелять.

### 3.1. `regime_confidence=0` обнуляет гейт min_score_to_trade

`flash_allocator.py:484-485,588-590`. При `regime_confidence=0`:

```
gate_score = score * 0 = 0
...
elif gate_score < self._config.min_score_to_trade:  # 0 < 0 → False
```

Default `min_score_to_trade=0`. То есть на бар, где регулятор режимов абсолютно неуверен (`regime_confidence=0`), порог по гейт-скору **не блокирует никого**. Другие фильтры могут отсечь (нет данных, quarantined, и т. д.), но именно score-гейт деградирует до бесполезности.

Реальная стоимость зависит от того, насколько часто регулятор выдаёт `regime_confidence=0`. В тестах это, скорее всего, не сценарий, в проде — может появляться на переходах режимов.

**Фикс**: использовать `gate_score <= self._config.min_score_to_trade` либо отдельный happy-path: при `regime_confidence < min_confidence_floor` → NoTrade с reason="regime_undefined".

### 3.2. `actionable_bonus` обходит масштабирование regime_confidence

`flash_allocator.py:484-488`:

```python
gate_score *= _regime_confidence_scale(market)
effective_score = gate_score
if output.label in actionable:
    effective_score += self._config.actionable_bonus
```

Bonus прибавляется после умножения на confidence. На confidence=0 actor с `actionable=True` получает effective_score=0.25 (полный bonus), у остальных — 0. То есть actionable-список доминирует ранжированием на низкой уверенности.

**Фикс**: умножать `actionable_bonus` на тот же confidence:
```python
effective_score = gate_score + self._config.actionable_bonus * _regime_confidence_scale(market)
```

### 3.3. Switch margin не запоминает оригинал при cap-обрезке

`main_loop.py:937-944`:

```python
pipeline._flash_previous_actor_by_symbol = {
    str(...): _flash_actor_key_from_decision(decision)
    for decision in decisions
    if getattr(decision, "signal", None) is not None
    and str(getattr(decision, "selected_actor", "") or "") != "NoTrade"
}
```

После `_apply_actor_signal_cap` отрезанные сигналы становятся `selected_actor="NoTrade"` с `original_selected_actor=<name>`. Маппа `_flash_previous_actor_by_symbol` их пропускает. Это значит: на bar N актор A был выбран на BTC, но cap=1 и AVAX тоже у него — BTC обрезан → previous_actor_by_symbol[BTC] не запомнит A. На bar N+1 нет previous → switch_margin не сработает. На практике это уменьшает hysteresis на cap-задавленных символах.

**Фикс**: записывать `previous_actor_key` по `original_selected_actor` тоже, либо построить функцию-обёртку, которая использует `original_selected_actor or selected_actor`.

### 3.4. Recovery бар-чувствителен к `recovery_min_closed`

`main_loop.py:1462-1467`:

```python
def _flash_degradation_recent_pnl(bucket: object, closed_trades: int) -> float:
    values = [float(value) for value in list(bucket or ())]
    closed = max(1, int(closed_trades or 1))
    if len(values) < closed:
        return float("-inf")
    return sum(values[-closed:])
```

Если в bucket меньше `recovery_min_closed` элементов → `-inf` → recovery не сработает. При `recovery_min_closed > window` валидатор в `__post_init__` поднимает ошибку, так что эта ситуация не пройдёт. Но при `recovery_min_closed == window` recovery требует, чтобы все последние `window` сделок суммарно ≥ threshold — то есть полное вытеснение деградировавших сделок. Это очень медленное восстановление, особенно при малых `window=3`.

Не баг, но дизайн: recovery медленнее, чем дегра­дация. Имеет смысл задокументировать, что recovery — это «через `window` свежих сделок».

### 3.5. `shadow_full_open_unconfirmed` фильтр не учитывает `shadow_confirmed_by_base`

`flash_allocator.py:535-544`:

```python
elif (
    self._config.shadow_confirmation_enabled
    and output.action.is_open
    and output.action.is_fraction_full
    and self._config.shadow_confirmation_min_full_open_closed_trades > 0
    and shadow.closed_trades
    < self._config.shadow_confirmation_min_full_open_closed_trades
):
    rejected = True
    reason = "shadow_full_open_unconfirmed"
```

Когда сработал base_fallback, `shadow.closed_trades = metrics.closed_trades`. Если у production-метки много закрытых сделок (>full_open_min), это нормально, фильтр пропускает. Но в audit reason при отклонении неточный: `shadow_*` в reason, а данные production. Это полу-баг 2.1 из V2: путаница источника. Сейчас сужена, но не полностью.

### 3.6. `_metrics_delta` (DegradationGate) не передаёт net-PnL поля

`memory/degradation.py:217-238`. Делается `Metrics(pnl_pct=..., closed_trades=..., wins=..., losses=...)` — без `pnl_gross_pct/fee_pct/funding_pct`. После внедрения net-PnL это значит, что session-delta всегда имеет gross=0/fee=0/funding=0. В DegradationGate эти поля не используются, но если кто-то в будущем построит отчёт по net на session-delta — получит мусор.

Также — потенциальная ловушка: `wins+losses` берётся `max(0, current-baseline)`, и если baseline и current развалятся (baseline 5 closed/3 wins/0 losses; current 8 closed/5 wins/3 losses) → delta closed=3, wins=2, losses=3 → `wins+losses=5 > closed=3` → `Metrics.__post_init__` поднимет ValueError и весь session-delta упадёт.

**Фикс**: внутри `_metrics_delta` нормализовать `wins/losses` относительно `closed_trades` либо строить session-metrics без валидации.

### 3.7. `prefer_solo_player_wrappers_enabled` без proven всё ещё подавляет raw безусловно

`flash_allocator.py:723-754`. Если флаг `prefer_solo_player_wrappers_enabled=True`, любой raw агент с существующей Solo-обёрткой исключается. Из аудита V2 (риск 2.4) — флаг по умолчанию false, но если кто-то включит для эксперимента, raw агент гарантированно удаляется. Нет audit-события (V2-2.5 тоже не закрыт).

### 3.8. `Metrics.pnl_net_pct` — property-stub

`domain/types.py:336-338`:

```python
@property
def pnl_net_pct(self) -> float:
    """Net PnL percentage after fees and funding."""
    return self.pnl_pct
```

Docstring обещает «after fees and funding», но возвращается `pnl_pct` без вычитания. Логика правильная (потому что `pnl_pct` уже net в PerformanceMemory), но **если новый код решит сделать `metrics.pnl_gross_pct - metrics.fee_pct - metrics.funding_pct` чтобы получить net — он получит то же самое, что и `pnl_pct`, но через дублирование**. Документация property должна явно сказать: «`pnl_pct` уже net; gross/fee/funding — диагностические поля».

### 3.9. Конфигурационный взрыв продолжается

`FlashAllocatorConfig` — 40+ полей с тонкими взаимозависимостями. Появились новые: `shadow_base_fallback_actor_keys`, `actor_switch_margin`, `degradation_recovery_*`, `overextension_volatility_normalized_*`. По-прежнему нет «именованных пресетов» (`safe`/`default`/`aggressive`) в коде/тестах. Тесты есть на отдельные фичи, но нет invariant-теста типа «при default preset не должна быть включена ни одна experimental опция».

---

## 4. Обновлённая стратегия и шаги дальнейших улучшений

Текущее состояние: вертикаль базовой защиты (net-PnL, switch_margin, recovery, vol-нормированный overextension, regime_confidence, base-fallback allowlist) реально интегрирована. Контракт «Пантеон ≥ best_standalone» теперь стал методически достижим (раньше блокирующих pre-conditions было пять, осталось два-три). Следующий виток должен убрать оставшиеся локальные косяки в гейтах и закрыть три структурных долга (NoTrade as candidate, Bayesian promotion, main_loop decomposition).

### 4.1. P0 на следующий цикл

1. **Закрыть деградацию score-гейта на `regime_confidence=0`** (3.1). Минимально: `gate_score <= min_score_to_trade` или happy-path NoTrade при `regime_confidence < floor`.
2. **`actionable_bonus` тоже умножать на `regime_confidence`** (3.2). Одна строка, но восстанавливает интент.
3. **`previous_actor_by_symbol` запоминать оригинал при cap-обрезке** (3.3). Hysteresis должен переживать cap.
4. **Audit-событие «raw_suppressed_by_solo»** (V2-2.5). Каждое подавление логировать с причиной и lookup на raw метрики, чтобы можно было увидеть, не теряем ли мы edge.
5. **Нормировать `wins+losses` в `_metrics_delta`** (3.6). Сейчас может уронить DegradationGate.
6. **Пресеты конфигурации**. Зафиксировать `FLASH_PRESET_SAFE`, `FLASH_PRESET_DEFAULT`, `FLASH_PRESET_AGGRESSIVE` в `flash_allocator.py` (или composer). Invariant-тест: `safe` имеет все experimental флаги false, `aggressive` — все включены. Это останавливает дрейф конфигурации в продакшен.
7. **Уточнить docstring `Metrics.pnl_net_pct`** (3.8). Указать, что `pnl_pct` уже net.

### 4.2. P1 на следующий цикл

1. **NoTradePlayer как первоклассный кандидат** (1.10). Внести синтетический позитивный score, который зависит от ожидаемых fees при сделке.
2. **Bayesian promotion gate** (1.23). Заменить fixed thresholds на LCB Beta-posterior. Сейчас, после net-PnL, можно строить distribution на `pnl_per_trade`, считать posterior, гейтить по нижней доверительной границе.
3. **`regime_confidence` per-actor calibration** (1.2 + 1.17). Доверие к актору в новом режиме на малой выборке должно быть отдельно от `regime_confidence` рынка.
4. **`EnsemblePlayer.vote()` → возвращать `(signals, errors)`** (1.11). Убрать мутацию `last_vote_errors`.
5. **Tie-breaker pure-deterministic, но без label-bias** (1.7). Использовать хеш `(label, actor_type, bar)` как финальный ключ.
6. **`shadow_full_open_unconfirmed` учитывать `shadow_confirmed_by_base`** (3.5). Либо переименовать reason в `production_full_open_unconfirmed` для base_fallback.

### 4.3. P2 (структурные долги)

1. **Decomposition `main_loop.py`** (1.24). Вынести `_run_flash_decision_path`, `_flash_update_degradation_state_from_events`, `_flash_shadow_confirmation_scores`, `_flash_shadow_player_signals_with_position_replay` в отдельные модули. Сейчас файл по сути hub.
2. **AgentRegistry с lock** (1.21). RWLock либо ImmutableSnapshot-копия для каждого бар.
3. **Concentration penalty (Herfindahl)** на уровне `_apply_actor_signal_cap`. Сейчас cap — hard limit; нужна soft-регулярность.
4. **Soft voting (Agent → AgentVote с confidence)** — глобальная переработка контракта агента.
5. **Invariant-тесты для voting policy** (1.25). `WeightedConsensus == RiskParity(equal vols)` при равных весах; `StrongConsensus(open) ⊂ WeightedConsensus(open)` при тех же thresholds.

### 4.4. Контракт «Пантеон не хуже своих составляющих»

После закрытия net-PnL и switch_margin контракт теперь технически достижим. Следующий шаг — формальный CI с пятью инвариантами:

| Тест | Условие |
| --- | --- |
| panteon_lcb_dominance | `equity_net(Panteon) >= 0.95 * equity_net(best_standalone)` на каждом walk-forward |
| churn_budget | `avg_actor_switches_per_day(Panteon) <= 2 * best_standalone_switches` |
| regime_floor | в каждом из 4 режимов pnl_net >= 0 на out-of-sample |
| notrade_value | сравнить equity с `min_score_to_trade=current` vs `min_score_to_trade=-inf` |
| ensemble_lift | для каждого ensemble player `pnl_net(ensemble) - pnl_net(best_member) >= 0` на out-of-sample |

С учётом текущего состояния — switch_margin, net-PnL, recovery уже введены — эти тесты можно включать в CI. Те, что не проходят, документировать как «known gap» и работать над их закрытием.

### 4.5. Приоритезированная очередь работ

| Очередь | Задача | Категория | Размер |
| --- | --- | --- | --- |
| 1 | Фикс score-гейта при confidence=0 (3.1) | P0 | XS |
| 2 | actionable_bonus × regime_confidence (3.2) | P0 | XS |
| 3 | previous_actor для cap-обрезанных (3.3) | P0 | S |
| 4 | Audit "raw_suppressed_by_solo" (V2-2.5) | P0 | S |
| 5 | Нормировать wins+losses в _metrics_delta (3.6) | P0 | XS |
| 6 | Пресеты FlashAllocatorConfig + invariant-тест (4.1.6) | P0 | M |
| 7 | NoTradePlayer как кандидат (1.10) | P1 | M |
| 8 | Bayesian promotion (1.23) | P1 | M |
| 9 | EnsemblePlayer.vote → return errors (1.11) | P1 | S |
| 10 | Tie-breaker hash (1.7) | P1 | XS |
| 11 | Decomposition main_loop.py | P2 | L |
| 12 | AgentRegistry под lock (1.21) | P2 | S |
| 13 | Concentration penalty (Herfindahl) | P2 | M |
| 14 | Soft voting через AgentVote | P2 | L |
| 15 | Invariant-тесты voting policies (1.25) | P2 | S |

P0-задачи 1-6 — это «гигиена» текущего цикла: они мелкие, не требуют новых концепций, и закрывают остаточные косяки. После них можно с уверенностью включать CI-контракт (4.4).

---

## 5. Краткое резюме третьего цикла

- Закрыто 7 пунктов из тех, что были открыты после V2: 1.3, 1.12, 1.15, 1.19, 1.20, V2-2.1, V2-2.2, V2-2.3, V2-2.6. Это лучший цикл с начала аудита.
- Net-PnL правильно течёт через PerformanceMemory в scoring и audit; switch_margin интегрирован в Flash вместе с recovery; overextension стал per-symbol через z-score; base-fallback теперь под allowlist; regime_confidence масштабирует gate_score; shadow_signal_handoff ограничен known actors; original_selected_actor сохраняется при cap-обрезке.
- Остались 13 пунктов первого отчёта (1.2, 1.7, 1.9, 1.10, 1.11, 1.16, 1.17, 1.18, 1.21, 1.22, 1.23, 1.24, 1.25) и часть V2-рисков (2.4, 2.5, 2.7, 2.10, 2.11).
- Возникли 8 новых мелких проблем (раздел 3): обнуление гейта на confidence=0, актIONABLE без масштаба, потеря previous_actor при cap, slow recovery, base_fallback и full_open_unconfirmed нестыковка, `_metrics_delta` без net-полей, безусловный prefer_solo, путаница в docstring `pnl_net_pct`.
- После закрытия P0-задач (1-6) контракт «Пантеон ≥ best_standalone» становится формально измеримым и можно включать инвариант-тесты в CI.

Следующий разумный шаг — выполнить очередь 1-6 (всё XS/S/M, кроме пресетов M) и зафиксировать `FLASH_PRESET_SAFE` как обязательный default для production. Это переведёт систему из режима «фичи добавляются быстрее, чем тестируются» в режим «фичи только через explicit opt-in».
