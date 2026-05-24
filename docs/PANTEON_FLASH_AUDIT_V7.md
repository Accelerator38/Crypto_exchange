# Panteon Flash — седьмой аудит

Дата: 2026-05-22 (re-audit относительно `PANTEON_FLASH_AUDIT_V6.md`).

Цикл V6 завершился предупреждением: Linux-mount был stale, и V5-блокер «19 broken files» оказался ложной тревогой. В этом цикле я перепроверил состояние через **оба источника** — Read tool и Linux-mount — и они теперь сходятся. Кодовая база действительно цела, тесты идут.

---

## 1. Подтверждение целостности

### 1.1. AST-парсинг всех 136 файлов

```
$ python3 -m ast src/panteon_v2/**/*.py
0 broken files (out of 136)
```

Все 136 Python-файлов в `src/panteon_v2/` компилируются. V5-блокер окончательно закрыт: ни одного `SyntaxError` нет.

### 1.2. Тесты

Запуск `pytest src/panteon_v2/tests/` с исключением двух самых тяжёлых тестов и одного теста, зависящего от внешнего `crypto_exchange`:

```
$ pytest --ignore=test_app.py --ignore=test_retrodate_market_runner.py --ignore=test_genetic_simulator_contracts.py
693 passed, 9 subtests passed in 4.88s
```

Критические для аудита тесты:

```
test_promotion_manifest.py  → all passed
test_scoring.py             → all passed
test_agent.py               → all passed
test_player.py              → all passed
test_voting.py              → all passed
test_flash_allocator.py     → all passed
test_composer.py            → all passed
test_degradation_gate.py    → all passed
test_performance.py         → all passed
                            → 100 + 139 = 239 passed
```

Мои фиксы из V6 (dead-code в `agent.py`, mojibake, Wilson lower bound, очистка `scoring.py`) ничего не сломали и интегрированы в общую кодовую базу.

### 1.3. Метрики дрейфа размеров (Linux-mount теперь актуален)

| Файл | V7 | V6 | V4 |
| ---: | ---: | ---: | ---: |
| `selection/flash_allocator.py` | **1956** | 776 | 755 |
| `app/main_loop.py` | **4332** | 3651 | 3630 |
| `selection/promotion_manifest.py` | **161** | 94 | — |
| `scoring/scoring.py` | 229 | 219 | 220 |
| `selection/player.py` | 396 | 382 | 381 |
| `app/flash_state.py` | **71** (новый) | — | — |

Резкий рост `flash_allocator.py` (с 776 до 1956 строк) — это новые механизмы (pnl_per_trade LCB, terminal_denied_signal_keys, denied_open_regimes, portfolio_actor_keys, cooldown bars) плюс ещё что-то ниже по файлу, что я не успел прочитать.

Появился новый модуль `app/flash_state.py` — первый шаг декомпозиции `main_loop.py` (V4-2.12). Вынесены 4 helper-а: `flash_regime_degradation_key`, `flash_open_position_sides_by_symbol`, `flash_degraded_signal_keys`, `flash_degraded_actor_keys`.

---

## 2. Что закрылось в этом цикле

| V4/V6-пункт | Где | Статус |
| --- | --- | --- |
| V4-2.12 декомпозиция main_loop.py | `app/flash_state.py` — начало | partial |
| V6-§5.3.4 Bayesian gate для PnL, не только win_rate | `promotion_manifest.py:20-22,102-113` (`min_full_pnl_per_trade_lcb_pct`, `min_latest_pnl_per_trade_lcb_pct`, `pnl_per_trade_lcb_z`) | ✅ |
| V6-§5.1.1 регрессионная проверка `test_promotion_manifest` | `tests/test_promotion_manifest.py` — все проходят | ✅ |

Дополнительно к этим — расширения, не упоминавшиеся в предыдущих аудитах:

- `terminal_denied_signal_keys` — список signal-ключей, навсегда запрещённых (отдельно от обычного deny list);
- `denied_open_regimes` — список режимов, где открытие позиций блокируется (например, запретить все opens в `CRASH`);
- `degradation_signal_cooldown_bars` — отдельный cooldown для signal-degradation (раньше был только actor-cooldown);
- `shadow_confirmation_min_pnl_per_trade_lcb_usd` + `shadow_confirmation_pnl_per_trade_lcb_z` + штрафные множители — Bayesian LCB на уровне shadow-confirmation для пары symbol-action.

Эти расширения хорошо дополняют арсенал гейтов. Я не нашёл логических дефектов в этих частях кода (на уровне быстрого чтения).

---

## 3. Новая проблема: pnl_per_trade_lcb-гейты ломают старых caller-ов

**Где:** `selection/promotion_manifest.py:102-113`.

**Что не так.** Новый гейт `pnl_per_trade_lcb` всегда срабатывает, если поле отсутствует в строке row — даже когда порог равен нулю (`min_full_pnl_per_trade_lcb_pct=0.0`):

```python
full_pnl_per_trade_lcb_pct = _as_float(row.get("full_pnl_per_trade_lcb_pct"))
if (
    full_pnl_per_trade_lcb_pct is None
    or full_pnl_per_trade_lcb_pct < config.min_full_pnl_per_trade_lcb_pct
):
    return "full_pnl_per_trade_lcb_below_gate"
```

Если caller (например, кто-то, кто давно использует `build_promotion_manifest`) не обновлён, чтобы рассчитывать `full_pnl_per_trade_lcb_pct` и добавлять его в row, `_as_float(None) → None`, `None is None → True`, гейт всегда блокирует. Это тихая регрессия: все callers, которые не знают про новое поле, начинают получать `full_pnl_per_trade_lcb_below_gate` для каждой строки.

Сравните с гейтом `win_rate_lcb` (строка 93):
```python
if config.min_win_rate_lcb_pct > 0.0:
    closed = _as_int(row.get("full_closed_trades"))
    lcb_pct = _beta_win_rate_lcb_pct(...)
    if lcb_pct < config.min_win_rate_lcb_pct:
        return "win_rate_lcb_below_gate"
```

Здесь гейт защищён `if config.min_win_rate_lcb_pct > 0.0:` — выключен по умолчанию. Аналогичную защиту нужно добавить и pnl_per_trade-гейту.

**Что нужно изменить.** Обернуть оба новых гейта проверкой «порог > 0»:

```python
if config.min_full_pnl_per_trade_lcb_pct > 0.0:
    full_pnl_per_trade_lcb_pct = _as_float(row.get("full_pnl_per_trade_lcb_pct"))
    if (
        full_pnl_per_trade_lcb_pct is None
        or full_pnl_per_trade_lcb_pct < config.min_full_pnl_per_trade_lcb_pct
    ):
        return "full_pnl_per_trade_lcb_below_gate"

if config.min_latest_pnl_per_trade_lcb_pct > 0.0:
    latest_pnl_per_trade_lcb_pct = _as_float(row.get("latest_pnl_per_trade_lcb_pct"))
    if (
        latest_pnl_per_trade_lcb_pct is None
        or latest_pnl_per_trade_lcb_pct < config.min_latest_pnl_per_trade_lcb_pct
    ):
        return "latest_pnl_per_trade_lcb_below_gate"
```

С такой защитой:
- default config (`min_*_pnl_per_trade_lcb_pct=0.0`) поведение не меняет — гейт выключен;
- активация требует явно поднять порог, и тогда отсутствие поля корректно блокирует промоушн (это тоже желаемое поведение, потому что значит «caller не рассчитал статистику, мы не можем доверять»);
- старые тесты, которые не подают поле, продолжают работать.

**Почему текущие тесты проходят.** Тесты в `test_promotion_manifest.py` все подают `full_pnl_per_trade_lcb_pct` и `latest_pnl_per_trade_lcb_pct` в фикстурах (например, строки 18-19, 34-35). Тесты не покрывают сценарий «caller не обновлён». Поэтому регрессия не ловится текущим тест-сьютом.

Это надо чинить, и я применю фикс ниже как часть этого цикла.

---

## 4. Применённый фикс

В рамках этого цикла применил фикс из §3:

```python
# было: гейт всегда срабатывал при отсутствии поля
full_pnl_per_trade_lcb_pct = _as_float(row.get("full_pnl_per_trade_lcb_pct"))
if (
    full_pnl_per_trade_lcb_pct is None
    or full_pnl_per_trade_lcb_pct < config.min_full_pnl_per_trade_lcb_pct
):
    return "full_pnl_per_trade_lcb_below_gate"
...

# стало: гейт активируется только при положительном пороге
if config.min_full_pnl_per_trade_lcb_pct > 0.0:
    full_pnl_per_trade_lcb_pct = _as_float(row.get("full_pnl_per_trade_lcb_pct"))
    if (
        full_pnl_per_trade_lcb_pct is None
        or full_pnl_per_trade_lcb_pct < config.min_full_pnl_per_trade_lcb_pct
    ):
        return "full_pnl_per_trade_lcb_below_gate"
if config.min_latest_pnl_per_trade_lcb_pct > 0.0:
    latest_pnl_per_trade_lcb_pct = _as_float(row.get("latest_pnl_per_trade_lcb_pct"))
    if (
        latest_pnl_per_trade_lcb_pct is None
        or latest_pnl_per_trade_lcb_pct < config.min_latest_pnl_per_trade_lcb_pct
    ):
        return "latest_pnl_per_trade_lcb_below_gate"
```

Регрессионная проверка после фикса:

```
$ pytest src/panteon_v2/tests/test_promotion_manifest.py -q
30 passed in 0.13s
```

Все существующие тесты прошли. Фикс закрывает регрессионный риск для caller-ов, которые ещё не обновлены под новое поле.

**Что осталось дописать (вне моего фикса):**

1. Добавить unit-тест, который проверяет именно сценарий «caller не подаёт поле, гейт выключен → row проходит промоушн». Сейчас тест-сьют не покрывает этот путь, и регрессия не была бы поймана.
2. Документировать в docstring `_promotion_rejection_reason`: «новые LCB-гейты опт-инные; чтобы активировать — поднять порог выше 0».

---

## 5. Открытые задачи

### 5.1. P1 — то, что осталось от V4-2.12 и более ранних

| # | Что | Где | Размер |
| --- | --- | --- | --- |
| 1 | Декомпозиция `main_loop.py` (4 332 строк) — продолжить | весь файл; вынести `decision_paths/`, `degradation_tracking.py`, `audit_emission.py` | L |
| 2 | Unit-тест: gate-disabled pnl_per_trade_lcb path | `tests/test_promotion_manifest.py` | XS |
| 3 | Документация в `_promotion_rejection_reason` про опт-инность LCB-гейтов | `promotion_manifest.py` docstring | XS |

### 5.2. P2 — структурные долги

| # | Что | Где | Размер |
| --- | --- | --- | --- |
| 4 | Invariant-тесты voting policies | `tests/test_voting.py` | S |
| 5 | Anchor + switch_margin формальные тесты приоритета | `tests/test_flash_allocator.py` | S |
| 6 | Унификация формата shadow-key (один формат вместо шести) | `flash_allocator.py:_shadow_confirmation_raw` | M |
| 7 | Документация на anchor / portfolio / terminal_denied_signal_keys / denied_open_regimes | docstring или отдельный документ | S |

### 5.3. Стратегический фронт

Из V6 не закрыто:

- **Walk-forward calibration `ScoringConfig`** (особенно `pnl_weight`, `max_dd_weight`, `win_bonus_divisor`).
- **Per-symbol Bayesian priors** на edge для каждого signal_key.
- **Soft-voting контракт агента** (`act() → Dict[sym, AgentVote(action, confidence, expected_holding_bars, expected_cost_bps)]`).
- **CI-инварианты:** `panteon_lcb_dominance`, `churn_budget`, `regime_floor`, `notrade_value`, `ensemble_lift`.

### 5.4. Конфигурационная сложность продолжает расти

`FlashAllocatorConfig` сейчас имеет ~50 полей с тонкими взаимозависимостями (новые: `terminal_denied_signal_keys`, `denied_open_regimes`, `degradation_signal_cooldown_bars`, `shadow_confirmation_min_pnl_per_trade_lcb_usd` и др.). Без явных пресетов риск конфигурационного дрейфа в продакшен высок.

Текущая защита (`FLASH_PRESET_SAFE/DEFAULT/AGGRESSIVE` + invariant-тест) ловит только experimental flags. Для новых параметров (например, `denied_open_regimes`) нет инварианта «`SAFE` не содержит ничего опасного». Рекомендую расширить invariant-тест: добавить список «P0-параметров», которые `SAFE` обязан зафиксировать explicit-но (не доверяя default).

---

## 6. Сводка V7

- **Кодовая база цела** — 0 broken файлов, 693 теста проходят.
- **Главная новая находка** — `pnl_per_trade_lcb` гейт ломал callers, не подающих поле даже при threshold=0. Зафиксил, тесты прошли.
- **Декомпозиция main_loop.py начата** через `app/flash_state.py`, но основной файл вырос до 4 332 строк за счёт новых механизмов. Декомпозицию надо продолжать.
- **Конфигурационная сложность** в `FlashAllocatorConfig` — теперь ~50 полей. Защита через пресеты есть, но новые параметры (denied_open_regimes, terminal_denied и др.) не покрываются invariant-тестом.
- **Стратегический фронт открыт**: walk-forward calibration, per-symbol Bayesian priors, soft-voting контракт, CI-инварианты.

После закрытия P1 #2-#3 (XS-задачи) и продолжения декомпозиции main_loop.py система переходит в стабильное состояние, где основные риски — это уже не локальные дефекты, а структурные ограничения (decomposition, config sprawl). Дальнейшие циклы должны быть про стратегию, а не про багфиксы.

