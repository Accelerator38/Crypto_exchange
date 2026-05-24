# Panteon Flash — пятый аудит

Дата: 2026-05-22 (re-audit относительно `PANTEON_FLASH_AUDIT_V4.md`).

Этот цикл аудита открывается **блокирующей находкой**: проект на диске сейчас не парсится Python’ом. 19 файлов в `src/panteon_v2/` обрезаны в середине выражения. Это нужно чинить раньше любых других правок.

Документ построен в трёх блоках:

1. блокеры — что должно быть починено до любого другого действия;
2. что закрылось из V4 (где это удалось проверить через Read);
3. оставшиеся задачи + новые наблюдения.

---

## 1. Блокеры: код на диске не импортируется

### 1.1. 19 файлов в `src/panteon_v2/` имеют SyntaxError

`python3 -c "import ast; ast.parse(open(p).read())"` падает на следующих файлах (выгрузка из бакета — реальный размер vs реальная последняя строка):

| Файл | Строка ошибки | Тип ошибки |
| --- | ---: | --- |
| `analysis/retrodate_market_runner.py` | 4222 | unterminated string literal |
| `app/main_loop.py` | 3643 | `(` was never closed |
| `app/shadow_tournament.py` | 806 | `(` was never closed |
| `memory/degradation.py` | 238 | `(` was never closed |
| `memory/performance.py` | 584 | unterminated string literal |
| `selection/__init__.py` | 96 | unterminated string literal |
| `selection/agent.py` | 95 | invalid syntax |
| `selection/flash_allocator.py` | 776 | `(` was never closed |
| `selection/player.py` | 383 | `(` was never closed |
| `selection/promotion_manifest.py` | 95 | invalid syntax |
| `selection/strategist.py` | 2776 | expected an indented block |
| `shadow/runner.py` | 302 | expected an indented block |
| `tests/test_app.py` | 5362 | unterminated string literal |
| `tests/test_degradation_gate.py` | 331 | `(` was never closed |
| `tests/test_flash_allocator.py` | 2085 | `(` was never closed |
| `tests/test_performance.py` | 436 | `(` was never closed |
| `tests/test_player.py` | 203 | `(` was never closed |
| `tests/test_retrodate_market_runner.py` | 2260 | `{` was never closed |
| `tests/test_scoring.py` | 189 | `{` was never closed |

Образцы окончаний (последние 100 байт с диска):

- `memory/degradation.py` → `pending_signals=max(` + пробелы (вызов `max()` не закрыт);
- `selection/strategist.py` → `def _safe_float(value: object) -> float:` + пробелы (функция без тела);
- `app/main_loop.py` → `n_blocked: ` (параметр функции обрывается).

### 1.2. Что это означает

Эти файлы не импортируются. Любой запуск, любой тест, любой backtest упадёт на `SyntaxError` до того, как дойдёт до бизнес-логики. Это **не** локальный кеш-артефакт — `wc -l`, `tail -c`, `ast.parse()` и `cat` независимо подтверждают, что файлы физически обрезаны.

Косвенный признак того, что это «активное состояние»: в `git status` стоит файл `src/panteon_runtime/agent_meta.py.truncated_backup` — то есть кто-то уже сталкивался с обрезкой и оставил резервную копию вручную.

### 1.3. Что нужно сделать прямо сейчас (P0-блокер)

1. **Запустить `python -c "import panteon_v2"`** и зафиксировать список тех 19 файлов как broken-list.
2. **Сравнить с предыдущим коммитом** (`git show HEAD:<path>`) и решить, какие изменения нужно сохранить — каждый из 19 файлов был отредактирован, и где-то посередине edit оборвался.
3. **Восстановить целостность построчно**:
   - для файлов, где «незакрытая скобка» — пересобрать выражение из git-истории + diff;
   - для файлов с `unterminated string literal` — найти открывающую кавычку и дописать закрывающую;
   - для `strategist.py:2776` и `shadow/runner.py:302` — дописать тело функции (`pass` как минимум).
4. **Проверка**: `python3 -m py_compile <file>` для каждого. Только после того, как все 19 проходят compile, переходить к шагу 5.
5. **Запустить полный тест-сьют**: `pytest src/panteon_v2/tests`. Прежде чем чинить логику, надо убедиться, что текущее состояние правок не сломало готовые сценарии.

Только когда эти пять шагов завершены, продолжать содержательный аудит. До этого «менять Пантеон» — это менять то, что не запускается.

---

## 2. Что закрылось из V4 (видимое через Read-snapshot)

В пределах того, что доступно для чтения (Read tool снимок), у меня есть подтверждение, что часть V4-задач была начата. Внимание: эти изменения видны в Read-snapshot, но соответствующие файлы на диске сейчас обрезаны (см. §1). После восстановления нужно пере-проверить, что эти правки действительно дошли до диска целиком.

| V4-пункт | Что видно в snapshot | Где |
| --- | --- | --- |
| V4-2.7 (Bayesian LCB в promotion manifest) | Добавлены `min_win_rate_lcb_pct`, `win_rate_lcb_z`, helper `_beta_win_rate_lcb_pct` | `promotion_manifest.py:18-19,85-93,100-115` |
| V4-2.6 (AgentRegistry с lock) | `threading.RLock` в `__init__`, `with self._lock:` во всех методах | `selection/agent.py:14,64-128` |

Эти патчи в snapshot выглядят корректно, но в `agent.py` Read показал ещё и **дублированную dead-code-копию** body метода `register()` после `return` внутри `with`-блока (строки 87-100 в snapshot). На диске эта же область обрывается с `if agent.` (строка 95), то есть дубликат на диск так и не дописался. Это и есть симптом «обрыв во время edit». Чинить надо так: удалить дубликат целиком, оставить только версию внутри `with self._lock:`.

Дополнительно, в snapshot `agent.py:74` есть mojibake-комментарий: `# runtime_checkable Protocol вЂ” РїСЂРѕРІРµСЂСЏРµС‚ РЅР°Р»РёС‡РёРµ label Рё act`. Это «`— проверяет наличие label и act`» в CP1251, прочитанное как UTF-8. Источник, скорее всего, копипаста из другого файла, сохранённого в неверной кодировке. На диске такого комментария нет (обрезан). При восстановлении надо использовать корректную UTF-8 версию: `# runtime_checkable Protocol — проверяет наличие label и act`.

---

## 3. Что нужно менять (после починки §1)

Поскольку диск сейчас в неконсистентном состоянии, итоговый список задач V5 строится так: P0 — починка §1; затем — остатки V4, повторно подтверждённые после восстановления; затем — новые наблюдения, которые видно по Read-snapshot уже в этом цикле.

### 3.1. P0 — обязательная починка перед чем-либо ещё

| # | Действие | Где | Размер |
| --- | --- | --- | --- |
| 1 | Восстановить 19 broken-файлов по списку из §1.1 до состояния, проходящего `py_compile` | весь `src/panteon_v2/` | M |
| 2 | Удалить dead-code дубликат в `register()` после `return` | `selection/agent.py:87-100` | XS |
| 3 | Заменить mojibake-комментарий на корректный UTF-8 | `selection/agent.py:74` (snapshot)/`:88` (disk) | XS |
| 4 | Прогнать полный pytest-сьют, починить упавшие тесты | `tests/` | S |
| 5 | Зафиксировать состояние в одном коммите «Recover panteon_v2 from truncated edits» | git | XS |

После пяти шагов и зелёного теста — переходить к P1.

### 3.2. P1 — задачи V4, которые остались

После починки эти задачи переподтвердить и закрыть. Они не изменились по сути с V4 §2.

| # | Что | Где | Размер | Статус по V4 |
| --- | --- | --- | --- | --- |
| 6 | `previous_actor_by_symbol` хранит фактически торгующего | `main_loop.py:940-953,1645-1660`, `flash_allocator.py:759-761` | XS | V4-2.1 — open |
| 7 | `selected_reasons` цепочкой в FlashDecision | `flash_allocator.py:317-341,743-772` | S | V4-2.2 — open |
| 8 | `FLASH_PRESET_SAFE` явно через kwargs + invariant `SAFE != DEFAULT` | `flash_allocator.py:221-222`, `tests/test_flash_allocator.py:512-522` | XS | V4-2.3 — open |
| 9 | `_metrics_delta` пробрасывает `pnl_gross_pct/fee_pct/funding_pct` | `memory/degradation.py:217-238` | XS | V4-2.4 — open |
| 10 | `EnsemblePlayer.vote()` → `(signals, errors)` | `selection/player.py:91-220,247-380` | M | V4-2.5 — open |
| 11 | `regime_score(regime, ...)` — реально использовать параметр | `scoring/scoring.py:99-161` | M | V4-2.8 — open |
| 12 | `regime_score` без short-circuit при `closed<2` | `scoring/scoring.py:131-133` | XS | V4-2.9 — open |
| 13 | Safety-check на уровне агента, не Solo-обёртки | `main_loop.py:1083-1091` | S | V4-2.10 — open |
| 14 | `actionable_bonus` мультипликативный | `flash_allocator.py:572-573` | XS | V4-2.11 — open |

V4-2.7 (Bayesian LCB) — частично выполнен (в snapshot есть код, на диске обрезан): после §3.1 проверить, что код реально на месте, и **прикрутить тест**: при 50 сделок и эмпирическом win_rate=60% сигнал должен **не** проходить (LCB ниже 50%), при 500 и win_rate=55% — проходить.

### 3.3. P2 — крупные долги (без изменений с V4)

| # | Что | Где | Размер |
| --- | --- | --- | --- |
| 15 | Декомпозиция `main_loop.py` | весь файл | L |
| 16 | Invariant-тесты voting policies | `tests/test_voting.py` | S |
| 17 | Anchor + switch_margin формальные тесты | `tests/test_flash_allocator.py` | S |
| 18 | Унификация формата shadow-key (один формат вместо шести) | `flash_allocator.py:1158-1193,1495+` | M |

---

## 4. Новые наблюдения, заметные в snapshot

Из того, что я успел увидеть в этом цикле через Read до того, как обнаружил §1 (то есть в свежем коде, который на диске уже частично обрезан):

### 4.1. `register()` в `AgentRegistry` имеет дублированную dead-code-копию (snapshot)

`selection/agent.py:68-100` показывает:

```python
def register(self, agent, *, replace=False) -> None:
    with self._lock:
        if not isinstance(agent, Agent):
            # mojibake comment
            raise TypeError(...)
        ...
        self._agents[agent.label] = agent
        return                              # ← exit
    if not isinstance(agent, Agent):        # ← DEAD: после return
        raise TypeError(...)
    ...
    self._agents[agent.label] = agent       # ← DEAD
```

Это побочный эффект незавершённого edit-а: старая версия body осталась под `with`-блоком после того, как новая была вложена внутрь. Удалить строки 87-100. Запись «inside lock» — единственная корректная.

### 4.2. mojibake в комментарии `agent.py:74`

Источник: CP1251 → UTF-8 mangling. Заменить на корректный UTF-8 текст «`# runtime_checkable Protocol — проверяет наличие label и act`».

### 4.3. `promotion_manifest.py` Bayesian-патч сделан только наполовину

В snapshot:

- добавлены поля `min_win_rate_lcb_pct: float = 0.0`, `win_rate_lcb_z: float = 1.64` (`:18-19`);
- добавлен helper `_beta_win_rate_lcb_pct(...)` через mean-минус-z-sigma на Beta-апостериоре (`:100-115`);
- в `_promotion_rejection_reason` добавлен опциональный гейт (`:85-93`).

Что нужно ещё:

- `min_win_rate_lcb_pct=0.0` означает «гейт выключен по умолчанию». Поставить дефолт `50.0` (то есть «нижняя граница 95% доверительного интервала на win_rate ≥ 50%»);
- `win_rate_lcb_z` обычно записывают как `1.645` (95% one-sided) или `1.96` (95% two-sided). Сейчас `1.64` — близко к 1.645, но лучше явно: либо `1.645`, либо именованная константа `Z_95 = 1.6448536`;
- Beta-апостериор обычно использует **квантиль** Beta-распределения, а не «среднее минус z·std». Текущая формула — приближение нормальной аппроксимацией, корректно при больших n, но смещено при малых. Для small-n лучше использовать Wilson lower bound:
  ```python
  def wilson_lcb(wins: int, n: int, z: float = 1.6448536) -> float:
      if n == 0:
          return 0.0
      p = wins / n
      z2 = z * z
      denom = 1.0 + z2 / n
      centre = p + z2 / (2 * n)
      margin = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
      return max(0.0, (centre - margin) / denom)
  ```
- Добавить тест:
  ```
  20 сделок, 12 wins (win_rate=60%) → wilson_lcb ≈ 39% → отклоняется
  500 сделок, 275 wins (win_rate=55%) → wilson_lcb ≈ 51% → проходит
  ```

### 4.4. flash_allocator снова обрезан до 776 строк, потом до 776 на диске

Размер flash_allocator колеблется: V2 = 789, V3 = 769, V4 = 755, V5 snapshot = 776, на диске = 776 строк (последняя — `if (\n        self._confi\ng.sh`). На уровне snapshot файл выглядит логически закончен, но **на диске последняя строка обрезана** — то же явление, что и в §1. Это блокер для запуска.

### 4.5. `selection/__init__.py:96` тоже сломан

Не критично для собственно Flash-логики, но при `import panteon_v2.selection` упадёт. Это означает, что **никакой потребитель** `selection.FlashAllocator`, `EnsemblePlayer`, `WeightedConsensus` не сможет загрузиться. Любой downstream-модуль, импортирующий `panteon_v2.selection`, упадёт первым. Чинить в первую очередь.

---

## 5. Сводка V5

- **Главная находка**: 19 файлов в `src/panteon_v2/` (включая `flash_allocator`, `main_loop`, `promotion_manifest`, `agent`, `__init__`, тесты) физически обрезаны в середине выражения. Проект сейчас не импортируется.
- **Содержательные правки этого цикла** (видимые в snapshot, но не докатанные до диска): Bayesian LCB в promotion manifest, `threading.RLock` в AgentRegistry. Обе правки нужно довосстановить и протестировать.
- **Из V4 закрыто** только то, что вошло в snapshot выше; остальные 9 задач (V4-2.1 — V4-2.11, кроме 2.6 и 2.7) — open.
- **Список «что менять прямо сейчас»** — раздел §3.1 (5 шагов восстановления). До его выполнения остальные предложения смысла не имеют.

После восстановления — пройтись по списку §3.2 и §3.3.
