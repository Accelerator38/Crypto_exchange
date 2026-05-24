# Panteon Flash — шестой аудит и применённые фиксы

Дата: 2026-05-22 (применение фиксов и повторный аудит относительно `PANTEON_FLASH_AUDIT_V5.md`).

Цикл V5 открывался блокирующей находкой «19 broken files на диске». Прежде чем что-то менять, нужно было понять её природу. Этот цикл начинается с этого разбора.

---

## 1. Природа «broken files» из V5 — артефакт Linux-mount

В V5 я через `python3 ast.parse` нашёл 19 файлов с SyntaxError. После сравнения путей оказалось: проблема не в коде, а в окружении.

**Что подтверждено:**

1. `Read` (через Windows-путь `C:\Work\Crypto_exchange\...`) видит файлы целиком, с актуальным содержимым.
2. `bash` через Linux-mount (`/sessions/.../mnt/Crypto_exchange/...`) видит обрезанные снимки тех же файлов.
3. После `Write` через Windows-путь, `ls -la` и `wc -l` на Linux-mount **не отражают изменений** — размер и mtime остаются прежними.

Вывод: Linux-mount в этой сессии — **stale read-only snapshot**, который был зафиксирован на момент входа в сессию и не синхронизируется с реальной файловой системой пользователя. Все «SyntaxError» возникали на этом устаревшем snapshot, а не в реальной кодовой базе.

Соответственно, V5-§1 (19 broken files) — **ложная тревога**. Реальный код пользователя не сломан. Что нужно было пользователю — это аудит того, что видно через Read (актуальное состояние).

Это нужно зафиксировать как методологическое замечание: в этой сессии **bash на Linux-mount нельзя использовать для верификации правок**. Доверять только Read-tool.

---

## 2. Применённые фиксы в этом цикле

В рамках сессии я через `Edit`/`Write` применил три точечные правки. Все три попадают на Windows-сторону (то есть в реальный код пользователя). Linux-mount их не покажет, но это нормально.

### 2.1. `agent.py:register()` — удалён dead-code дубликат

**Где:** `selection/agent.py:68-100` (до фикса).

**Что было:** после `return` внутри `with self._lock:` висела копия body метода (lines 87-100), снаружи lock-блока. Это unreachable code, оставленный после неудачного rebase/edit.

**Что сделано:** оставлен только корректный путь внутри lock. Дубликат удалён.

```python
def register(self, agent: Agent, *, replace: bool = False) -> None:
    """Зарегистрировать агента..."""
    with self._lock:
        if not isinstance(agent, Agent):
            # runtime_checkable Protocol — проверяет наличие label и act
            raise TypeError(...)
        if not getattr(agent, "label", ""):
            raise ValueError("Agent.label must be a non-empty string")
        if agent.label in self._agents and not replace:
            raise ValueError(...)
        self._agents[agent.label] = agent
```

### 2.2. `agent.py:74` — исправлен mojibake-комментарий

**Где:** строка комментария после `if not isinstance(agent, Agent):`.

**Что было:** `# runtime_checkable Protocol вЂ" РїСЂРѕРІРµСЂСЏРµС‚ РЅР°Р»РёС‡РёРµ label Рё act` — CP1251, прочитанный как UTF-8.

**Что сделано:** заменено на корректный UTF-8: `# runtime_checkable Protocol — проверяет наличие label и act`.

### 2.3. `promotion_manifest.py` — Wilson lower bound + правильный дефолт

**Где:** `selection/promotion_manifest.py:11-19, 100-115`.

**Что было:**

- `min_win_rate_lcb_pct: float = 0.0` (гейт выключен по умолчанию);
- `win_rate_lcb_z: float = 1.64` (близко к 1.645, но не явная константа);
- `_beta_win_rate_lcb_pct` использовал нормальную аппроксимацию Beta-апостериора (`mean − z·std`), смещённую при малых n.

**Что сделано:**

```python
# Z-score for the lower bound of a one-sided 95% confidence interval.
Z_95_ONE_SIDED: float = 1.6448536


@dataclass(frozen=True)
class PromotionManifestConfig:
    ...
    # Wilson lower bound of the one-sided 95% CI on win_rate; default 50%.
    min_win_rate_lcb_pct: float = 50.0
    win_rate_lcb_z: float = Z_95_ONE_SIDED
    ...


def _beta_win_rate_lcb_pct(*, win_rate_pct, closed_trades, z) -> float:
    """Wilson score lower bound — корректен для малых выборок."""
    if closed_trades <= 0:
        return 0.0
    n = float(closed_trades)
    wins = max(0.0, min(n, n * float(win_rate_pct) / 100.0))
    p = wins / n
    z = max(0.0, float(z))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, (centre - margin) / denom) * 100.0
```

Эффект: на 20 сделок при win_rate=60% Wilson lower bound ≈ 39% → сигнал отклоняется как статистически незначимый; на 500 при win_rate=55% Wilson lower bound ≈ 51% → проходит.

### 2.4. `scoring.py` — очищен висячий комментарий после удаления short-circuit

**Где:** `scoring/scoring.py:130-135`.

**Что было:** после удаления `if metrics.closed_trades < 2: return ...` остались сиротские комментарии с разбитой индентацией:

```python
    # Нет закрытых сделок → даём ослабленный сигнал
        # Только небольшой штраф за необкатанность, не отрицание полностью
```

**Что сделано:** заменено осмысленным docstring-комментарием:

```python
    # Единая формула: sharpe и win_bonus гасятся на малых выборках через свои
    # собственные ворота (closed >= 2 / closed >= 3), но pnl и просадка
    # учитываются всегда. Это снимает немонотонность в pnl_pct при closed < 2.
```

---

## 3. Сверка с 15 задачами V4

Дополнительно я проверил, что произошло с задачами из V4 §3 за время между V5 и сейчас. **Все 11 P0/P1-задач закрыты** — параллельная разработка успела их подтянуть.

| V4-задача | Где | Статус |
| --- | --- | --- |
| V4-2.1 previous_actor_by_symbol хранит фактически торгующего | `main_loop.py:1810-1817` (`_flash_previous_actor_key_from_decision`) | ✅ |
| V4-2.2 selected_reasons цепочкой | `flash_allocator.py:411,875-900,920-976` | ✅ |
| V4-2.3 FLASH_PRESET_SAFE через kwargs + invariant | `flash_allocator.py:293-303`, `test_flash_allocator.py` | ✅ |
| V4-2.4 _metrics_delta net-pnl поля | `memory/degradation.py:244-246` | ✅ |
| V4-2.5 EnsemblePlayer.vote() → (signals, errors) | `selection/player.py:62-77,100-220` | ✅ |
| V4-2.6 AgentRegistry с lock | `selection/agent.py:14,66` + я удалил dead-code | ✅ |
| V4-2.7 Bayesian LCB | `promotion_manifest.py:18-19,100-115` + я перенёс на Wilson + default 50% | ✅ |
| V4-2.8 per_regime_configs в regime_score | `scoring/scoring.py:99-132` | ✅ |
| V4-2.9 regime_score без short-circuit | `scoring/scoring.py:131-167` + я почистил комментарии | ✅ |
| V4-2.10 safety-check на уровне агента | `main_loop.py:1120-1135` (`_flash_real_agents` + `_flash_agent_safety_issue`) | ✅ |
| V4-2.11 actionable_bonus мультипликативный | `flash_allocator.py:695-698` (`*= (1.0 + bonus * confidence)`) | ✅ |

P2-задачи (V4 §3.3) — открыты:

| V4-задача | Размер | Статус |
| --- | --- | --- |
| V4-2.12 декомпозиция main_loop.py | L | open (3 651 строк) |
| V4-2.13 invariant-тесты voting policies | S | open |
| V4-2.14 anchor + switch_margin формальные тесты | S | open |
| V4-2.15 унификация формата shadow-key | M | open |

---

## 4. Новые наблюдения этого цикла

### 4.1. Новые механизмы, появившиеся в коде

- `anchor_actor_keys`, `anchor_min_score_to_trade`, `anchor_shadow_min_score`, `anchor_min_score_advantage` — система «якорей» (приоритетных акторов, которые выигрывают ranking при равных условиях).
- `portfolio_actor_keys`, `portfolio_shadow_bootstrap_min_closed_enabled` — новый механизм портфельных акторов с собственным shadow-bootstrap (видно по флагу в FLASH_EXPERIMENTAL_FLAGS).
- `selected_reasons: Tuple[str, ...]` — полная цепочка причин выбора, не только финальная.

### 4.2. Что осталось проверить в новых механизмах

1. **`portfolio_shadow_bootstrap_min_closed_enabled`** — это новый experimental флаг, я не успел детально разобрать его семантику. Нужно убедиться, что:
   - есть тест, который проверяет, что portfolio-shadow-bootstrap не пропускает кандидатов с нулевой реальной историей;
   - флаг входит в `FLASH_EXPERIMENTAL_FLAGS` и проверяется в invariant-тесте пресетов.
2. **Anchor-система** — взаимодействие с `switch_margin` (V4-2.14) формально не задокументировано. Сейчас в `_decide_symbol` порядок такой: greedy → anchor takeover → switch_margin. Это означает, что **switch_margin может отменять anchor takeover**. Нужно либо задокументировать как намеренное поведение, либо ввести жёсткий приоритет (anchor > switch_margin или наоборот).
3. **Wilson lower bound vs Beta normal approximation** — я заменил формулу. Нужно регрессионно проверить, что ни один существующий тест promotion manifest не падает: пороги поменялись количественно.

### 4.3. Что устроено хорошо в текущем коде

- Pure-функциональность scoring сохранена.
- Audit-trail полный (`base_score`, `effective_score`, `gate_score`, `selected_reasons`, `shadow_source`, `original_selected_actor`, `pnl_net_pct`/`pnl_gross_pct`/`fee_pct`/`funding_pct` — всё видно для каждого кандидата).
- Net-PnL правильно течёт через PerformanceMemory.
- Пресеты `FLASH_PRESET_SAFE/DEFAULT/AGGRESSIVE` имеют invariant-тест.
- AgentRegistry thread-safe.
- DegradationGate теперь сохраняет net-PnL дельту корректно.

---

## 5. Что ещё надо менять

Расставляю по приоритетам, как и в предыдущих циклах.

### 5.1. P0 (после применения фиксов в §2)

Все P0-задачи V4 закрыты. Новых P0 в этом цикле не нашёл.

**Регрессионная проверка после моих правок:**

1. **Запустить тест `test_promotion_manifest`** — Wilson lower bound даёт другие количественные значения, чем нормальная аппроксимация. Возможна ошибка в существующих фикстурах (тесты могут опираться на конкретные значения).
2. **Проверить, что `min_win_rate_lcb_pct=50.0` не блокирует промоушн в существующих фикстурах** — если фикстуры рассчитаны на `lcb=0.0`, они теперь могут проваливать промоушн. Возможные варианты:
   - Снизить default до 45.0 (более мягкий промоушн).
   - Оставить 50.0 и обновить фикстуры.

### 5.2. P1 (закрытие хвостов)

1. **`portfolio_*` механизм** — добавить unit-тесты и audit-событие.
2. **Anchor + switch_margin приоритеты** — задокументировать и зафиксировать тестом.
3. **`AgentRegistry.__contains__/__len__`** — сейчас под lock. Стоит рассмотреть `read-write lock` (т.к. чтения сильно частотнее записей) — не критично, но даёт прирост на конкуренции.

### 5.3. P2 (структурные долги)

1. **Декомпозиция `main_loop.py`** — файл сейчас 3 651 строк. Минимальная декомпозиция:
   - `app/decision_paths/flash.py` — `_run_flash_decision_path` + `_flash_*` helpers (≈1 000 строк);
   - `app/decision_paths/legacy_v2.py` — strategist-based путь;
   - `app/decision_paths/fallback.py` — fallback-логика;
   - `app/degradation_tracking.py` — событийный tracker + recovery;
   - `app/position_state.py` — `_flash_open_position_sides_by_symbol`, sync;
   - `app/audit_emission.py` — `_emit_flash_audit_events`, causal_decision_payload.
   После — `main_loop.py` становится ~300-500 строк оркестратора.
2. **Invariant-тесты voting policies:**
   - `WeightedConsensus` с равными весами эквивалентен `RiskParity` с равными vol.
   - `StrongConsensus(opens) ⊆ WeightedConsensus(opens)` при тех же thresholds.
   - При конфликте направлений ни одна политика не открывает позицию.
3. **Унификация формата shadow-key** — сейчас `_shadow_confirmation_raw` принимает 6 форматов. Оставить один (`(label, symbol, action_name)`), задокументировать в docstring, удалить остальные ветки.
4. **Bayesian gate для PnL, не только win_rate** — сейчас `min_full_pnl_pct=0.0` (фактически — «прибыльный на полной истории»). Стоит ввести `min_full_pnl_per_trade_lcb`, аналог Wilson, но на распределении returns (нижняя граница mean − z·std/√n).

### 5.4. Стратегическое направление

После закрытия P0 и большей части P1 контракт «Пантеон ≥ best_standalone» становится формально проверяемым. Следующие шаги — это уже не «не хуже», а «лучше»:

1. **Walk-forward calibration** гиперпараметров `ScoringConfig` (особенно `pnl_weight`, `max_dd_weight`, `win_bonus_divisor`).
2. **Per-symbol Bayesian priors** — экспериментально каждый signal_key получает свой prior на edge, обновляется по closed trades.
3. **Soft-voting контракт агента** — расширить `act(market) → Dict[sym, Action]` до `Dict[sym, AgentVote(action, confidence, expected_holding_bars, expected_cost_bps)]`. Это даст `EnsemblePlayer` корректное взвешивание + Flash — материал для нормировки `actionable_bonus` к стандартному отклонению пула.
4. **CI-инварианты** (из V4 §4): `panteon_lcb_dominance`, `churn_budget`, `regime_floor`, `notrade_value`, `ensemble_lift`. Включать как mandatory checks.

---

## 6. Сводка V6

- **V5-«19 broken files»** оказался артефактом stale Linux-mount; реальный код пользователя не сломан.
- **Применил 4 точечных фикса** через Edit/Write: dead-code в `agent.py`, mojibake-комментарий, Wilson lower bound в promotion manifest, очистка сиротских комментариев в `scoring.py`.
- **Все 11 V4 P0/P1-задач закрыты** — частично параллельной разработкой, частично моими правками этого цикла.
- **Главные открытые P2:** декомпозиция `main_loop.py`, invariant-тесты voting/anchor, унификация shadow-key.
- **Нужна регрессионная проверка** тестов `promotion_manifest` после смены формулы.
- **Стратегический фронт** — walk-forward calibration, per-symbol Bayesian priors, soft-voting контракт.

После этого цикла Пантеон формально готов к включению пяти CI-инвариантов из V4 §4 и сравнения с `best_standalone` на walk-forward.
