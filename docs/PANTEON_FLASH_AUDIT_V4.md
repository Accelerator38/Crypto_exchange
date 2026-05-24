# Panteon Flash — четвёртый аудит и точный список изменений

Дата: 2026-05-21 (re-audit относительно `PANTEON_FLASH_AUDIT_V3.md`).

Метод: чтение текущих версий ключевых файлов, выявление новых механизмов и оставшихся дефектов. Документ организован как action-list: каждый пункт — конкретное место, конкретное изменение.

Сводка размеров (для контекста дрейфа):

| Файл | Сейчас | V3 | V2 |
| --- | ---: | ---: | ---: |
| `selection/flash_allocator.py` | 755 | 769 | 789 |
| `app/main_loop.py` | 3 630 | 3 618 | 3 629 |
| `memory/performance.py` | 583 | — | — |
| `memory/degradation.py` | 238 | — | — |
| `domain/types.py` | 361 | 361 | — |

Flash allocator уменьшился за счёт упрощения логики shadow fallback и переноса вспомогательных функций; main_loop вырос за счёт новых обвязок для anchor / position-state / actor-regime degradation.

---

## 1. Что закрыто между V3 и V4

Все семь P0-задач V3 закрыты. Это второй продуктивный цикл подряд.

| V3-пункт | Где исправлено | Что сделано |
| --- | --- | --- |
| V3-3.1: гейт `<` при confidence=0 | `flash_allocator.py:682` | `gate_score <= self._min_score_to_trade_for(output)` (строгое неравенство) |
| V3-3.2: `actionable_bonus` без масштабирования | `flash_allocator.py:572-573` | `effective_score += self._config.actionable_bonus * regime_confidence_scale` |
| V3-3.3: `previous_actor` теряется при cap | `main_loop.py:940-953,1645-1660` | `_flash_actor_key_from_decision(..., prefer_original=True)` |
| V3-3.5: full_open_unconfirmed без base-fallback | `flash_allocator.py:628-638` | добавлен `and not shadow_confirmed_by_base` |
| V3-3.6: `_metrics_delta` без нормализации | `memory/degradation.py:217-223` | новая нормализация `wins`/`losses` к `closed_trades` |
| V3-3.8: docstring `pnl_net_pct` | `domain/types.py:337` | теперь явно сказано, что `pnl_pct` уже net |
| V3-4.1.6: пресеты + invariant-тест | `flash_allocator.py:203-239`, тесты `test_flash_allocator.py:512-522` | `FLASH_EXPERIMENTAL_FLAGS`, `FLASH_PRESET_SAFE/DEFAULT/AGGRESSIVE`, инвариант на флаги |

Помимо P0, закрыты ещё четыре пункта:

| Источник | Что закрыто | Где |
| --- | --- | --- |
| первый отчёт 1.7 | детерминированный tie-breaker без bias | `flash_allocator.py:721-728,1431-1434` (`blake2b` от `label|actor_type|bar`) |
| первый отчёт 1.10 | `NoTradePlayer` стал первоклассным кандидатом | `flash_allocator.py:1238-1251` (`_no_trade_outputs`), `flash_allocator.py:601-602` (`reason="eligible_no_trade"`), `flash_allocator.py:847-849` (NoTrade `_score_actor → 0.0`) |
| V2-2.5 | audit-событие подавления raw агентов | `flash_allocator.py:603-608` (`reason="raw_suppressed_by_solo"`) |
| V2-2.7 (частично) | per-regime degradation | `flash_allocator.py:87-88`, `main_loop.py:1538-1551,1591-1615` (`degradation_actor_scope`, `degradation_actor_cooldown_bars`) |

Новые механизмы, привнесённые этим циклом:

- **anchor-система** (`anchor_actor_keys`, `anchor_min_score_to_trade`, `anchor_shadow_min_score`, `anchor_min_score_advantage`) с отдельным гейтом и dominance-логикой;
- **`_no_trade_outputs`** — синтетический NoTrade кандидат на каждом баре, ranked вместе с реальными;
- **stale_close / duplicate_open guards** в FlashAllocator через `open_position_sides_by_symbol`;
- **per-regime actor degradation** (scope `actor_regime`) с cooldown bars;
- **hash-based tiebreak** через blake2b.

---

## 2. Точный список того, что надо менять сейчас

Каждый пункт — это: место в коде, проблема, конкретный фикс. Сгруппировано по приоритетам.

### P0 (баги/регрессии текущего цикла, мелкие правки)

#### 2.1. `previous_actor_by_symbol` теперь врёт при anchor / switch_margin holds

**Где:** `main_loop.py:940-953` + `flash_allocator.py:743-772`.

**Проблема:** маппа `_flash_previous_actor_by_symbol` строится с `prefer_original=True`, то есть берёт `decision.original_selected_actor`. Это правильно для cap-обрезки (исходный актор выбран, но сигнал стёрт). Но для `anchor_dominance_hold` и `switch_margin_hold` реально торгует НЕ original_selected_actor, а selected (anchor или previous). На следующем баре switch_margin сравнит score с «фантомным предыдущим», которого в позиции на бирже нет. Это:

- ломает интент switch_margin (сохранить континуитет торгующего актора);
- создаёт лишние «ложные» удержания (switch_margin_hold для актора, который не торгует);
- усложняет диагностику reason chain (`original_selected → ?`).

**Фикс:** в `main_loop.py:940-953` использовать **актор, который реально торговал**, и переключаться на `original` только если signal был отменён (cap/stale_close/duplicate_open/missing_price/no_eligible_actor).

Конкретно:

```python
# было
_flash_actor_key_from_decision(decision, prefer_original=True)
# нужно
_flash_actor_key_from_decision(
    decision,
    prefer_original=(decision.signal is None and decision.original_selected_actor),
)
```

Затем `_flash_actor_key_from_decision` должен принимать `prefer_original: bool` и применять его условно (сейчас он уже принимает; нужно передавать аргумент условно).

**Дополнительно:** удалить из вычитки в `flash_allocator._decide_symbol:759-761`:

```python
row.actor_key == previous_actor_key
or row.label == previous_actor_key  # dead branch, см. ниже
```

`row.label == previous_actor_key` никогда не срабатывает, потому что `previous_actor_key` — это всегда формат `"type:label"` (или `"NoTrade"`), а `row.label` — голая метка. Удалить ветку.

#### 2.2. Anchor + switch_margin: в `FlashDecision.reason` теряется промежуточный шаг

**Где:** `flash_allocator.py:743-772`.

**Проблема:** последовательность три:
1. greedy выбор (top-of-rank) → `selected`, `selected_reason="selected"`.
2. anchor takeover, если он есть → `selected = anchor`, `selected_reason="anchor_dominance_hold"`.
3. switch_margin hold, если он применим → `selected = previous`, `selected_reason="switch_margin_hold"`.

При прохождении всех трёх шагов в FlashDecision сохраняется только финальный `reason`. То есть нельзя по логам понять, был ли anchor takeover до того, как switch_margin вернул previous. Audit-trail обрывается.

**Фикс:** заменить `selected_reason: str` на `selected_reasons: tuple[str, ...]` в FlashDecision (новое поле, дополнительное к `reason` для обратной совместимости):

```python
selected_reasons: Tuple[str, ...] = ()
```

И в `_decide_symbol`:

```python
selected_reasons: List[str] = ["selected"]
anchor = self._anchor_dominance_candidate(ranked, selected)
if anchor is not None:
    selected = anchor
    selected_reasons.append("anchor_dominance_hold")
...
if (previous is not None and ...):
    selected = previous
    selected_reasons.append("switch_margin_hold")
```

В FlashDecision: `reason = selected_reasons[-1]`, `selected_reasons = tuple(selected_reasons)`. Audit будет полным.

#### 2.3. `FLASH_PRESET_SAFE` ≡ `FLASH_PRESET_DEFAULT`

**Где:** `flash_allocator.py:221-222`.

**Проблема:** обе константы — `FlashAllocatorConfig()`, идентичные по полям. Тест `test_flash_presets_keep_safe_default_and_aggressive_invariants` (`test_flash_allocator.py:512-522`) проверяет, что:

- `DEFAULT == FlashAllocatorConfig()` (то есть это default-конструктор);
- `SAFE` имеет experimental flags = False;
- `AGGRESSIVE` имеет все experimental flags = True.

Семантически `SAFE` должен означать «доказанно безопасный для production», а `DEFAULT` — «дефолт конструктора». Сейчас они совпадают, что вводит в заблуждение: если в `DEFAULT` добавят какой-нибудь innocent параметр (без experimental flag), SAFE тоже автоматически его получит — это не то, чего хочется от «SAFE».

**Фикс:** явно зафиксировать `FLASH_PRESET_SAFE` через kwargs (даже если эти значения совпадают с default):

```python
FLASH_PRESET_SAFE = FlashAllocatorConfig(
    min_score_to_trade=0.0,
    min_closed_trades_to_trade=3,
    min_pnl_pct_to_trade=0.0,
    actor_switch_margin=0.0,
    overextension_lookback_bars=12,
    degradation_window_closed_trades=3,
    degradation_min_closed_trades=3,
    degradation_max_recent_pnl_usd=-25.0,
)
```

И добавить invariant-тест: `FLASH_PRESET_SAFE != FLASH_PRESET_DEFAULT` (либо `FLASH_PRESET_SAFE.<x> != ...`), чтобы будущее изменение default не утекало в SAFE.

#### 2.4. `_metrics_delta` всё ещё не пробрасывает net-PnL поля

**Где:** `memory/degradation.py:217-238`.

**Проблема:** `_metrics_delta` возвращает `Metrics` без `pnl_gross_pct`, `fee_pct`, `funding_pct` (они приходят дефолтным `0.0`). Сейчас DegradationGate их не использует, но любой отчёт, построенный поверх session-metrics, увидит `pnl_gross_pct=0` при ненулевом `pnl_pct`. Это противоречит инварианту «pnl_pct = pnl_gross - fee - funding» и может вводить в заблуждение reporting layer.

**Фикс:** добавить три строки в `_metrics_delta`:

```python
return Metrics(
    ...
    pnl_gross_pct=float(current.pnl_gross_pct) - float(baseline.pnl_gross_pct),
    fee_pct=max(0.0, float(current.fee_pct) - float(baseline.fee_pct)),
    funding_pct=float(current.funding_pct) - float(baseline.funding_pct),
)
```

`fee_pct` — `max(0, ...)` потому что комиссии монотонно растут. `funding_pct` может быть отрицательным (rebate), поэтому без `max`.

### P1 (структурные слабые места, остающиеся открытыми)

#### 2.5. `EnsemblePlayer.last_vote_errors` всё ещё мутирует не-frozen dataclass

**Где:** `selection/player.py:91-115,156-159`.

**Проблема:** поле `last_vote_errors: List[VoteError] = field(default_factory=list, init=False, repr=False)` мутируется внутри `vote()`. Класс `@dataclass` (не frozen). Из-за этого:

- два потока на одном экземпляре => гонка по `last_vote_errors`;
- Player перестаёт быть pure (любая запись в state ломает «один и тот же market → один и тот же ответ» в условиях concurrent vote);
- сложнее тестировать (`vote()` возвращает signals, а errors нужно читать отдельно).

**Фикс:** изменить контракт `Player.vote()` на возврат `(List[Signal], List[VoteError])`:

```python
def vote(
    self,
    market: MarketSnapshot,
    *,
    signal_id_start: int,
) -> Tuple[List[Signal], List[VoteError]]:
    ...
```

В каждом вызывающем месте (main_loop, FlashAllocator._player_outputs) — распаковывать tuple. Удалить поле `last_vote_errors` из EnsemblePlayer и RotatingAgentPlayer. `vote_error_count` в StepResult читать из возвращаемого tuple, не из state Player.

**Альтернатива (минимально инвазивная):** оставить интерфейс, но сделать Player frozen и хранить errors в thread-local context object, передаваемом параметром. Менее чисто, но меньше изменений в caller-coupling.

#### 2.6. AgentRegistry не thread-safe

**Где:** `selection/agent.py:49-103`.

**Проблема:** `_agents: Dict[str, Agent]` мутируется в `register/unregister`, читается в `decide()`. Если bootstrap/hot-swap агента запустится параллельно с main loop — гонка. Сейчас в production это не наблюдается, потому что регистрация делается на старте, но любая live promotion флаги или dynamic loading сразу попадёт в проблему.

**Фикс:** одна из двух опций.

A. **Минимальная:** добавить `threading.RLock` в `AgentRegistry.__init__` и обернуть `register/unregister/get/has/all_labels/all_agents` в `with self._lock:`.

B. **Чище:** сделать `AgentRegistry` иммутабельным; на любую модификацию создавать новый экземпляр (copy-on-write). Главный consumer (FlashAllocator) принимает snapshot регистрации.

Вариант A быстрее, B сильнее.

#### 2.7. Promotion manifest — фиксированные пороги без статистической значимости

**Где:** `selection/promotion_manifest.py:11-86`.

**Проблема:** `PromotionManifestConfig` имеет `min_full_closed_trades=50`, `min_win_rate_pct=52.0`. Это пороги без учёта шума выборки: на 50 сделках стандартное отклонение win_rate ≈ 7%, то есть актор с истинным win_rate=49% имеет ~33% шанс пройти. Promotion получает не «доказанно прибыльные» сигналы, а «достаточно везучие».

**Фикс:** заменить win_rate-гейт на **Lower Confidence Bound (LCB)** Beta-апостериора:

```python
import math

def beta_lcb(wins: int, losses: int, z: float = 1.645) -> float:
    """One-sided Wilson lower bound на win_rate."""
    n = wins + losses
    if n == 0:
        return 0.0
    p_hat = wins / n
    z2 = z * z
    denom = 1 + z2 / n
    centre = p_hat + z2 / (2 * n)
    margin = z * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))
    return max(0.0, (centre - margin) / denom)
```

В `_promotion_rejection_reason`:

```python
wins = _as_int(row.get("wins"))
losses = _as_int(row.get("losses"))
win_rate_lcb = beta_lcb(wins, losses) * 100.0
if win_rate_lcb < config.min_win_rate_lcb_pct:
    return "win_rate_lcb_below_gate"
```

В `PromotionManifestConfig` заменить `min_win_rate_pct: float = 52.0` на `min_win_rate_lcb_pct: float = 50.0`. Эффект: маленькие выборки больше не пройдут, даже если эмпирический win_rate = 60%.

То же для `min_full_pnl_pct` — использовать `pnl_per_trade - z * std(returns) / sqrt(n)` как нижнюю границу.

#### 2.8. `regime_score` игнорирует параметр `regime`

**Где:** `scoring/scoring.py:99-161`.

**Проблема:** параметр принят, но в формуле не используется. Это значит, что любой per-regime тюнинг сейчас невозможен без переписывания функции.

**Фикс:** добавить опциональный `Dict[Regime, ScoringConfig]`:

```python
def regime_score(
    metrics: Metrics,
    regime: Regime,
    *,
    config: ScoringConfig = DEFAULT_SCORING,
    per_regime_configs: Optional[Mapping[Regime, ScoringConfig]] = None,
    confidence: float = -1.0,
) -> float:
    cfg = (per_regime_configs or {}).get(regime, config)
    ...
```

В FlashAllocator передавать `per_regime_configs` из `self._scoring_per_regime`. Это потребует обновления `__init__` FlashAllocator:

```python
def __init__(
    self,
    *,
    perf: PerformanceMemory,
    qm: QuarantineManager,
    config: FlashAllocatorConfig = FlashAllocatorConfig(),
    scoring_config: ScoringConfig = DEFAULT_SCORING,
    scoring_per_regime: Optional[Mapping[Regime, ScoringConfig]] = None,
) -> None:
```

Эффект: можно дать `CRASH` пониженный `max_dd_weight` (потому что просадки в crash — нормально), `BULLISH` — повышенный `pnl_weight`, и т. д.

#### 2.9. Score теряет монотонность при `closed_trades < 2`

**Где:** `scoring/scoring.py:131-133`.

**Проблема:**

```python
if metrics.closed_trades < 2:
    return metrics.pnl_pct * 0.40 * confidence
```

`confidence` при `closed=0` → 0. `confidence` при `closed=1` → 1/6 ≈ 0.167. То есть один закрытый агент с +5% PnL получает score ≈ 0.33, а агент без закрытых, но с положительным `pnl_pct` (что технически возможно через `_component_score` для ensemble без has_data) — получит 0. Кривая немонотонна на стыке.

**Фикс:** убрать short-circuit и использовать единую формулу с `confidence`:

```python
def regime_score(metrics, regime, *, config=DEFAULT_SCORING, confidence=-1.0):
    if not metrics.has_data:
        return 0.0
    if confidence < 0:
        confidence = confidence_from_sample(metrics.closed_trades, config=config)
    pnl_component = metrics.pnl_pct * config.pnl_weight
    sharpe_component = metrics.sharpe * config.sharpe_weight if metrics.closed_trades >= 2 else 0.0
    dd_component = -abs(metrics.max_dd_pct) * config.max_dd_weight
    activity_component = min(metrics.signals, config.activity_signals_cap) * config.activity_weight
    win_bonus = (metrics.win_rate - 50.0) / config.win_bonus_divisor if metrics.closed_trades >= 3 else 0.0
    score = (pnl_component + sharpe_component + dd_component + activity_component + win_bonus) * confidence
    if metrics.closed_trades == 0 and metrics.signals == 0 and metrics.entries == 0:
        score -= config.inactivity_penalty
    return float(score)
```

Главное: убран `if closed_trades < 2: return pnl_pct * 0.40 * confidence`. Теперь формула одна, и она монотонна.

#### 2.10. Solo-safety check работает на wrapper-уровне и удаляет агента из любых ensemble

**Где:** `main_loop.py:1083-1091` (см. V1-audit 1.18).

**Проблема:** каждый агент обёрнут в Solo-wrapper и проверен через `_fallback_candidate_safety_issue`. Если safety drop сработал для Solo-обёртки, агент удаляется из пула. Но Solo-обёртка имеет другую статистику, чем агент в составе ensemble. Здоровый агент в ensemble может пропасть из-за «нездоровой» solo-обёртки.

**Фикс:** перенести safety-check на уровень агента или ensemble, а не wrapper-а:

```python
def _agent_is_real_executable(agent: object) -> bool:
    if bool(getattr(agent, "shadow_only", False)):
        return False
    if getattr(agent, "live_trading_eligible", True) is False:
        return False
    if _agent_health_check_failed(agent):
        return False
    return True
```

И в `_flash_real_agents`:

```python
for agent in all_agents():
    if not _agent_is_real_executable(agent):
        continue
    out.append(agent)
```

Без обёртки в Solo-кандидата для safety-check.

#### 2.11. `actionable_bonus` всё ещё аддитивный и не калиброванный

**Где:** `flash_allocator.py:572-573`.

**Проблема:** даже с масштабированием на regime_confidence, `actionable_bonus=0.25` остаётся аддитивным. Когда дисперсия `gate_score` по пулу порядка 0.05 (маленькие edge differences), bonus = 0.25 доминирует ранжирование. Когда дисперсия 5.0 — bonus = шум. То есть тот же параметр в разных условиях имеет разный эффект.

**Фикс:** заменить аддитивный bonus на мультипликативный:

```python
if output.label in actionable:
    effective_score *= (1.0 + self._config.actionable_bonus * regime_confidence_scale)
```

С `actionable_bonus=0.25` это означает «+25% к score». Этот множитель пропорционален текущему score, что устраняет шкалозависимость.

**Альтернатива:** нормировать к стандартному отклонению score по пулу кандидатов того же бара. Но это усложняет логику и зависит от состава пула.

### P2 (большие структурные долги)

#### 2.12. `main_loop.py` всё ещё монолит на 3 630 строк

**Где:** `app/main_loop.py`.

**Проблема:** четыре независимых пути (real, shadow, flash, fallback) живут в одном файле, плюс всё обвязочное (degradation tracking, audit emission, position state, switching и т. п.). Любая правка одного фильтра требует ревизии всех веток.

**Фикс:** декомпозиция по модулям:

- `app/decision_paths/flash.py` — `_run_flash_decision_path`, `_flash_*` helpers;
- `app/decision_paths/legacy_v2.py` — strategist-based path;
- `app/decision_paths/fallback.py` — fallback attempts;
- `app/degradation_tracking.py` — `_flash_update_degradation_state_from_events`, recovery, actor scope;
- `app/position_state.py` — `_flash_open_position_sides_by_symbol`, sync helpers;
- `app/audit_emission.py` — `_emit_flash_audit_events`, `_causal_decision_payload`.

`main_loop.py` становится тонким оркестратором (200-300 строк).

#### 2.13. Voting policy без invariant-тестов

**Где:** `selection/voting.py`, `tests/test_voting.py` (если есть).

**Проблема:** WeightedConsensus, StrongConsensus, RiskParity — три политики без формальных инвариантов.

**Фикс:** добавить набор инвариант-тестов:

```python
def test_strong_consensus_subset_of_weighted_consensus_for_opens(self):
    """StrongConsensus opens ⊆ WeightedConsensus opens при равных thresholds."""
    ...

def test_risk_parity_equals_weighted_consensus_with_equal_vol(self):
    """RiskParity с одинаковыми vol эквивалентен WeightedConsensus с равными весами."""
    ...

def test_no_open_when_directions_conflict_in_strong(self):
    ...
```

#### 2.14. Anchor + switch_margin не имеют формальной семантики приоритетов

**Где:** `flash_allocator.py:743-772`.

**Проблема:** anchor takeover применяется ПЕРЕД switch_margin. Это значит, что anchor может быть «отменён» switch_margin-ом. Документация молчит о приоритете.

**Фикс:** определить и протестировать формальную семантику. Например: anchor — это «soft preference», switch_margin — «жёсткий continuity», и switch_margin побеждает. Либо наоборот. Закрепить тестом:

```python
def test_anchor_dominance_yields_to_switch_margin_when_both_apply(self):
    """При одновременных anchor takeover и switch_margin hold, switch_margin выигрывает."""
    ...

def test_anchor_dominance_overrides_greedy_when_no_previous(self):
    ...
```

#### 2.15. Shadow score keys — три формата одновременно

**Где:** `flash_allocator.py:1158-1193` (`_shadow_confirmation_for`) + `flash_allocator.py:1495+` (`_shadow_confirmation_raw`).

**Проблема:** allocator принимает shadow_confirmation в шести вариантах ключей. Это «совместимость», но реально источник тонких багов: при одном изменении формата на caller-стороне поведение может молча измениться (одно из 6 совпадений).

**Фикс:** жёстко принимать только формат `(label, symbol, action_name)` (tuple, action как `Action.name`). Удалить остальные варианты. Caller-сторона обязана нормализовать. Документировать в docstring.

---

## 3. Приоритезированная очередь (для следующего цикла)

| # | Что сделать | Файлы | Размер | Категория |
| --- | --- | --- | --- | --- |
| 1 | `previous_actor_by_symbol` хранит фактически торгующего | `main_loop.py:940-953,1645-1660`, `flash_allocator.py:759-761` | XS | P0 |
| 2 | `selected_reasons` цепочкой в FlashDecision | `flash_allocator.py:317-341,743-772` | S | P0 |
| 3 | `FLASH_PRESET_SAFE` явно через kwargs + invariant | `flash_allocator.py:221-222`, `test_flash_allocator.py:512-522` | XS | P0 |
| 4 | `_metrics_delta` пробрасывает net-PnL поля | `memory/degradation.py:217-238` | XS | P0 |
| 5 | `EnsemblePlayer.vote()` → `(signals, errors)` | `selection/player.py:91-220,247-380`, все caller’ы | M | P1 |
| 6 | `AgentRegistry` с lock | `selection/agent.py:49-103` | S | P1 |
| 7 | Bayesian LCB в promotion manifest | `selection/promotion_manifest.py:11-86`, тесты | M | P1 |
| 8 | `regime_score` поддерживает `per_regime_configs` | `scoring/scoring.py:99-161`, `flash_allocator.py:_score_actor` | M | P1 |
| 9 | `regime_score` без short-circuit при `closed<2` | `scoring/scoring.py:131-133` | XS | P1 |
| 10 | Safety-check агента, не Solo-обёртки | `main_loop.py:1083-1091`, `_flash_real_agents` | S | P1 |
| 11 | `actionable_bonus` мультипликативный | `flash_allocator.py:572-573` | XS | P1 |
| 12 | Декомпозиция `main_loop.py` | весь файл | L | P2 |
| 13 | Invariant-тесты voting policies | `tests/test_voting.py` (новый) | S | P2 |
| 14 | Anchor + switch_margin формальные тесты | `test_flash_allocator.py` | S | P2 |
| 15 | Унификация shadow-key формата | `flash_allocator.py:1158-1193,1495+` | M | P2 |

P0-блок (#1-#4) — XS/S задачи, дающие чистый audit-trail, корректный hysteresis и invariant-защиту preset’ов. Их можно закрыть в один pull request.

---

## 4. Контракт «Пантеон не хуже своих составляющих»

После закрытия P0 #1-#4 и P1 #5-#11 контракт становится **проверяемым в CI**. Пять инвариант-тестов:

| Тест | Условие | Где запускать |
| --- | --- | --- |
| `panteon_lcb_dominance` | `equity_net(Panteon) >= 0.95 * equity_net(best_standalone)` | walk-forward |
| `churn_budget` | `avg_actor_switches_per_day(Panteon) <= 2 * best_standalone_switches` | walk-forward |
| `regime_floor` | `pnl_net(regime) >= 0` для каждого `regime ∈ {BULLISH, BEARISH, NEUTRAL, CRASH}` | OOS |
| `notrade_value` | `equity(min_score=current) > equity(min_score=-inf)` | OOS |
| `ensemble_lift` | `pnl_net(ensemble) - pnl_net(best_member) >= 0` | OOS, per ensemble |

Когда тест не проходит — фиксировать как known gap с привязкой к причине (например, regime_floor падает в CRASH → нужен per-regime ScoringConfig из #8).

---

## 5. Сводка изменений по сравнению с V3

- Закрыто 11 пунктов (все 7 P0 из V3 + 4 структурных из первого отчёта).
- Открыто 15 точечных правок (P0×4, P1×7, P2×4).
- Новые мощные механизмы (anchor, NoTrade-as-candidate, position-state guards, per-regime degradation) интегрированы корректно, но создали 3 новые мини-проблемы (previous_actor врёт на holds, audit-trail обрывается, FLASH_PRESET_SAFE неявный) — они в P0.
- Контракт «Пантеон ≥ best_standalone» становится формально проверяемым после закрытия P0 текущего цикла.

Следующий разумный шаг — выполнить P0 (1-4), затем P1 (5, 6, 9, 11) как «гигиенический» цикл, после чего включать пять CI-тестов из раздела 4. После них переходить к Bayesian promotion (#7) и per-regime scoring (#8) — это уже шаги к «лучше», а не к «не хуже».
