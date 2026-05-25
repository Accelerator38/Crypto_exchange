# Сравнение агентов: текущий 5-летний прогон vs legacy v1

Дата: 2026-05-24. Источник: `Results/PanteonFlashCooldown720ActorRegime72Round3Deny8_Full2022_2026_20260523/.../2026-05-23_17-49-26_retrodate_market_v2/` (5 лет, 2022-2026, 38 375 баров).

---

## 1. Текущий прогон — что есть, как работает

### 1.1. Зарегистрированные агенты Panteon Flash (19 шт.)

5-летняя статистика, отсортированная по PnL:

| # | Агент | PnL % | Закр. | Win% | Sharpe | MaxDD |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | PlayerFunding | **+203.71** | 10 413 | 42.05 | 0.021 | 70.10 |
| 2 | LiveOIBreakout | **+168.88** | 5 809 | 38.29 | 0.024 | 92.67 |
| 3 | MomentumScalper | +78.12 | 13 178 | 39.50 | 0.009 | 94.35 |
| 4 | ResearchValidatorAgent | +73.05 | 2 293 | 35.41 | 0.028 | 59.09 |
| 5 | **VolBreakoutHunter** | +52.79 | 893 | 38.86 | **0.080** | **18.01** |
| 6 | RichardDennis | +16.83 | 1 240 | 22.74 | 0.014 | 24.33 |
| 7 | NeutralRangeScalper | +0.13 | 13 | 61.54 | 0.057 | 0.43 |
| 8 | NeutralLiquiditySweep | -0.12 | 56 | 41.07 | -0.010 | 1.96 |
| 9 | BearReliefFadeAgent | -2.06 | 50 | 32.00 | -0.103 | 4.57 |
| 10 | BullRotationAgent | -2.88 | 293 | 36.18 | -0.025 | 10.09 |
| 11 | CrashPanicShortAgent | -9.62 | 583 | 33.79 | -0.044 | 23.04 |
| 12 | LiveVolCompress | -19.34 | 2 817 | 40.58 | -0.042 | 33.13 |
| 13 | LiveRegimePullback | -23.07 | 2 408 | 37.38 | -0.015 | 59.53 |
| 14 | LiveTrendFollow | -31.45 | 4 196 | 41.44 | -0.013 | 59.52 |
| 15 | FundingArb | -34.83 | 6 033 | 42.50 | -0.008 | 55.83 |
| 16 | LiveCrashHunter | -80.27 | 3 833 | 45.13 | -0.017 | 78.61 |
| 17 | LiveMeanRev | -94.15 | 3 216 | 51.59 | -0.069 | 69.75 |
| 18 | **LiveAfterShock** | **-223.76** | 11 130 | 43.67 | -0.046 | 94.13 |

(Зарегистрированы ещё `AnchorFlowMomentum` — 0 сделок и пара других неактивных.)

Ключевые наблюдения:

1. **Топ-2 (PlayerFunding, LiveOIBreakout)** дают +372% в сумме, но имеют MaxDD 70-92% — слишком волатильные для прямого использования без size-нормализации.
2. **VolBreakoutHunter** — единственный «здоровый» агент: Sharpe 0.080 (лучший среди тех, у кого > 100 трейдов), MaxDD 18%, PnL 52.79%. Это лучший standalone-кандидат.
3. **LiveAfterShock** — -223% катастрофа, но в Perfect-Oracle он выбран лучшим в 3 месяцах из 53 (2022-10, 2023-02, 2023-07, 2026-03). Это значит: edge существует, но утопает в шуме большую часть времени.
4. **LiveCrashHunter** — суммарно -80%, но в 2022-02 (реальный crash) дал +11.59% с 78% win rate. Тот же сюжет: edge в нужном режиме, гарантированный убыток в неподходящем.

### 1.2. Игроки и Perfect Panteon Oracle

53 месяца × выбранный «оптимальный» актор:

| Игрок/актор | Раз выбран | Кумулятивный PnL $ | Особенности |
| --- | ---: | ---: | --- |
| `Perfect_OIBreakout` | 11 | ~860 | bullish/bearish/crash/neutral all → LiveOIBreakout (single-agent) |
| `Antonius_conservative` | 4 | ~700 | bull→VolBreakout, bear→FundingArb, neutral→Research, crash→CrashPanicShort |
| `Solo_LiveAfterShock` | 4 | ~240 | -223% standalone, но в редких месяцах — лучший |
| `TrendResearch` | 4 | ~265 | Тренд-профиль ансамбля |
| `NeutralEdgeResearch` | 3 | ~748 (один месяц 587!) | Нейтральный edge |
| `Perfect_CrashSwitch` | 3 | ~311 | bearish/crash → LiveCrashHunter |
| `Solo_MomentumScalper` | 3 | ~135 | Раз/месяц |
| `Solo_LiveTrendFollow` | 3 | ~152 | -31% standalone, в правильном месяце +5-6% |
| `Solo_LiveOIBreakout` | 3 | ~486 | Дублирует Perfect_OIBreakout |
| `Solo_FundingArb` | 2 | ~271 | -34% standalone, но в 2024-03 +23.5% |
| `Solo_PlayerFunding` | 2 | ~360 | В январе ⇒ декабре 2024 |
| `Solo_LiveCrashHunter` | 1 | ~116 | Только 2022-02 |
| `MeanRevResearch` | 2 | ~60 | Нейтральный режим |
| `DefensiveResearch` | 2 | ~314 | bull/crash defensive |
| `BombermanStrong` | 1 | ~7 | 100% win rate, 1 трейд |
| `Optimal_StaticRotator` | 1 | ~72 | 190 трейдов |

**Perfect Panteon достижим:** $5400 за 5 лет (540%) при 1.99% MaxDD и 59% win rate. Это потолок при идеальном monthly switching.

**Текущий Panteon Flash:** $1054 (105%) — то есть **19.5% от потенциала Perfect Oracle**. Beats best_standalone (Antonius_conservative 62%) на +43%.

### 1.3. Главная проблема селекции — `standalone vs flash_selected`

```
Solo_MomentumScalper: standalone 58.92%, Flash-selected 27.66%, alpha -31.26% (521 раз выбран)
LiveOIBreakout:       standalone  9.28%, Flash-selected 11.29%, alpha  +2.00% (120 раз)
```

То есть **Flash при выборе MomentumScalper отбирает плохие моменты** — недополучает -31% от потенциала актора. С LiveOIBreakout наоборот, отбирает чуть лучше среднего.

Это и есть прямая иллюстрация того, что говорит V8: selection layer работает, но **economic layer (sizing/timing) отсутствует**.

---

## 2. Что есть в legacy v1 и не используется в Flash

### 2.1. Агенты v1, не зарегистрированные в `agent_bootstrap.py`

В `src/panteon_runtime/panteon_agents.py` нашёл агентов, которых нет в KNOWN_V1_AGENTS:

| Класс | Назначение | Почему стоит вернуть |
| --- | --- | --- |
| **`CarryFlowAgentV2`** | funding + crowding + OI expansion с safe fallback | Логически близок к `PlayerFunding` (+203% лидер), но с лучшим fallback и более узкой триггерной зоной. На fundingrich-парах может давать дополнительный edge. |
| **`CandlePatternAgent`** | 6 свечных паттернов (Engulfing, Hammer, Star) с trend-confirmation | Это **completely orthogonal сигнал** к существующим momentum/funding/breakout-агентам. Низкая корреляция = хороший ensemble-кандидат. |
| **`ExternalSignalAgent`** | Импорт сигналов из файла (внешний анализ) | Подходит для подключения discretionary-сигналов или ML-моделей в shadow mode. |
| `Bomberman` | Strong-consensus тип игрока | Уже фактически реализован через `PROFILE_BOMBERMAN_STRONG` в Flash. |

### 2.2. Идеи из legacy SubPlayer (master_player.py)

В `master_player.py` есть `SubPlayer_Alpha/Beta/Gamma/Player7Ways/Ultima/Neuro` — это была старая иерархия. Полезные дизайн-идеи:

| Идея | Что значит для Flash |
| --- | --- |
| `SubPlayer_Alpha` — per-regime разный набор агентов | Уже есть как `RotatingAgentPlayer` / `Antonius_conservative`. Лидер 5-летней статистики. |
| `SubPlayer_Beta` — defensive: 3Commas DCA + GeneticsBearish + CorrBreakout | **`ThreeCommasDCA` отсутствует.** Это averaging-down стратегия с STEP_PCT/MAX_SAFETY — даёт edge в range-bound и в небольших pull-back-ах. |
| `SubPlayer_Beta` — `LOCK_TRIGGER=3%`, `LOCK_FRACTION=40%` (раннее закрытие части прибыли) | Flash не управляет позицией между open/close (см. V8 §2.8 Dynamic SL/TP). Здесь готовая логика. |
| `SubPlayer_Gamma` — aggressive: лучшие по Sharpe | Сейчас это и есть Flash-greedy. |
| Внешние агенты, использовавшиеся в SubPlayers: `DualMomentum`, `VolatilityBreakoutAgent`, `CorrBreakoutAgent`, `ThreeCommasDCA` | Не нашёл `crypto_agents.py` в текущей кодовой базе — модуль либо удалён, либо переименован. Если внутреннее IP сохранилось, **`CorrBreakoutAgent` и `ThreeCommasDCA` стоит восстановить**. |

### 2.3. Что точно стоит вернуть

**Высокий приоритет:**

1. **`CandlePatternAgent`** — единственный orthogonal-сигнал к существующим. Closed history → synthetic OHLC → 6 паттернов с trend-confirmation. Низкая корреляция с momentum/funding/OI-breakout = чистая диверсификация Sharpe.
2. **`CarryFlowAgentV2`** — улучшенная funding-логика с safe fallback. Может частично заменить волатильный `PlayerFunding` (Sharpe 0.021, MaxDD 70%).
3. **Раннее частичное закрытие позиции** (`LOCK_TRIGGER=3%`, `LOCK_FRACTION=40%`) — берётся из `SubPlayer_Beta` дизайна, но интегрируется в `executor.py`, а не в агент.

**Средний приоритет:**

4. **`ThreeCommasDCA`** — averaging-down. Не как «полный агент», а как **size-management overlay**: если позиция против тренда на N%, открыть половинную добавку (только при сильном signal_key). Это лекарство для slope-проблем VolBreakout-ов.
5. **`CorrBreakoutAgent`** — некоррелированная стратегия. Если IP сохранилось, восстановить.

**Низкий приоритет (нишевые):**

6. **`ExternalSignalAgent`** — полезно как канал для подключения ML-сигналов в shadow.

---

## 3. Идеи для новых агентов с конкретной специализацией

На основе анализа того, **где Perfect Oracle выбирал какого актора**, видны явные ниши, которые сейчас никто не закрывает или закрывает плохо.

### 3.1. Ниши, где edge доказан и где надо строить узкого специалиста

#### A. **`CrashMonthHunter`** — для CRASH-месяцев

**Данные:** Perfect Oracle выбирает `Solo_LiveCrashHunter` / `Perfect_CrashSwitch` в редких, но самых прибыльных месяцах (2022-02 +11.59%, 2022-07 +24.21%, 2023-09 +1.05%, 2025-12 +5.81%). Текущий `LiveCrashHunter` накапливает -80% за 5 лет, потому что триггерится и в не-crash месяцах.

**Конструкция:** обёртка `ActionFilterAgent(LiveCrashHunter, allowed_regimes=("crash",), allowed_actions=(FUT_SHORT_HALF, FUT_SHORT_FULL))` + **дополнительная гарантия по market regime confidence** (`market.regime_confidence >= 0.7`). Это уже частично сделано через `CrashPanicCrashOnly` wrapper, но текущий wrapper всё равно проседает -9.62%. Нужно ужесточить:

```python
ActionFilterAgent(
    label="CrashHunterStrict",
    base_agent=LiveCrashHunterAdapter(),
    allowed_actions=(Action.FUT_SHORT_FULL,),
    allowed_regimes=("crash",),
    # дополнительно: input market.lookback_returns_pct[symbol][24] <= -10%
)
```

И опубликовать его в `anchor_actor_keys` для crash-режима, чтобы при правильном triggered он получал предпочтение.

#### B. **`VolBreakoutFundingAware`** — расширение лучшего Sharpe-агента

**Данные:** VolBreakoutHunter — единственный с Sharpe > 0.05 на 893 трейдах. Sharpe ограничивается тем, что часть open’ов происходит в «случайных» окнах. Если добавить funding-фильтр (открывать long только при `funding > 0`, short только при `funding < 0`), false positive должны уйти.

**Конструкция:** wrapper над `VolBreakoutHunter`, который смотрит `market.funding[symbol]` перед каждым open’ом:

```python
class VolBreakoutFundingAware:
    def __init__(self, base):
        self.base = base
        self.label = "VolBreakoutFundingAware"
    def act(self, market):
        raw = self.base.act(market)
        for sym, action in list(raw.items()):
            funding = market.funding.get(sym, 0.0)
            if action.is_long_open and funding >= 0:
                raw[sym] = Action.HOLD   # позитивный funding = плата за long
            elif action.is_short_open and funding <= 0:
                raw[sym] = Action.HOLD
        return raw
```

Ожидаемый эффект: Sharpe 0.08 → 0.10-0.12 (за счёт удаления неудачных open’ов с funding bleed).

#### C. **`AfterShockRegimeOnly`** — спасение LiveAfterShock

**Данные:** LiveAfterShock сейчас -223%, но в 4 месяцах был Perfect-optimal (post-shock rebound). Это значит логика правильная, **applicability слишком широкая**.

**Конструкция:** триггер ТОЛЬКО при:
- последние `lookback_returns_pct[symbol][6]` ≤ -3% (свежий shock в последние 6 баров);
- `lookback_returns_pct[symbol][24]` ≤ -5% (за 24 бара тоже минус, не shake-out);
- `regime ∈ {BEARISH, CRASH}` или `regime_confidence < 0.5` (переходный момент);
- action только `FUT_LONG_HALF` (rebound, не full).

Это сильно сократит число трейдов (с 11 130 до ~2000), но качество должно вырасти на порядок.

#### D. **`NeutralFundingHarvester`** — для NEUTRAL месяцев

**Данные:** В NEUTRAL месяцах (по Perfect Oracle) выбираются: `Solo_FundingArb` (2024-03 +23.5%, 2022-12 +3.55%), `Solo_PlayerFunding` (2024-01 +12.45%, 2024-12 +23.59%), `NeutralEdgeResearch` (2024-11 +58.77%!!). При NEUTRAL рынке funding-стратегии собирают carry без направленного риска.

**Конструкция:** **delta-neutral funding harvester** — открывает long spot + short perp на парах с positive funding (или наоборот при negative). Это классический cash-and-carry.

```python
class DeltaNeutralFundingHarvester:
    MIN_FUNDING_BPS = 0.5  # >0.05% bps
    def act(self, market):
        if market.regime != Regime.NEUTRAL:
            return {}
        actions = {}
        for sym, funding in market.funding.items():
            if funding > self.MIN_FUNDING_BPS / 10000:
                # positive funding → long spot, short perp (но это требует двух signals)
                # упрощение: SPOT_BUY_HALF, агент-партнёр откроет FUT_SHORT_HALF
                actions[sym] = Action.SPOT_BUY_HALF
            elif funding < -self.MIN_FUNDING_BPS / 10000:
                actions[sym] = Action.FUT_SHORT_HALF
        return actions
```

Понадобится parallel-агент для парного входа или composer-уровень логики «pair signals».

#### E. **`OIBreakoutSpotPullback`** — раскрытие топ-агента

**Данные:** LiveOIBreakout единолично занимает 11 месяцев Perfect Oracle и даёт +168% standalone. Но MaxDD 92.67% — entry timing проблема. Решение: **spot-only pullback variant** — после OI spike дождаться pullback к VWAP/MA20 (long entry), а не входить на пике.

**Конструкция:** wrapper, который ловит OI spike (как сейчас), но затем ждёт `lookback_returns_pct[symbol][3] ≤ -1%` перед открытием (mini-dip после spike).

#### F. **`BullishTrendOnly`** + **`BearishPanic Only`** — разделение универсалов

**Данные:** `LiveTrendFollow` -31% (универсал), `LiveRegimePullback` -23%. Оба триггерятся вне своего комфорта. Разделение через `ActionFilterAgent`:

- `BullishTrendOnly` = LiveTrendFollow ∩ {regime=BULLISH, action ∈ long_opens}
- `BearishPanicOnly` = LiveTrendFollow ∩ {regime=BEARISH, action ∈ short_opens}

Это уже частично сделано в `MomentumScalperShortBearOnly`, `MomentumScalperShortCrashOnly` — продолжить паттерн на других универсалов.

### 3.2. Совсем новые специализации

#### G. **`CorrelationDislocationAgent`** — на расхождениях BTC/альт

**Идея:** когда `BTC` рынки выросли > 2%, но альты не последовали в течение 6 баров → ожидается catch-up. Long альт-баскет.

```python
class CorrelationDislocationAgent:
    LEADER = "BTC/USDT"
    DISLOCATION_BPS = 200  # 2%
    LAG_BARS = 6
    def act(self, market):
        btc_ret = market.lookback_returns_pct.get(self.LEADER, {}).get(self.LAG_BARS, 0)
        if abs(btc_ret) < self.DISLOCATION_BPS / 100:
            return {}
        actions = {}
        for sym in market.prices:
            if sym == self.LEADER: continue
            alt_ret = market.lookback_returns_pct.get(sym, {}).get(self.LAG_BARS, 0)
            if btc_ret > 0 and alt_ret < btc_ret * 0.3:  # альт отстаёт
                actions[sym] = Action.SPOT_BUY_HALF
            elif btc_ret < 0 and alt_ret > btc_ret * 0.3:  # альт не упал
                actions[sym] = Action.FUT_SHORT_HALF
        return actions
```

Эта стратегия диверсифицирована относительно existing momentum-агентов (другой mean-reverting frame).

#### H. **`SessionReversalAgent`** — на смену торговых сессий

**Идея:** на крипто-рынке смены Asia/EU/US сессий часто сопровождаются reversal patterns. На основе `market.timestamp.hour`:

- 00 UTC (Asia open) и 14 UTC (US open) — точки потенциального reversal;
- если последние 4 бара трендили в одну сторону, на reversal time открыть half-size противоположную позицию.

Простая, но низко-коррелированная с momentum-агентами стратегия.

#### I. **`HighVolatilityFader`** — fade-стратегия для shock баров

**Идея:** когда `lookback_returns_pct[symbol][1]` ≤ -3% за 1 бар (внезапный flash crash), часто следует bounce в следующие 2-3 бара. Open `FUT_LONG_HALF`, SL -1.5%, TP +1%. Mean-reversion на узких таймфреймах.

Это **аналог** `LiveAfterShock`, но с другими параметрами окна (1 бар вместо длинного lookback).

#### J. **`CapsuleRiskParityPlayer`** — новый тип игрока

**Идея:** ensemble не на основе voting, а **на основе риск-паритета по historic-DD**. Каждый бар вес агента = 1/historic_max_dd_pct. Это автоматически отдаст низкий вес LiveAfterShock (DD 94%) и высокий — VolBreakoutHunter (DD 18%).

`RiskParity` уже существует в voting.py, но **получает volatilities через конструктор** (frozen). Нужно сделать **dynamic version**:

```python
class DynamicRiskParityPlayer(EnsemblePlayer):
    def __init__(self, ..., perf: PerformanceMemory):
        self._perf = perf
        super().__init__(...)
    def _recompute_weights(self, regime):
        weights = {}
        for label in self.agent_labels:
            m = self._perf.get(label, regime=regime)
            inv_dd = 1.0 / max(0.5, m.max_dd_pct)  # avoid div0
            weights[label] = inv_dd
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}
    def vote(self, market, *, signal_id_start):
        # use _recompute_weights перед vote
        ...
```

---

## 4. Рекомендации к внедрению

### 4.1. Что сделать в первую очередь

| # | Действие | Размер | Ожидаемый эффект |
| ---: | --- | --- | --- |
| 1 | Зарегистрировать `CandlePatternAgent` из panteon_agents.py в `KNOWN_V1_AGENTS` | XS | +5-15% Sharpe (orthogonal signal) |
| 2 | Зарегистрировать `CarryFlowAgentV2` | XS | +2-5% PnL на funding-rich парах |
| 3 | Создать wrapper `CrashHunterStrict` (3.1.A) | S | спасает -80% LiveCrashHunter |
| 4 | Создать wrapper `VolBreakoutFundingAware` (3.1.B) | S | +0.02-0.04 к Sharpe лучшего агента |
| 5 | Создать wrapper `AfterShockRegimeOnly` (3.1.C) | S | спасает -223% LiveAfterShock |

### 4.1.1. Статус внедрения 2026-05-24

- Внедрено в `src/panteon_v2/app/agent_bootstrap.py`: `CandlePatternAgent` и `CarryFlowAgentV2` добавлены в известные v1 labels, чтобы Flash мог собирать standalone/shadow-статистику.
- Ограничитель сохранен: `Solo_CarryFlowAgentV2` остается в `StrategistConfig.hard_policy_deny_labels`; регистрация не делает его production-лидером по умолчанию.
- Внедрены opt-in experimental wrappers: `CrashHunterStrict`, `VolBreakoutFundingAware`, `AfterShockRegimeOnly`, `LiveTrendFollowBullOnly`, `LiveMeanRevNeutralOnly`.
- Новые переиспользуемые gates для wrapper-ов: `min_regime_confidence`, `funding_cost_aligned_opens`, `min_lookback_return_pct_by_bars`, `max_lookback_return_pct_by_bars`.
- Осознанно отложено: `ExternalSignalAgent`, partial close на уровне executor, `DeltaNeutralFundingHarvester`, `CorrelationDislocationAgent`, `SessionReversalAgent`, `HighVolatilityFader`. Эти идеи меняют runtime/data contract и требуют отдельного walk-forward цикла.

### 4.2. Что добавить во второй итерации

| # | Действие | Размер | Эффект |
| ---: | --- | --- | --- |
| 6 | `OIBreakoutSpotPullback` wrapper | S | снизить MaxDD топ-агента LiveOIBreakout |
| 7 | `BullishTrendOnly` / `BearishPanicOnly` ActionFilters | S | спасает -31% LiveTrendFollow |
| 8 | `DeltaNeutralFundingHarvester` (новый агент) | M | +15-30% на NEUTRAL месяцах (см. NeutralEdgeResearch 587% в 2024-11) |
| 9 | `CorrelationDislocationAgent` (новый) | M | диверсификация |
| 10 | `DynamicRiskParityPlayer` (новый тип игрока) | M | автоматическое снижение веса деградирующих агентов |
| 11 | `SessionReversalAgent` (новый) | S | +1-3%, низкая корреляция |
| 12 | `HighVolatilityFader` (новый) | S | дополняет LiveAfterShock variant |

### 4.3. Возвращать ли legacy SubPlayer_*?

**Нет**, как классы — не нужно. Их роль уже занимают `PROFILE_*` ансамбли. **Но дизайн-идеи стоит изучить:**

1. `SubPlayer_Beta` LOCK_TRIGGER + LOCK_FRACTION (раннее partial close) → реализовать в `executor.py` как `Signal.partial_close_at_pnl_pct`.
2. `SubPlayer_Beta` ThreeCommasDCA → реализовать как **size-overlay**, не как агент.

---

## 5. Резюме

**Что есть в текущем прогоне:**

- 19 базовых агентов, из них только 6 в плюсе за 5 лет.
- 21 player в leaderboard, лучший — `Antonius_conservative` (62%, per-regime rotation).
- Panteon Flash достигает 105% за 5 лет, **19.5% от Perfect Oracle (540%)**.
- Главная утечка alpha: `Solo_MomentumScalper` Flash-selected отстаёт от standalone на -31%.

**Что стоит вернуть из legacy:**

- `CandlePatternAgent` (orthogonal signal).
- `CarryFlowAgentV2` (funding с safe fallback).
- Идею раннего partial close из `SubPlayer_Beta` — реализовать на уровне executor.

**Какие новые специализации стоит создать:**

- `CrashHunterStrict`, `VolBreakoutFundingAware`, `AfterShockRegimeOnly`, `OIBreakoutSpotPullback` — wrapper’ы, спасающие существующих агентов с известным edge в узком регионе.
- `DeltaNeutralFundingHarvester`, `CorrelationDislocationAgent`, `DynamicRiskParityPlayer` — новые типы, закрывающие пустые ниши.

**Главный концептуальный шаг:** перестать добавлять «универсальных» агентов и работать в парадигме «specialist agents + smart selection». Все самые прибыльные `Solo_*` в Perfect Oracle — это специалисты, активные только когда условия совпадают. Текущие универсалы (LiveAfterShock, LiveCrashHunter, LiveTrendFollow, LiveMeanRev) активны всегда и поэтому несут -94% .. -223% штрафа за «работу не в свой режим».

Если выполнить рекомендации 4.1, ожидаемый прирост Sharpe текущего портфеля — 0.04-0.10, плюс снятие большей части PnL-утечек от слабых wrappers. Это переводит Panteon Flash из «105% за 5 лет» в диапазон 150-200% при том же уровне риска.
