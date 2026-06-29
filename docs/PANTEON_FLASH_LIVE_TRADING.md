# Panteon Flash live trading documentation

> Historical note 2026-06-29: this file documents the May 2026 Flash live
> state. Current Panteon 3 operations use the unified root launcher
> `Start_panteon.py`; see `docs/PANTEON_3_OPERATOR_GUIDE.md`.

Документ описывает текущую live-сборку `Panteon_Flash`, которая запущена в
торгах из ветки `codex/Panteon_Flash` и должна быть опубликована в ветку
`panteon_flash`.

Проверка live-состояния выполнена 2026-05-28 01:16 MSK по процессам
`Start_MEXC_v2.py` и `Start_BITGET_v2.py`.

## Краткий статус live

На момент проверки оба бота работают в режиме `live_futures`.

| Биржа | Каталог результатов | Статус | Feed | Leader / Selected / Executed | PnL, % | Open | Closed | Ошибки |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| MEXC | `Results/MEXC/2026-05-27_15-44-45_v2` | `running` | `active` | `Panteon_Flash / Panteon_Flash / Panteon_Flash` | `0.11799` | 2 | 0 | нет |
| BITGET | `Results/BITGET/2026-05-27_15-44-25_v2` | `running` | `active` | `Panteon_Flash / Panteon_Flash / Panteon_Flash` | `-0.01086` | 0 | 1 | нет |

Последнее подтвержденное реальное действие после исправления resume/close:

```text
2026-05-27 18:45:47 BITGET close_all OK: WLFI -> order_id=1443557081884540929
bar=5746 raw_signals=1 signals=1 filled=1 rejected=0 blocked=0 flash_actors=... WLFI:LiveTrendFollow ...
```

Это важно: Flash не только выбран как лидер, но реально дошел до биржевого
исполнения close-сигнала.

## Ретротестовый ориентир

Текущий live-профиль риска был сближен с лучшим зафиксированным прогоном:

| Прогон | Окно | Risk fraction | PnL | MaxDD | Closed trades |
| --- | --- | ---: | ---: | ---: | ---: |
| `PanteonFlashRiskFractionSweep_20260526/full2022_2026_risk_12` | 2022-01-01..2026-05-18 | 12% | 112.4549% | 6.24% | 658 |

Источник: `Reports/PanteonFlashFiveYearTimeline_20260525/summary.json` и
`Reports/PanteonFlashFiveYearTimeline_20260525/fixed_vs_current_rolling5y_summary.json`.

Важно: rolling-5y на окне 2021-05-26..2026-05-25 показал другой результат
`15.6333%` при MaxDD `6.27%`. Поэтому `112.5%` используется как ориентир
лучшего fixed-window профиля, а не как обещание live-доходности.

## Runtime-профиль текущих торгов

Фактические runtime-настройки берутся из локального `settings.txt`. Этот файл
игнорируется git и не должен попадать в коммит, потому что является локальным
операционным профилем.

Ключевые параметры текущей live-сборки:

| Параметр | Значение | Назначение |
| --- | ---: | --- |
| `trade_fraction` / `bitget_trade_fraction` | `0.12` | базовая доля капитала на сделку |
| `leverage` | `2` | плечо биржевого исполнения |
| `v2_risk_max_open_positions` | `8` | общий лимит открытых Panteon-позиций |
| `v2_max_daily_loss_pct` | `5` | дневной kill-switch |
| `v2_max_equity_peak_drawdown_pct` | `6.5` | kill-switch по просадке от equity peak |
| `v2_flash_enabled` | `on` | включает путь `Panteon_Flash` |
| `v2_flash_min_score_to_trade` | `4.0` | минимальный score для open-кандидата |
| `v2_flash_shadow_confirmation_enabled` | `on` | включает подтверждение shadow-статистикой |
| `v2_flash_symbol_shadow_confirmation_enabled` | `on` | включает symbol/action shadow-ключи |
| `v2_flash_shadow_signal_handoff_enabled` | `on` | разрешает handoff свежих shadow-сигналов |
| `v2_shadow_fresh_handoff_max_age_bars` | `24` | максимальный возраст shadow-позиции для replay |
| `v2_flash_degradation_guard_enabled` | `on` | включает деградационные deny/cooldown |
| `v2_flash_stale_position_exit_enabled` | `on` | включает закрытие старых live-позиций |
| `v2_flash_stale_position_exit_max_age_bars` | `168` | возраст stale-позиции для close guard |
| `v2_adopt_existing_positions_enabled` | `on` | разрешает подхват биржевых позиций при рестарте |

Операционные guardrails при старте:

```text
capital_fraction=12.00%
max_open_positions=8
daily_loss=5.00%
equity_peak_drawdown=6.50%
slippage=0.75%
api_errors=3
stale_polls=12
pending_timeout=180s
```

## Общая архитектура

`Panteon_Flash` не является одним статическим агентом. Это meta-selector,
который каждый бар выбирает лучшего исполнителя отдельно по каждому символу.
Исполнителем может быть:

- одиночный агент, например `LiveTrendFollow`;
- solo-wrapper игрока, например `Solo_MomentumScalper`;
- ensemble-player, например `Antonius_conservative` или `Optimal_StaticRotator`;
- shadow-replay кандидат, если реальный кандидат молчит, но в shadow есть свежая подтвержденная позиция;
- `NoTrade`, если нет допустимого преимущества или риск-гейты запрещают вход.

Производственный поток:

1. `Start_panteon.py` вызывает `start_production` для выбранной биржи после live pre-flight.
2. `startup.py` загружает биржевой адаптер, капитал, риск, live guardrails,
   Flash-конфиг, список агентов и профили игроков.
3. Создается `ProductionPipeline`: registry агентов, память, карантин,
   selector, composer, strategist, `FlashAllocator`, executor, order ledger,
   event log, dashboard renderer.
4. Загружается snapshot: performance memory, real performance, order ledger,
   tracker позиций, shadow-позиции.
5. Выполняется reconciliation с биржей: существующие позиции либо помечаются
   как внешние, либо усыновляются Panteon Flash, если включен adoption.
6. Запускается v1-bridge. Он дает live market snapshot, funding, fees,
   regime, цены и технический контекст.
7. На каждом баре `main_loop` запускает shadow tournament, обновляет
   память/карантин, затем вызывает `run_flash_decision_path`.
8. Flash выбирает per-symbol actor, генерирует real signals, прогоняет их через
   live guards и передает только разрешенные сигналы в `TradeExecutor`.
9. Executor отправляет ордера на биржу, обновляет `PositionTracker`,
   `PerformanceMemory`, `OrderLedger`, события и статус.

## Агенты

Агент - это минимальный источник торгового действия. В v2 он должен уметь
вызваться как `act(market)` и вернуть действия по символам. Список базовых
агентов регистрируется через `register_all_v1_agents`.

Основные live-агенты:

- `FundingArb`
- `MomentumScalper`
- `LiveAfterShock`
- `LiveCrashHunter`
- `LiveRegimePullback`
- `LiveMeanRev`
- `LiveTrendFollow`
- `LiveVolCompress`
- `RichardDennis`
- `LiveOIBreakout`
- `ResearchValidatorAgent`
- `VolBreakoutHunter`
- `CarryFlowAgentV2`
- `CandlePatternAgent`
- `BullRotationAgent`
- `BearReliefFadeAgent`
- `NeutralRangeScalper`
- `NeutralLiquiditySweep`
- `AnchorFlowMomentum`
- `CrashPanicShortAgent`
- `PlayerFunding`

Опциональные genetics-агенты подключаются только при соответствующих env/config
флагах. Для live они могут быть ограничены probation-логикой или переведены в
shadow-only, чтобы не ломать торговлю тяжелыми или недоказанными компонентами.

### Safety-фильтр агентов

Перед попаданием в реальные кандидаты агент проходит фильтры:

- не `shadow_only`;
- `live_trading_eligible` не равен `False`;
- агент не в карантине;
- агент не нарушает hard-policy;
- genetics-кандидаты могут требовать shadow-confirmation и отдельные лимиты.

Если агент не проходит фильтры, он может продолжать участвовать в shadow, но не
обязан попадать в real execution.

## Игроки

Игрок - это надстройка над агентами. Он превращает несколько агентских голосов
в один или несколько `Signal`.

### EnsemblePlayer

`EnsemblePlayer` строится из `PlayerProfile`:

- выбирает top-k агентов через `AgentSelector`;
- исключает карантинных и неисполняемых агентов;
- нормализует веса;
- применяет voting policy;
- на выходе создает `Signal` с `by_player`, `by_agent`, `vote_weights` и
  `vote_actions`.

Основные production-профили:

- `DefaultEnsemble`
- `NeutralEdgeResearch`
- `TrendResearch`
- `MeanRevResearch`
- `DefensiveResearch`
- `GeneticsResearch`
- `BombermanStrong`

### Regime-switch и rotating players

Кроме профильных ансамблей есть статические и режимные игроки:

- `Antonius_conservative` выбирает разных специалистов по режиму рынка;
- `Perfect_OIBreakout`, `Perfect_CrashSwitch`, `Perfect_NeutralValidator`,
  `Perfect_MeanRev` являются узкими режимными сборками;
- `Optimal_StaticRotator` содержит порядок агентов по режимам и исполняет
  первого текущего actionable агента;
- fixed-agent players держат фиксированный набор агентов с равными весами.

### Solo wrappers

Flash дополнительно создает solo-кандидатов `Solo_<AgentLabel>` для сильных
одиночных агентов. Это важно, потому что классический ensemble может разбавить
точный сигнал специалиста голосами остальных агентов.

Текущая реализация защищает от ложной переоценки solo-wrapper:

- label `Solo_*` сам по себе больше не подавляет raw agent;
- wrapper должен быть настоящим player/shadow-player или иметь торговое
  evidence;
- close-сигналы не требуют actor evidence, чтобы можно было закрыть уже
  открытую позицию.

### NoTrade

`NoTrade` - полноценный candidate-row, а не отсутствие решения. Flash выбирает
его, если:

- лучший open не прошел score/min-trades/pnl/shadow gates;
- сигнал находится в deny или terminal deny;
- деградационный guard временно отключил actor/symbol/signal;
- риск исполнения запрещает новый вход;
- лучший кандидат уступает уже выбранному anchor/previous actor по правилам
  switch margin.

## Память

`PerformanceMemory` является единым источником статистики по label x regime.
Один label может быть агентом или игроком.

Память хранит:

- количество сигналов, entries и закрытых сделок;
- wins/losses;
- net/gross PnL;
- fees и funding;
- returns для Sharpe;
- equity curve и max drawdown;
- blocked/rejected/pending execution counters;
- открытые позиции по `(position_scope, label, symbol)` для корректного close
  accounting;
- snapshot/restore для продолжения сессии.

При open память запоминает позицию. При close она относит результат к opener
labels. Для защитных закрытий (`StalePositionGuard`, `PartialProfitLock`,
`GeneticsProbationRegimeExit`, `CashFlat`/`NoTrade`) результат закрытия
атрибутируется исходному opener, а не техническому guard-агенту.

В production используются как минимум две важные памяти:

- `perf`: общая память для selector/strategist/shadow learning;
- `real_perf`: память реальных исполнений для live-отчетности и PnL.

Shadow tournament пишет виртуальные результаты в общую performance memory, но
не загрязняет реальные биржевые позиции и order ledger.

## Карантин и деградация

Карантин работает на уровне агентов/игроков, а Flash degradation - на уровне
более точных ключей:

```text
actor_type:label|symbol|action
actor_type:label|regime:<regime>
symbol
```

Текущий live-профиль:

- `v2_flash_degradation_guard_enabled=on`;
- `degradation_window_closed_trades=1`;
- `degradation_min_closed_trades=1`;
- `degradation_max_recent_pnl_usd=-1.0`;
- `degradation_actor_scope=actor_regime`;
- signal cooldown `720` баров;
- actor cooldown `72` бара.

Смысл: если конкретный signal key, actor-regime или symbol быстро показывает
отрицательный realized outcome, Flash временно запрещает новые opens по этому
ключу. Close-сигналы при этом не блокируются, чтобы не оставлять риск в рынке.

## Shadow tournament

Shadow tournament запускается каждый market bar до real selection.

Он:

1. клонирует/обновляет доступных агентов;
2. гоняет всех агентов и игроков на виртуальном `FakeExchange`;
3. использует отдельные `PositionTracker` на actor runtime;
4. записывает виртуальные fills/rejections/blocks в `PerformanceMemory`;
5. отдает в main loop:
   - последние shadow-сигналы игроков и агентов;
   - последние shadow-позиции;
   - `ShadowActorUpdated`;
   - summary counters.

Ключевая защита от look-ahead:

- real selection использует candidate snapshot, собранный до применения
  текущих shadow updates;
- текущий бар shadow не должен улучшать score кандидата в этом же баре.

Shadow используется тремя способами:

- как статистическое подтверждение кандидата (`shadow_confirmation`);
- как источник свежего handoff/replay, если shadow уже открыл позицию;
- как источник деградационных сигналов и диагностики.

## Как Flash выбирает стратегию

Flash работает per symbol. На каждом баре он получает:

- `market`;
- список real-executable agents;
- список player candidates;
- actionable labels из shadow;
- shadow confirmation score map;
- degraded signal/actor/symbol sets;
- promoted signal keys;
- previous actor by symbol;
- текущие стороны открытых позиций.

Дальше `FlashAllocator.decide()` делает следующее.

### 1. Строит candidate rows

Для каждого символа создаются строки от:

- raw agents;
- ensemble players;
- solo wrappers;
- shadow player signals;
- shadow agent signals;
- synthetic `NoTrade`.

Каждая строка содержит actor type, label, action, signal, base metrics,
shadow metrics, score, reasons, deny flags и audit fields.

### 2. Считает base score

Базовый score идет из performance metrics:

- PnL;
- closed trades;
- Sharpe;
- drawdown;
- signal quality;
- regime match;
- portfolio actor scope, если actor включен в `portfolio_actor_keys`.

Для кандидатов без истории используется `no_data_score`. Для actionable
кандидатов может добавляться `actionable_bonus`.

### 3. Применяет shadow confirmation

При включенном `v2_flash_shadow_confirmation_enabled` open-кандидат должен
иметь достаточную shadow-статистику.

Поддерживаются ключи:

```text
label
(label, symbol, action)
("__symbol_health__", symbol, action)
```

Текущий live-профиль включает symbol/action confirmation и actor fallback.
Это дает Flash возможность отличать, например, сильного агента вообще от
сильного действия именно по `BTC/USDT FUT_SHORT_FULL`.

### 4. Применяет hard gates

Open-кандидат может быть отклонен, если:

- score не выше `min_score_to_trade`;
- закрытых сделок меньше `min_closed_trades_to_trade`;
- PnL ниже `min_pnl_pct_to_trade`;
- shadow score/closed/win-rate/LCB не проходит пороги;
- signal key находится в `denied_signal_keys`;
- signal key находится в `terminal_denied_signal_keys`;
- symbol или regime запрещен для new open;
- actor/signal/symbol находится в degradation cooldown;
- actor превышает `max_signals_per_actor`;
- overextension guard запрещает late entry;
- технический hard gate запрещает entry.

Close-сигналы проходят более мягкий путь. Они не должны зависеть от open-only
evidence, потому что задача close - снять риск, а не доказать новый edge.

### 5. Учитывает технический слой

Технический слой поддерживает RSI(14), MACD(12,26,9) и ATR(14).

Режим работы:

- если `technical_overlay_enabled=off`, слой не влияет на live-решение;
- если overlay включен, RSI/MACD/ATR пишутся в audit rows;
- long alignment требует RSI в заданном диапазоне и положительный MACD
  histogram;
- short alignment требует RSI в заданном диапазоне и отрицательный MACD
  histogram;
- alignment может добавить `technical_score_bonus`;
- misalignment может добавить `technical_score_penalty`;
- `technical_hard_gate_enabled` превращает misalignment в hard NoTrade;
- ATR risk sizing масштабирует `risk_mult` к целевой волатильности.

В текущей live-сборке технический слой внедрен в код и покрыт тестами, но
hard-gate должен включаться только после ретротестового превосходства над
baseline. Это сделано намеренно: индикаторы дают контекст, но не должны
автоматически ухудшать уже лучший Flash-профиль.

### 6. Выбирает победителя

Сортировка отдает приоритет:

- неотклоненным кандидатам;
- close-сигналам перед open/hold, когда нужно снять риск;
- большему score;
- anchor/previous actor, если challenger не дает достаточного преимущества;
- promoted/selected subset overrides, если они заданы в профиле.

Если победитель не проходит финальные условия, итогом становится `NoTrade`, но
в audit остается `original_selected_actor`, чтобы можно было понять, кого Flash
хотел выбрать и почему заблокировал.

### 7. Генерирует real signal

Если победитель не `NoTrade`, Flash создает `Signal`:

- `by_player="Panteon_Flash"` или label выбранного player;
- `by_agent` равен выбранному агенту/главному contributor;
- `risk_mult` может быть изменен actor risk sizing, funding sizing,
  volatility sizing, selected subset overrides, ATR sizing или degradation
  sizing;
- `position_scope` используется для учета open/close.

## Как Panteon торгуется

После Flash-выбора real signals идут через `filter_real_signals_against_tracker`.

Этот guard:

- удаляет close без позиции;
- удаляет duplicate open по тому же символу;
- превращает противоположный open в close, если уже есть позиция другой стороны;
- применяет `max_new_opens_per_bar`;
- применяет `max_open_positions`;
- не дает трогать внешние позиции, если они не усыновлены Panteon Flash.

Затем `TradeExecutor.execute()` делает единственный допустимый путь
`Signal -> Exchange -> Memory + Events`.

Open:

1. проверяется symbol health;
2. проверяется exchange health guard;
3. читается min notional;
4. `RiskLimits` считает notional:

```text
notional = balance_usd * capital_fraction * action_fraction * risk_mult
```

5. qty квантуется под биржу;
6. order idempotency key записывается в `OrderLedger`;
7. ордер отправляется через реальный exchange adapter;
8. при fill позиция появляется в `PositionTracker`, memory и event log.

Close:

1. проверяется наличие Panteon-owned позиции;
2. qty берется из tracker;
3. external positions не закрываются;
4. order ledger защищает от дубликатов;
5. при fill позиция закрывается, realized PnL обновляет opener memory.

## Возобновление сессии и позиции с предыдущих торгов

Проблема "Panteon не использует позиции с предыдущих торгов" закрывается
следующими механизмами.

### Snapshot restore

На старте `_load_or_migrate_state` восстанавливает:

- `PerformanceMemory`;
- `real_perf`;
- `OrderLedger`;
- `PositionTracker`;
- shadow positions.

### Exchange reconciliation

После восстановления pipeline читает реальные позиции с биржи. Если позиция
есть на бирже, но отсутствует или была внешней в tracker, включается adoption:

```text
by_player = PanteonFlashAdopted
by_agent  = AdoptedExchangePosition
```

Для такой позиции seed-ится open в `PerformanceMemory`, чтобы последующий close
получил корректный realized accounting.

Текущий live MEXC как раз держит две усыновленные позиции:

| Symbol | Side | Entry | Owner |
| --- | --- | ---: | --- |
| UNI | short | 3.368 | `PanteonFlashAdopted / AdoptedExchangePosition` |
| LINK | long | 9.724 | `PanteonFlashAdopted / AdoptedExchangePosition` |

BITGET после рестарта усыновил позицию WLFI и затем закрыл ее реальным
`close_all OK`, что подтверждает работоспособность resume/close path.

### Sync агентов с real state

Перед реальным голосованием `sync_player_agents_to_real_positions` очищает у
агентов stale symbols и inject-ит текущие реальные позиции. Это нужно, чтобы
v1-агенты не думали, что они flat, когда биржа уже держит позицию, и наоборот.

## Защитные close-механизмы

Помимо обычных сигналов от агентов Flash добавляет технические close-сигналы.

### StalePositionGuard

Если позиция слишком старая, Flash может закрыть ее через
`StalePositionGuard`.

Текущий профиль:

```text
enabled = on
max_age_bars = 168
require_nonpositive_unrealized = on
```

То есть старая позиция закрывается только если она не имеет положительного
unrealized PnL.

### PartialProfitLock

Механизм частичной фиксации прибыли поддержан в коде. Он может закрывать долю
позиции при достижении trigger PnL и после минимального возраста позиции.
Включение зависит от runtime-настроек.

### Genetics probation exits

Если genetics-кандидаты торгуются в probation-режиме, отдельный overlay может
закрывать их позиции при выходе из допустимого режима.

## Наблюдаемость

Для диагностики используются:

- `status.json` - live состояние, leader/selected/executed, PnL, позиции,
  counters, kill switch, step error;
- `trading.log` - краткая строка каждого бара и реальные filled/rejected/blocked;
- `leaderboard_agents.json` - рейтинг агентов;
- `leaderboard_players.json` - рейтинг игроков;
- `causal_entry_decisions.jsonl` - полные causal decisions;
- `dashboard_latest.png` - live dashboard;
- `shadow_dashboard.png` - shadow tournament view;
- `memory_dashboard.png` - состояние памяти;
- `regime_dashboard.png` - режимы.

В live-проверке надо всегда отличать:

- `shadow_signals` и `shadow_filled` - виртуальные действия;
- `raw_signals`, `signals`, `filled` - реальные сигналы текущего бара;
- `open_positions` с `PanteonFlashAdopted` - позиции, подхваченные после
  рестарта, а не заново открытые Flash на этом запуске.

## Что было исправлено перед текущим push

1. `Panteon_Flash` действительно выбран и исполняется как `selected_leader` и
   `executed_leader`, а не остается shadow-only.
2. Close-сигналы больше не требуют open-only actor evidence, min-score или
   shadow-confirmation. Это нужно для безопасного выхода из уже открытых
   позиций.
3. `NoTrade` больше не подавляет close, когда есть допустимый close-кандидат.
4. `Solo_*` actionable label сам по себе больше не подавляет raw-agent сигнал
   без торгового evidence.
5. Resume path восстанавливает snapshot, сверяется с биржей, усыновляет
   разрешенные позиции и синхронизирует состояние v1-агентов перед торговлей.
6. Live-риск сближен с лучшим fixed-window прогоном: базовый fraction поднят
   до `12%`, equity peak drawdown guard установлен на `6.5%`.

## Практический алгоритм одного бара

```text
1. Получить MarketSnapshot из v1-bridge.
2. Обновить live balance, positions, funding, fees, regime, technical context.
3. Составить candidates из production profiles, regime-switch players,
   fixed/rotating players и solo wrappers.
4. Запустить shadow tournament по тем же market data.
5. Сохранить pending shadow updates, но не допустить look-ahead в real score.
6. Обновить quarantine/degradation state.
7. Вызвать FlashAllocator.decide() per symbol.
8. Получить decisions: actor или NoTrade для каждого символа.
9. Добавить stale-position, regime-exit и partial-profit close signals.
10. Отфильтровать real signals против PositionTracker и risk caps.
11. Отправить разрешенные signals в TradeExecutor.
12. Записать fills/rejections/blocks в OrderLedger, PositionTracker,
    PerformanceMemory, EventLog и status/dashboard.
13. Сохранить snapshot для следующего рестарта.
```

## Операционные правила

- Не считать shadow fills реальными сделками.
- Не удалять `settings.txt` и не коммитить его принудительно: это runtime
  профиль и локальные ключи.
- После изменения Flash gates сначала нужен fixed-window и rolling-window
  ретротест, затем live smoke.
- Для сравнения "лучше ли Panteon" использовать одновременно:
  `status.json`, `leaderboard_players.json`, `leaderboard_agents.json`,
  `trading.log` и causal decisions.
- Если `selected_leader != executed_leader`, `step_error` не пустой или
  `kill_switch_reason` не пустой, live-статус нельзя считать здоровым.
- Позиции `RecoveredExchangePosition` считаются внешними и не закрываются
  Panteon. Позиции `PanteonFlashAdopted` считаются принятыми под управление
  Flash и могут закрываться его close-логикой.
