# Exia: минимальный план режимной торговой системы

Статус: исследовательский план. Paper/live не разрешены.

## Реализованный статус на 2026-08-04

- `P0` выполнен: добавлены строгие `CandidateSpec` и `ExperimentManifest` с
  fail-closed safety flags и SHA-проверкой strategy/taxonomy/dataset.
- `P1` выполнен: четыре каузальных состояния реализованы непосредственно в
  Freqtrade strategy path; prefix, warm-up, gap и volatility-shock tests проходят.
- Default Freqtrade/Docker entrypoint использует `ExiaMarketModeStrategyV2`,
  `max_open_trades=1`, leverage 1 и hard-coded zero entries.
- Full8 distribution report имеет статус
  `FOUNDATION_READY_FOR_CANDIDATE_DESIGN`; все четыре states представлены во всех
  исторических окнах.
- `P2` выполнен: один Freqtrade experiment runner сохраняет immutable manifest,
  normalized trade ledger, block-bootstrap metrics и пересобираемый offline
  catalog по state/direction/symbol.
- Foundation experiment прошёл 8/8 запусков с ожидаемыми нулевыми входами и
  статусом `ENGINEERING_PASS_NO_TRADES`.
- `P3-v1` завершён отрицательно: три заранее зарегистрированных full8 trend-кандидата
  проверены только на development с base/stress costs. Ни один не получил
  положительный block-bootstrap LCB, поэтому validation не открыта.
- Freqtrade recursive-analysis выявил restart-зависимость taxonomy v1: 4320-часовой
  volatility lookback не помещался в доступный Bitget startup history, а 744 бара
  создавали искусственный переход из `UNSAFE` после перезапуска.
- Реализована versioned taxonomy `exia.market_mode.v2`: фиксированное 720-часовое
  окно, 999 startup bars и ограниченные diagnostic counters. Full8 restart audit
  прошёл на 8/8 символах, foundation experiment прошёл 8/8 no-trade запусков.
- `P3-v2` завершён отрицательно: единственный заранее зарегистрированный
  ADA-only Donchian48 candidate на taxonomy v2 дал 180 development trades, stress
  mean `-9.19 bps` и block LCB `-80.01 bps`. Validation/OOS/lookahead не
  запускались, поскольку первичный alpha gate уже провален.

Этот статус подтверждает только корректность foundation. Он не является
положительным backtest, paper gate или разрешением на реальные ордера.

## 1. Результат проверки сложности

Первоначальный Exia-план был перегружен. Четыре оси давали 72 комбинации, поверх
них вводились восемь архетипов, специализированные агенты, отдельный
SimpleResearch-симулятор и последующий перенос во Freqtrade. Формально эти слои
были диагностическими, но на практике они повторили бы основные проблемы старого
Pantheon.

| Риск исходного плана | Аналогичная проблема Pantheon | Решение Exia Minimal |
|---|---|---|
| 72 ячейки и восемь архетипов | разреженная matrix и случайно хорошие малые срезы | четыре конечных состояния без дополнительной проекции |
| отдельные axis/archetype/symbol/direction scores | длинная цепочка gates и противоречивые статусы | пять основных alpha-метрик и ограниченные диагностические таблицы |
| SimpleResearch плюс Freqtrade | replay/canary/live имели разную семантику | Freqtrade является единственным backtest/dry/live engine |
| специализированные агенты и будущий allocator | Flash/players снова создавали непрозрачный выбор | одна активная стратегия, без actor routing и ensemble |
| taxonomy с большим числом параметров | настройки разных слоёв меняли фактическое поведение | один кодовый профиль без live overrides |
| глобальный BTC state, breadth и local state | несколько конкурирующих источников режима | только локальный state символа в v1 |

Вывод: четыре независимые оси полезны как исследовательская идея, но не как
первая производственная архитектура. Они исключены из Exia Minimal. Возврат
любой дополнительной оси допускается только после отдельного ablation report и
не ранее успешного micro-live одной простой стратегии.

## 2. Целевая архитектура

```text
immutable Bitget OHLCV
          |
          v
one causal market-mode function
          |
          v
one Freqtrade strategy implementation
          |
          +--> Freqtrade backtest (base/stress)
          +--> Freqtrade dry-run
          +--> manual micro-live
          |
          v
Freqtrade trade export -> one compact audit report
```

В promotion path нет Pantheon, Flash, Genetics, player, actor, policy allocator,
candidate matrix, второго симулятора или автоматического выбора стратегии.

`SimpleResearch` остаётся архивным инструментом для дешёвых исследований и
сравнения старых результатов. Его отчёт не может дать promotion. Авторитетным
результатом является только прогон той же Freqtrade strategy, которая затем
используется в dry-run и micro-live.

При этом research path может содержать много кандидатов. Ограничение "одна
стратегия" относится к одному experiment run и к production, а не к общему числу
исследуемых идей. Все кандидаты тестируются последовательно одним runner и одним
движком, поэтому удобство сравнения не требует live-ансамбля.

## 3. Четыре состояния рынка

| State | Определение | Разрешённое исследование |
|---|---|---|
| `TREND_UP` | close > EMA24 > EMA96 и 24h momentum > 0 | long trend/pullback |
| `TREND_DOWN` | close < EMA24 < EMA96 и 24h momentum < 0 | short trend/pullback |
| `RANGE` | валидные данные, но нет согласованного тренда | low-turnover range либо NoTrade |
| `UNSAFE` | неполный warm-up, data gap или volatility shock | только NoTrade/close-only |

Это четыре конечных runtime-состояния, а не четыре оси. Направление уже включено
в trend state. Переход не является отдельным режимом: он наблюдается через
`previous_state` и `state_age_bars`. Compression и volatility percentile
остаются непрерывными диагностическими признаками стратегии, но не создают новые
категории или ветви маршрутизации.

### 3.1. Минимальный контракт классификатора

Классификатор работает отдельно для каждого символа и использует только закрытые
1h свечи не позднее текущей свечи `t`.

Порядок классификации:

1. Невалидные данные или незавершённый warm-up дают `UNSAFE`.
2. Realized volatility выше trailing 95-го процентиля даёт `UNSAFE`.
3. Согласованный EMA24/EMA96 и momentum дают `TREND_UP` или `TREND_DOWN`.
4. Всё остальное даёт `RANGE`.

Сигнал строится после закрытия `t`, а исполняется не раньше открытия `t+1`.
Trailing volatility distribution сдвигается на один бар. Добавление будущих
данных не должно менять метки уже существующего префикса.

Профиль первой версии содержит только закреплённые в коде периоды EMA 24/96,
momentum 24h, realized volatility 24h, trailing окно 180 дней и unsafe percentile
95%. Они не доступны как Freqtrade hyperopt, environment variables или live
настройки. Их изменение создаёт новую taxonomy version и новый candidate ID.

### 3.2. Что сознательно исключено

- отдельные volatility, structure и phase categories;
- 72-cell key и восемь derived archetypes;
- BTC global regime и full8 breadth;
- per-symbol thresholds;
- hysteresis и динамическая перенастройка;
- model-dependent regime definitions;
- fallback из одного state/agent в другой.

Если простой state мигает, это сначала измеряется через `state_age_bars`. Нельзя
сразу добавлять hysteresis: он создаёт ещё один state machine и усложняет parity.

## 4. Одна стратегия, один конфиг

Одновременно разрабатывается и запускается только один candidate. Strategy
не выбирает других агентов и не меняет параметры во время выполнения.

Candidate manifest содержит только:

- `candidate_id` и strategy source SHA;
- taxonomy version/source SHA;
- symbols и timeframe;
- неизменяемые strategy parameters;
- base/stress cost assumptions;
- stake, max open trades, leverage, daily loss и expiry.

В manifest нет actor weights, regime matrix, promotion bypass, Flash, Genetics,
fallback или thresholds классификатора.

Число независимых торговых настроек ограничивается тремя группами:

1. strategy parameters, закреплённые до validation;
2. exchange-independent risk limits;
3. exchange execution settings Freqtrade.

Каждая группа имеет одного владельца. Один параметр не дублируется в registry,
strategy, dry-run config и live manifest.

## 5. Данные

Authoritative dataset — единый full8 Bitget futures OHLCV 1h snapshot. Публичная
история догружается по диапазону и проверяется на timestamps, continuity,
duplicates и OHLCV validity. Непрерывный локальный collector для alpha-проверки
не нужен.

Обязательные artifacts:

1. raw Freqtrade OHLCV files и integrity manifest;
2. candidate manifest;
3. Freqtrade backtest exports для каждого окна/cost scenario;
4. единый audit JSON/Markdown;
5. dry-run database/log и итоговый operational report только после alpha PASS.

Отдельный regime tape не является runtime-зависимостью. Для анализа состояния
экспортируются вместе со сделками: `signal_timestamp`, state, `state_age_bars`,
trend score, volatility percentile и compression ratio. State связывается с
моментом сигнала, а не восстанавливается постфактум через время входа.

## 6. Исторические окна и holdout

| Окно | Период | Роль |
|---|---|---|
| development | 2022-01-01 .. 2023-12-31 | создание экономической гипотезы |
| validation | 2024-01-04 .. 2024-12-31 | первая зафиксированная проверка |
| historical OOS | 2025-01-04 .. 2025-12-31 | walk-forward проверка |
| historical sanity | 2026-01-04 .. 2026-07-15 | недавняя устойчивость |

Эти данные уже многократно просматривались, поэтому весь диапазон теперь
считается retrospective research, а не финальным независимым доказательством.

Для каждого нового candidate до просмотра новых данных фиксируются strategy SHA,
taxonomy SHA, параметры, symbols, costs, baseline и минимальное число сделок.
После этого открывается один новый sealed holdout. Провал означает terminal reject
этого ID; thresholds на раскрытом holdout не меняются.

## 7. Один backtest path

Freqtrade выполняет development, validation, OOS, sanity, base costs и stress
costs. Отдельная реализация target positions в SimpleResearch не поддерживается
как evidence path.

Одна команда должна:

1. проверить dataset и manifest SHA;
2. проверить отсутствие live credentials и `dry_run=true` для runtime smoke;
3. выполнить все закреплённые timeranges с одной strategy class;
4. выполнить Freqtrade lookahead/recursive analysis;
5. прочитать штатные trade exports;
6. построить один итоговый отчёт;
7. всегда оставить `live_allowed=false` до отдельного ручного шага.

Не существует отдельных matrix, policy compiler и canary strategy config.

## 8. Минимальный скоринг

### 8.1. Пять основных alpha-метрик

| Метрика | Назначение |
|---|---|
| `closed_trades` | достаточность наблюдений |
| `stress_mean_net_bps` | expectancy после fee/slippage stress |
| `block_lcb_net_bps` | консервативная оценка с временной зависимостью |
| `max_drawdown` | риск полного портфеля |
| `baseline_differential_lcb` | доказательство превосходства простой альтернативы |

Остальные значения — gross PnL, fees, turnover, win rate, profit factor,
concentration — показываются только как диагностика. Они не образуют отдельные
promotion stairs и не складываются в непрозрачный общий score.

### 8.2. Допустимые таблицы

Отчёт имеет только пять уровней:

1. overall;
2. windows;
3. market state;
4. direction;
5. symbol.

Пересечения `symbol x state x direction x window` не строятся автоматически.
Такой срез допустим только если он заранее является самой гипотезой и имеет
достаточное число сделок.

### 8.3. Alpha gate

Candidate проходит только если одновременно:

- не менее 50 retrospective holdout trades;
- не менее 10 trades как минимум в двух независимых historical holdout windows;
- stress mean > 0;
- moving-block-bootstrap LCB > 0;
- baseline differential LCB > 0;
- max drawdown <= 0.20;
- нет отрицательного eligible window с минимум 10 trades;
- новый sealed holdout проходит те же expectancy/LCB требования.

Положительный mean не компенсирует отрицательный LCB. Высокий результат одного
symbol не компенсирует collapse всей стратегии. Gate нельзя снижать из-за низкой
частоты активации.

## 9. Ограниченный набор гипотез

На первом цикле разрешены не более трёх новых economic families:

1. low-turnover trend-following для `TREND_UP/TREND_DOWN`;
2. compression-to-trend breakout, где state transition является условием самого
   сигнала, а не отдельным routing layer;
3. cost-aware range mean reversion для `RANGE`.

На family допускается максимум три заранее зарегистрированных варианта. Старые
Pantheon players не переносятся массово. Используются только стратегии с ясной
экономической механикой, которые можно выразить одной Freqtrade strategy class.

Первой проверяется одна family. Следующая начинается после terminal verdict
предыдущей, а не параллельно через общий турнир.

Ансамбль, allocator и Genetics не входят в план первой live-версии. Даже при двух
успешных стратегиях сначала запускается лучшая одиночная стратегия. Портфельная
комбинация является отдельным будущим проектом и требует собственного OOS.

## 10. Research bench и накопление результатов

### 10.1. Единый интерфейс кандидата

Разные агенты допустимы как исследовательские signal models. Каждый кандидат
обязан иметь один immutable `CandidateSpec`:

- `candidate_id`, family и semantic version;
- strategy module/class и source SHA;
- economic claim;
- фиксированные parameters;
- symbols, timeframe и заявленные market states;
- baseline и trial-family;
- status: `DEVELOPMENT`, `LOCKED`, `REJECTED`, `PROMOTABLE`.

Research runner принимает ровно один `candidate_id`, запускает одну Freqtrade
strategy и завершает experiment. Он не выбирает победителя и не меняет параметры.
Следующий кандидат запускается отдельной командой. Для dry/live разрешён только
статически закреплённый `LOCKED` candidate SHA.

### 10.2. Immutable experiment artifact

Каждый прогон сохраняется в отдельный каталог по `experiment_id`:

```text
Reports/Exia/experiments/<experiment_id>/
  manifest.json
  trades.parquet
  metrics.json
  report.md
```

`manifest.json` закрепляет candidate, strategy, taxonomy, dataset, Freqtrade
revision, timeranges, costs и команду запуска. Повторный запуск не перезаписывает
старый experiment.

Trade ledger хранит состояние рынка на момент сигнала. Минимальные поля:

- experiment/candidate/family/version identifiers;
- signal, entry и exit timestamps;
- symbol, direction и market state;
- gross PnL, fees, slippage и net PnL;
- exit reason, split и cost scenario;
- strategy, dataset и taxonomy SHA.

### 10.3. Восстанавливаемый каталог

Отдельная команда пересобирает производный каталог из immutable experiment
folders. Каталог можно удалить и полностью восстановить, поэтому он не становится
ещё одним источником истины.

Нормализованная строка каталога имеет ключ:

```text
(candidate_id, candidate_version, dataset_sha, taxonomy_sha,
 split, cost_scenario, scope, scope_value)
```

`scope` ограничен значениями `overall`, `state`, `direction`, `symbol`, `window`.
Автоматический cross-product запрещён. Для строки сохраняются пять основных
alpha-метрик, sample count, status и ссылка на исходный experiment.

### 10.4. Удобные представления

Из одного каталога строятся без повторного backtest:

- таблица `candidate x market state` с expectancy, LCB и числом сделок;
- таблица `candidate x window` для проверки временного collapse;
- таблица `candidate x symbol` для concentration;
- список лучших подтверждённых кандидатов для выбранного state;
- история версий одного candidate family;
- coverage map: где данных недостаточно и требуется новый эксперимент.

Ячейка с недостаточным sample count помечается `INSUFFICIENT`, а не участвует в
ранжировании. Сравнивать можно только результаты с одинаковыми dataset SHA,
taxonomy SHA, timeranges и cost contract. Каталог не выдаёт live-разрешение:
promotion verdict остаётся свойством конкретного immutable experiment.

### 10.5. Защита от превращения каталога в Pantheon

- каталог анализируется offline и не загружается live runtime;
- нет весов агентов и автоматического winner selection;
- новый backtest не изменяет старый verdict;
- просмотренный OOS увеличивает trial count и блокирует retuning того же ID;
- market-state таблица является диагностикой, пока state specialization не
  зарегистрирована как новая гипотеза;
- production manifest ссылается на один candidate и один experiment SHA.

## 11. Тестирование

### 11.1. Классификатор

- prefix invariance после добавления будущих свечей;
- next-open timing;
- shifted trailing quantile;
- warm-up/data-gap -> `UNSAFE`;
- детерминированные границы четырёх states;
- одинаковые результаты при повторном запуске;
- отсутствие per-symbol/live overrides.

### 11.2. Strategy и backtest

- unit tests entry/exit на небольших фиксированных candle fixtures;
- Freqtrade lookahead analysis;
- Freqtrade recursive analysis;
- один golden backtest fixture с закреплённым trade count и timestamps;
- base/stress costs ненулевые;
- одинаковый strategy SHA во всех окнах;
- fail-closed при изменении dataset/taxonomy/candidate SHA.

### 11.3. Research catalog

- повторный experiment не перезаписывает предыдущий;
- каталог полностью восстанавливается из experiment folders;
- несовместимые dataset/taxonomy/cost результаты не сравниваются;
- insufficient slices не попадают в ranking;
- terminal-rejected candidate нельзя вернуть в `LOCKED` без нового ID;
- catalog artifacts никогда не дают `live_allowed=true`.

### 11.4. Dry-run и micro-live

Dry-run проверяет только операционную корректность:

- свежесть feed;
- ожидаемые signal/order/fill transitions;
- exchange filters и min notional;
- exits, reconciliation, restart и daily-loss latch;
- отсутствие реальных ордеров.

Dry-run не обязан заново статистически доказывать alpha. Short dry-run должен
дать хотя бы один полный ожидаемый trade lifecycle. Extended dry-run проверяет
устойчивость процесса и завершается по заранее заданному сроку, а не продолжается
до случайной прибыли.

## 12. Этапы реализации

### P0. Упрощённый контракт

- этот документ;
- schema минимального candidate manifest;
- schema `CandidateSpec` и immutable experiment artifact;
- четыре states и запрет runtime overrides;
- правила нового sealed holdout.

### P1. Единственный market-mode module

- реализовать чистую causal-функцию;
- добавить unit/prefix tests;
- встроить её непосредственно в Freqtrade strategy path;
- построить простой state-distribution report.

Stop: если четыре states требуют per-symbol threshold tuning.

### P2. Один authoritative experiment runner

- запуск Freqtrade по всем окнам и двум costs;
- штатные exports -> один audit report;
- block bootstrap, baseline differential и пять alpha-метрик;
- immutable experiment folder и восстанавливаемый research catalog;
- отчёты `candidate x state/window/symbol` без повторного backtest;
- lookahead/recursive checks.

Stop: если отчёт нельзя воспроизвести одной командой.

Реализовано 2026-08-04 командой `tools/run_exia_experiment_v1.py`. Первый
immutable experiment находится в `Reports/Exia/experiments`, а производный
каталог пересобирается `tools/build_exia_catalog_v1.py`.

### P3. Первый specialist candidate

- выбрать одну family;
- зарегистрировать максимум три варианта;
- использовать development только для выбора;
- открыть validation/OOS по протоколу;
- terminal reject при провале.

Результат `P3-v1` от 2026-08-04: `NO_CANDIDATE_FOR_VALIDATION`. На stress costs
state-onset получил mean `-3.15 bps`, LCB `-40.31 bps`; Donchian48 — mean
`-26.72 bps`, LCB `-75.87 bps`; pullback-reclaim — mean `+20.87 bps`, но LCB
`-21.22 bps`. У всех трёх drawdown ниже лимита и нет lookahead bias, однако
статистический gate не пройден. Положительные ADA-срезы являются exploratory
результатом множественного сравнения и сами по себе validation не разрешают.

### P4. Sealed holdout

- закрепить полный candidate SHA;
- догрузить новые публичные данные;
- открыть holdout один раз;
- разрешить dry-run только при полном alpha PASS.

### P5. Strict dry-run

- короткая проверка полного lifecycle;
- ограниченная extended operational session;
- без автоматического изменения стратегии или thresholds.

### P6. Ручной micro-live

- одна strategy;
- leverage 1;
- одна позиция;
- малый max notional;
- жёсткий daily-loss latch и expiry;
- ручное подтверждение точного manifest SHA.

## 13. Защита от возврата к Pantheon

Изменение отклоняется, если оно добавляет:

- второй backtest/execution engine в promotion path;
- actor/agent/player selection в runtime;
- автоматический fallback на другую стратегию;
- новый regime axis без ablation evidence;
- cross-product matrix как promotion gate;
- повторную настройку terminal-rejected ID;
- canary без заранее заданного срока;
- новый bypass или возможность изменить taxonomy из environment/config;
- загрузку research catalog в dry/live runtime;
- автоматический выбор кандидата по historical ranking;
- автоматическое включение live после успешного отчёта.

Главный принцип Exia Minimal: сложность может добавляться только после доказанной
необходимости. Первая цель — не покрыть все состояния рынка, а получить одну
понятную, воспроизводимую и ограниченную стратегию с положительным независимым
costed edge.

## 14. Ближайший шаг

ADA-only Donchian48 получил terminal reject. Не менять его EMA/Donchian periods,
stoploss, direction или symbol filter: это было бы повторной настройкой той же
проваленной гипотезы на просмотренном development.

Следующая разрешённая family должна проверять ровно одну ортогональную механику
на taxonomy v2. Приоритет — один fixed range mean-reversion candidate только в
`RANGE`, без ensemble и per-symbol tuning. Он сначала проходит тот же
development stress gate; lookahead запускается только после alpha PASS,
validation — только после обоих PASS. Paper/live по-прежнему запрещены.
