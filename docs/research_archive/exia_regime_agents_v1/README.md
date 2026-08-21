# Exia regime agent archive v1

Статус: исследовательский архив. Ни один профиль не прошёл полный promotion gate.

Архив сохраняет наиболее информативные 4h-профили по режимам рынка, их точные параметры, реализацию и costed evidence. Это не whitelist и не рекомендация для paper/live.

## Краткий каталог

| Regime | Rank | Agent | Role | Hist trades | Mean | Median | LCB | Family LCB | Holdout trades | Holdout mean | Holdout median | Holdout LCB |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bullish | 1 | `MomentumScalper__SLOW200` | primary_research_lead | 468 | 77.12 | -284.58 | 27.15 | -12.35 | 1 | -327.75 | -327.75 | n/a |
| bullish | 2 | `NEW_CrossSectionalResidualMomentum` | mechanism_reference | 2941 | 10.32 | -33.66 | -2.79 | -14.56 | 27 | 13.43 | -5.69 | -50.24 |
| bullish | 3 | `LiveOIBreakoutVolumeFallback` | breakout_reference | 1614 | 43.08 | -52.47 | 11.37 | -22.60 | 15 | -182.27 | -150.30 | -329.78 |
| bearish | 1 | `LiveCrashHunter__SLOW150` | primary_research_lead | 430 | 52.84 | 23.67 | 10.11 | -27.91 | 5 | -223.78 | -208.15 | -339.07 |
| bearish | 2 | `SR_CandleMomentum3__SLOW150` | high_incidence_reference | 4556 | 4.55 | -47.74 | -12.26 | -27.86 | 47 | -24.09 | -43.28 | -54.73 |
| bearish | 3 | `NEW_EfficiencyRatioTrend` | mechanism_reference | 2082 | 9.61 | -60.18 | -13.53 | -36.11 | 33 | -20.16 | -63.69 | -76.52 |
| neutral | 1 | `MeanRevConfirmedAgent__FAST067` | primary_research_lead | 570 | 16.63 | 54.47 | -17.98 | -53.15 | 25 | -4.81 | 46.19 | -86.93 |
| neutral | 2 | `NEW_VolumePriceDivergence` | distribution_reference | 1812 | -10.83 | 60.22 | -31.58 | -49.61 | 38 | 3.37 | 49.82 | -89.31 |
| neutral | 3 | `NEW_VolatilityRegimeSwitch` | hybrid_policy_reference | 3944 | -1.08 | 0.31 | -9.79 | -18.79 | 146 | -30.76 | -13.23 | -56.84 |
| neutral | 4 | `LiveVolCompress` | terminal_negative_reference | 695 | -9.59 | -78.32 | -40.36 | -75.30 | 35 | 43.56 | 17.40 | -3.01 |

## Bullish

### 1. `MomentumScalper__SLOW200`

Медленный трендовый профиль: согласование EMA 12/36/84, подтверждение объёмом и ограниченный lifecycle позиции.

- Роль: `primary_research_lead`.
- Механика: slow multi-EMA momentum with volume confirmation.
- Почему сохранён: Самый высокий historical bullish mean и положительный обычный LCB среди протестированных bullish-профилей.
- Ограничения: Медиана глубоко отрицательна, familywise LCB отрицателен, sanity и раскрытый holdout провалены.
- Фиксированные параметры: `{"CHECK_INT": 2, "EMA_F": 12, "EMA_M": 36, "EMA_S": 84, "VOL_WIN": 36}`.
- Исходник: `src\panteon_runtime\panteon_agents.py::MomentumScalper`; implementation SHA `fac3641746659b6d514aafb57870ab48176585b9ede77430a7984067289ee96e`.

### 2. `NEW_CrossSectionalResidualMomentum`

Ранжирует full8 по 12-барной доходности после вычитания медианного движения рынка и торгует крайние квартили.

- Роль: `mechanism_reference`.
- Механика: market-demeaned cross-sectional momentum.
- Почему сохранён: Большая выборка и лучший общий recent holdout след среди новых простых агентов; полезен как относительный, а не абсолютный momentum baseline.
- Ограничения: Historical и holdout median/LCB отрицательны; sanity ухудшается.
- Фиксированные параметры: `{"lookback": 12, "rank_threshold": 0.75}`.
- Исходник: `src\simple_research\strategies.py::_cross_sectional_residual_momentum`; implementation SHA `6c95f9a9c7994df794df232902b32e160f0218b939c314b2c4188b7ea3168c7d`.

### 3. `LiveOIBreakoutVolumeFallback`

Breakout-профиль с momentum, объёмным всплеском и OI expansion; в OHLCV-ретротесте фактически работает как volume fallback.

- Роль: `breakout_reference`.
- Механика: volume or OI expansion with directional momentum.
- Почему сохранён: Положительные bullish mean и обычный LCB на большой исторической выборке; удобный контраст для cross-sectional и EMA momentum.
- Ограничения: Отрицательная медиана и familywise LCB, сильный drawdown, sanity и holdout collapse; OHLCV fallback не доказывает OI-edge.
- Фиксированные параметры: `{}`.
- Исходник: `src\panteon_runtime\panteon_agents.py::LiveOIBreakout`; implementation SHA `9429f67f6788dcad8ee02d344f13091524e9baf871e09f0aecc767728be30bf1`.


## Bearish

### 1. `LiveCrashHunter__SLOW150`

В bear/crash ищет отрицательный momentum и открывает short; вне bear использует ограниченную RSI mean-reversion логику.

- Роль: `primary_research_lead`.
- Механика: slow crash continuation and RSI-conditioned reversal.
- Почему сохранён: Единственный сильный bearish след с положительными mean, median и обычным LCB на 430 historical сделках.
- Ограничения: Familywise LCB отрицателен, development mean около нуля, раскрытый holdout резко отрицателен.
- Фиксированные параметры: `{"CHECK_INT": 2, "ENTRY_COOLDOWN": 5, "HOLD": 9, "MOM_LB": 18, "RSI_N": 21}`.
- Исходник: `src\panteon_runtime\panteon_agents.py::LiveCrashHunter`; implementation SHA `337111cc48a105b339148b3e1495bab5df6c1d8d2040282cc75492a241b91bce`.

### 2. `SR_CandleMomentum3__SLOW150`

Минимальный causal baseline: направление пяти 4h баров при абсолютном движении более 100 bps.

- Роль: `high_incidence_reference`.
- Механика: five-bar candle momentum threshold.
- Почему сохранён: Высокая частота активации и простая воспроизводимая логика дают полезный bearish контроль для более сложных агентов.
- Ограничения: Отрицательные median и LCB, OOS и holdout mean отрицательны, drawdown чрезмерен.
- Фиксированные параметры: `{"bars": 5, "threshold_bps": 100.0}`.
- Исходник: `src\simple_research\strategies.py::_candle_momentum`; implementation SHA `830b99e4bdf273eab2512ae94f56f4d62f014465b313d660d2365256b17d1b83`.

### 3. `NEW_EfficiencyRatioTrend`

Отделяет направленное движение от шумного пути через efficiency ratio и минимальный 16-барный move.

- Роль: `mechanism_reference`.
- Механика: Kaufman-style path efficiency trend.
- Почему сохранён: Независимая от EMA и RSI трендовая механика; подходит как диагностический фильтр качества bearish движения.
- Ограничения: Median/LCB отрицательны, OOS и holdout mean отрицательны, standalone promotion запрещён.
- Фиксированные параметры: `{"minimum_efficiency": 0.35, "minimum_move_bps": 50.0, "window": 16}`.
- Исходник: `src\simple_research\strategies.py::_efficiency_ratio_trend`; implementation SHA `f9433e28bc859f4765f52834bf6ce464a81a6bc7cca4c230f28c799eeffad600`.


## Neutral

### 1. `MeanRevConfirmedAgent__FAST067`

Ускоренный mean-reversion профиль: BB period 17, подтверждение объёмом, возврат z-score к центру и hard stop.

- Роль: `primary_research_lead`.
- Механика: Bollinger z-score mean reversion with volume confirmation.
- Почему сохранён: Наиболее содержательный neutral historical профиль с положительными mean и median и тремя положительными окнами из четырёх.
- Ограничения: LCB отрицателен, validation mean отрицателен, holdout mean и LCB не подтверждают edge.
- Фиксированные параметры: `{"BB_PERIOD": 17, "CHECK_INT": 1, "ENTRY_COOLDOWN": 3, "HOLD_BARS": 5, "VOL_LOOKBACK": 17}`.
- Исходник: `src\panteon_runtime\agents_v2.py::MeanRevConfirmedAgent`; implementation SHA `5ef916e254af5c1615233f3d2e2d8aeaa8c641176f9038ddd400fff126128016`.

### 2. `NEW_VolumePriceDivergence`

Сравнивает 12-барное движение цены с направлением накопленного signed volume и закрывается у EMA-центра.

- Роль: `distribution_reference`.
- Механика: price and signed-volume divergence.
- Почему сохранён: Положительная neutral median во всех historical окнах и на recent audit показывает необычную, но сильно асимметричную payoff-структуру.
- Ограничения: Mean и LCB отрицательны: частые небольшие выигрыши перекрываются редкими крупными потерями.
- Фиксированные параметры: `{"lookback": 12, "price_threshold_bps": 80.0}`.
- Исходник: `src\simple_research\strategies.py::_volume_price_divergence`; implementation SHA `a1b8f61a19733702259309ad212239d240575e18f87645f1aab9114e360ffb85`.

### 3. `NEW_VolatilityRegimeSwitch`

Переключает одну causal policy между momentum при высокой относительной волатильности и mean reversion при низкой.

- Роль: `hybrid_policy_reference`.
- Механика: high-vol momentum plus low-vol z-score reversion.
- Почему сохранён: Почти нулевой historical mean и слегка положительная median делают его полезным контрольным гибридом для классификации волатильности.
- Ограничения: Historical LCB отрицателен; development/OOS и recent audit отрицательны.
- Фиксированные параметры: `{"long_window": 30, "short_window": 6, "trend_lookback": 6, "z_entry": 1.5, "z_window": 20}`.
- Исходник: `src\simple_research\strategies.py::_volatility_regime_switch`; implementation SHA `742f9b2dd4a1c8d0df98d2200040ba2bafa0d85373368157af5e3e38eb37d4e9`.

### 4. `LiveVolCompress`

BB squeeze с momentum-пробоем и фиксированными stop/target/hold ограничениями.

- Роль: `terminal_negative_reference`.
- Механика: Bollinger-band compression breakout.
- Почему сохранён: Recent neutral holdout был ближайшим к положительному LCB и породил отдельные exit/subregime/direct-state эксперименты.
- Ограничения: История, specialist exits и direct compression-transition не прошли; directional neutral compression family закрыта.
- Фиксированные параметры: `{}`.
- Исходник: `src\panteon_runtime\panteon_agents.py::LiveVolCompress`; implementation SHA `9991dc810b6a3e1b5f407a9a31cab92039dbe9638ef2abc27da17696464c66f8`.

## Правила повторного использования

1. Не включать архивный профиль в live whitelist или ensemble по факту нахождения в архиве.
2. Не подбирать параметры на уже раскрытых validation/OOS/sanity/holdout окнах.
3. Новая попытка требует materially different hypothesis и нового prospective evidence.
4. Обязательны realistic costs, positive mean+median+LCB, direction/regime checks и drawdown gate.
5. `LiveVolCompress` остаётся terminal negative reference; neutral directional compression family закрыта.

## Файлы

- `catalog.json` - полный машинный каталог.
- `selected_metrics.tsv` - history и holdout, отдельные ячейки Excel.
- `window_metrics.tsv` - development/validation/OOS/sanity.
- `source_specs.json` - точные параметры и SHA блоков реализации.
- `agent_catalog.xlsx` - человекочитаемый Excel-каталог.
- `manifest.json` - evidence и artifact hashes.
