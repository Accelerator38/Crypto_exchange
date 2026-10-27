# Genetic Primus: текущий аудит, сравнение агентов и критерии real-money допуска

Дата: 2026-08-26. Статус: research-only. Отчёт не является персональной инвестиционной рекомендацией и не утверждает применимость конкретных требований законодательства к владельцу проекта.

## Краткий вывод

Сейчас ни один генетический или обычный торговый агент не имеет достаточного подтверждения для торговли реальными деньгами. Единственная разрешённая модель — `NoTrade`.

По открытому январскому pilot6 максимальную сумму показал overlay v4 (`+373.909 bps`), но это diagnostic PnL: положительны только 3 из 6 origins, outer LCB отрицателен, внутренние validation/audit gates не пройдены, а в финальном artifact-smoke eligible finalists отсутствуют (`0/5`). Overlay v3 устойчивее по доле origins (`5/6`), но также не имеет ни одного подтверждённого predeployment gate.

На старой годовой validation лучший простой агент `MQ_REL_MOM24` имел LCB `-2.25 bps`, тогда как старый `Genetic_Primus_V3` — `-36.94 bps` и drawdown около 50%. То есть простая momentum-логика обобщалась менее плохо, но даже она не доказала положительное преимущество.

## Что именно сравнимо

Нельзя объединять в один рейтинг результаты с разными интервалами, cadence и protocol semantics. Поэтому ниже две отдельные таблицы:

1. Строгое сравнение на одинаковых шести 3-дневных origins 2025-01-01 — 2025-01-19, одинаковом dataset SHA и одинаковых costs.
2. Историческая validation 2024 года для старых directional genetic и Market Quorum агентов.

## Одинаковый открытый pilot6: overlay и простые агенты

| Агент | Сумма, bps | Compounded, bps | Положительных origins | Trades | Aggregate max DD | Статус |
|---|---:|---:|---:|---:|---:|---|
| Overlay v4 diagnostic | +373.909 | +371.102 | 3/6 | 96 | 3.585% | Не eligible |
| Overlay v3 diagnostic | +205.153 | +206.626 | 5/6 | 144 | 2.969% | Internal gates false |
| Momentum `return_1` | +117.047 | +91.629 | 3/6 | 424 | 11.610% | Baseline, не promoted |
| NoTrade | 0.000 | 0.000 | 0/6 | 0 | 0.000% | Текущий active fallback |
| Overlay v2 diagnostic | -81.657 | -81.657 | 0/6 | 12 | 1.180% | Rejected |
| Mean reversion `return_1` | -1874.407 | -1754.088 | 0/6 | 424 | 17.490% | Rejected |

Stress-результат v3: `+60.482 bps`; v4: `+277.200 bps`. Однако v4 зависит от первого origin (`+418.542 bps`) и имеет отрицательные origins `-118.689` и `-10.430 bps`. Его outer paired LCB `-7.439 bps`, stress LCB `-20.442 bps`, positive fraction 50%.

После исправления eligibility эти raw challenger rows получают `evidence_role=diagnostic_outer_deployment`, поэтому не могут формировать promotion evidence. В финальном artifact-smoke completed eligible outer origins равен нулю.

## Старые genetic и обычные агенты

### Development 2022–2023

| Модель | Trades | Mean, bps | LCB, bps | DD | Результат |
|---|---:|---:|---:|---:|---|
| Genetic Primus V1 | 4272 | 23.36 | +8.73 | 19.84% | Rejected slice gates |
| Genetic Primus V2 | 3319 | 55.43 | +37.47 | 16.56% | Rejected temporal robustness |
| Genetic Primus V3 | 3480 | 62.86 | +46.05 | 16.41% | Development gates passed |
| MQ EMA12/48 Long | 875 | 34.98 | -9.96 | 17.06% | Rejected |

Рост V1→V3 выглядел сильным только на development.

### Validation 2024

| Модель | Family | Trades | Mean, bps | LCB, bps | DD |
|---|---|---:|---:|---:|---:|
| MQ Relative Momentum 24 | Обычный | 759 | 30.79 | -2.25 | 10.88% |
| MQ EMA24/96 Both | Обычный | 626 | 48.98 | -14.05 | 32.72% |
| MQ Relative Momentum 12 | Обычный | 1026 | 13.56 | -15.03 | 21.45% |
| MQ EMA12/48 Long | Обычный | 516 | 70.03 | -21.06 | 21.44% |
| Genetic Primus V3 | Генетический | 1887 | -13.46 | -36.94 | 49.99% |

Ни одна модель не имела положительного validation LCB. Старый V3 потерял преимущество сильнее простых momentum/EMA правил. Это главный исторический признак переобучения.

### Genetic Primus V4 directional

`compact_16_8` содержал 659 параметров. Strict base-cost run: mean `+14.60 bps`, LCB `-0.620 bps`, drawdown `53.91%`; stress LCB `-1.118 bps`, drawdown `54.13%`. Модель была правильно отклонена.

## Что улучшилось в текущем overlay

- Binary ALLOW/ABSTAIN вместо полной directional стратегии.
- Linear architecture: 30 inputs + bias = 31 raw parameter.
- В v4 не более 6 активных features; среднее в pilot6 — 3.33.
- Rolling window 730 дней, retrain/deployment cadence 3 дня.
- Purge 7 четырёхчасовых баров = 28 часов.
- Внутри origin: inner train около 607 дней, validation 60 дней, independent audit 60 дней.
- Scaler обучается только на inner train каждого origin.
- Paired objective против NoTrade, momentum и incumbent.
- Half-life 90 дней с 20% uniform anchor.
- 5 islands × population 4 × 5 generations; максимум 107 зарегистрированных гипотез на origin.
- Общий selection bootstrap и отдельный audit bank.
- Полные dataset/config/fold/bank/model hashes.
- `NO_ELIGIBLE_CHALLENGER` и автоматический fallback.
- Functional action/logit drift на фиксированном 512-row hash-anchor.

## Основные оставшиеся проблемы

1. **Нет положительного internal evidence.** Средний v4 validation LCB `-12.530 bps`, audit LCB `-39.497 bps`; predeployment gate false во всех 6 origins.
2. **Январский период многократно использован.** v2, v3 и v4 уже сравнивались на нём. Он остаётся engineering-open и не может стать prospective sealed evidence. Чем больше вариантов проверяется на одном периоде, тем выше риск backtest overfitting.
3. **Мало outer slices.** Шесть origins — только 18 дней. При block size 5 это фактически около одного полного bootstrap-блока; надёжный LCB невозможен.
4. **v4 PnL концентрирован в одном origin.** Сумма высокая, но positive fraction только 50%.
5. **Drawdown gate относительно NoTrade логически заблокирован.** `max_drawdown_degradation=0` требует от первой торгующей модели DD не хуже нулевого DD NoTrade. Практически любая модель будет отклонена. Нужен отдельный абсолютный risk budget для перехода из NoTrade.
6. **Старый model-drift threshold устарел.** Порог 0.75 был задан для parameter-centroid drift. Новый bounded functional-logit drift лежит в диапазоне 0–1; порог нужно калибровать заново до следующего sealed run.
7. **Minimum trades конфликтует с abstention.** Требование активности может заставлять sparse overlay торговать чаще, чем оправдано. Нужно разделить статистическую достаточность opportunities и обязательное число сделок.
8. **Costs пока модельные.** Base 16/stress 24 bps не подтверждены prospective paper fills, latency, partial fills и проскальзыванием конкретной биржи.
9. **Family-level multiple testing не закрыт.** Лимит 107 контролирует один origin, но полный реестр всех V1–V4, matrix cells, thresholds и ручных решений ещё должен участвовать в Deflated Sharpe/PBO оценке.
10. **Нет execution evidence.** Credentials, paper/live, orders и promotion authority правильно выключены; реальные reconcile, duplicate-order, stale-data, disconnect и kill-switch сценарии ещё не проверены.

## Целевые показатели для допуска реальных денег

Ни один отдельный показатель не достаточен. Ниже предлагается последовательный gate.

### A. Минимум для кандидата на prospective paper

| Метрика | Цель |
|---|---|
| Current candidate status | `ELIGIBLE_CHALLENGER`; каждый торгуемый origin обязан пройти structural, coverage и paired gates |
| Internal validation LCB | `> 0` после selection-bias correction |
| Independent audit LCB | `> 0`, audit не используется для ranking |
| Stress-cost LCB | `> 0` при cost не ниже 1.5× фактического p95 round-trip cost |
| Eligibility | Ни одного обхода `NO_ELIGIBLE_CHALLENGER`; fallback всегда NoTrade |
| Hashes | Dataset, cutoff, folds, banks, scaler и model SHA полностью зафиксированы |

### B. Минимум для небольшого real-money canary

| Метрика | Предлагаемый порог |
|---|---:|
| Sealed non-overlapping outer origins | минимум 60 (180 дней); предпочтительно 120 перед существенным капиталом |
| Positive outer slices | ≥65%; целевой запас ≥70% |
| Paired outer block-bootstrap LCB vs active | >0 на двух независимых seed banks |
| Paired LCB vs NoTrade и momentum | >0 после fees/slippage/latency |
| Stress LCB | >0 |
| Deflated Sharpe probability | ≥95% после учёта всех реально испытанных вариантов |
| Probability of Backtest Overfitting | ≤10% |
| Historical max DD | ≤10% и внутри заранее установленного абсолютного risk budget |
| Symbol concentration | ≤35% trades или risk contribution на один символ |
| Direction concentration | ≤75–80% без отдельного directional mandate |
| Market-state concentration | ≤70% |
| Functional action drift | ≤20% на fixed anchor |
| Bounded functional-logit drift | начальный лимит ≤0.15; затем калибровать по стабильным incumbent cycles |
| Prospective paper/shadow | минимум 30 дней / 10 origins без изменения objective |
| Execution incidents | 0 duplicate orders, 0 unreconciled positions, 0 stale-data trades, 0 критических исключений |

Порог 60 origins является лишь минимумом: при пяти-origin block bootstrap он даёт около 12 блоков. Для масштабирования разумнее 120–180 origins либо эквивалентное число независимых режимных блоков.

### C. Операционный canary

Предлагаемые консервативные стартовые ограничения, которые нужно адаптировать к капиталу и юрисдикции:

- без leverage;
- не более 0.25–1% планируемого торгового капитала на первом canary;
- gross exposure не выше canary allocation;
- не более 25–35% canary risk на один символ;
- автоматические pre-trade notional, price-collar, message-rate и duplicate-order limits;
- hard daily-loss stop и cumulative canary drawdown stop, заданные до запуска;
- cancel-on-disconnect, reconciliation по exchange fills и независимый kill switch;
- ключи без withdrawal permission;
- масштабирование не более чем в 2 раза после каждого подтверждённого 30–60-дневного этапа.

FINRA рекомендует версионирование и независимое тестирование алгоритмического кода, возможность быстро отключить алгоритм, пилот ограниченного размера и повышенный мониторинг после запуска. SEC Market Access Rule для подпадающих под него участников требует автоматических pre-trade controls и заранее установленных credit/capital limits. Эти материалы используются здесь как инженерные ориентиры, а не как утверждение применимого права.

## Рекомендуемый roadmap

### 1. Исправить measurement/risk contract до нового objective

- Заменить `drawdown не хуже NoTrade` комбинацией: абсолютный DD budget + сравнение с торгующим incumbent при его наличии.
- Калибровать новый functional-logit drift threshold; не переносить старое значение 0.75.
- Отделить minimum opportunities/ESS от minimum trades, чтобы abstention не штрафовался за разумное бездействие.
- Вести глобальный immutable registry всех испытанных objectives/config hashes, а не только per-origin budget.
- Добавить DSR и CSCV/PBO в promotion report.

### 2. Улучшать модель без расширения архитектуры

- Сохранить binary linear overlay; hidden layer сейчас не оправдан.
- Использовать stability selection features: частота выбора и знак коэффициента по islands/origins.
- Отклонять features с нестабильным знаком или низкой bootstrap inclusion frequency.
- Добавить opportunity-normalized concentration penalties, а не только доли совершённых сделок.
- Добавить hysteresis/cooldown или минимальное время удержания для снижения turnover, но зарегистрировать правило до нового outer period.
- Сравнить single genome с consensus top-3 только внутри training/validation; audit остаётся исключительно gate.
- Сохранять NoTrade, momentum и incumbent как обязательные paired references.

### 3. Новый temporal эксперимент

- Все решения по objective зафиксировать до просмотра нового периода.
- Использовать старые данные для nested rolling development, но не возвращаться к январю для выбора параметров.
- Открыть новый sealed interval минимум на 60 трёхдневных origins.
- Не менять config по ходу interval; при изменении начать новый experiment family и пересчитать multiple-testing correction.

### 4. Paper и только затем canary

- Shadow/paper 30–90 дней с реальным market-data timing и симулированными exchange constraints.
- Калибровать base/stress costs по фактическим p50/p95 fills.
- Провести fault injection: disconnect, stale candle, duplicate response, partial fill, rate limit, restart, clock skew.
- После полного statistical и operational gate — небольшой canary с manual approval и жёстким kill switch.

## Итоговый рейтинг

- **Лучший разрешённый агент сейчас:** NoTrade.
- **Лучший по сумме diagnostic pilot6:** overlay v4, но статистически неустойчив и не eligible.
- **Лучший по стабильности diagnostic pilot6:** overlay v3 (5/6), но internal gates false.
- **Лучший старый агент по validation LCB:** простой MQ Relative Momentum 24 (`-2.25 bps`), но LCB всё равно отрицателен.
- **Главный исторический вывод:** genetic search лучше оптимизировал development, но пока не доказал лучшее temporal generalization.

## Внешние первичные источники

- Bailey et al., [Effects of Backtest Overfitting on Out-of-Sample Performance](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659).
- Bailey & López de Prado, [The Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf).
- [FINRA Regulatory Notice 15-09: algorithmic trading supervision and controls](https://www.finra.org/industry/notices/15-09).
- [SEC Risk Management Controls for Brokers or Dealers With Market Access](https://www.sec.gov/rules-regulations/2011/06/risk-management-controls-brokers-or-dealers-market-access).
