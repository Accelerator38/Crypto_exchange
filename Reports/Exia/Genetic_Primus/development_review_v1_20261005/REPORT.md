# Genetic Primus: результаты, алгоритмы и дальнейшая разработка

Дата обзора: 5 октября 2026 года. Все периоды ниже — UTC, правая граница исключена.

## 1. Главный вывод

Сейчас нет торговой модели с подтверждённым устойчивым положительным net-преимуществом над NoTrade.
Это не означает, что все исторические прогоны убыточны: в августовской диагностике GA-overlay
получил +15,65%. Однако положительный эпизод не перенёсся на последующие проверки, а нижняя
граница дополнительного эффекта над EMA не была положительной.

Инженерная система стала лучше: появились причинное переобучение, непрерывный portfolio ledger,
фиксация данных/кода/моделей, отдельные банки и fail-closed разрешения. Торговый edge при этом
не доказан. Прохождение технических тестов нельзя интерпретировать как прохождение финансовых gates.

Последний fixed reversal-кандидат остановлен: normal -2,1614%, stress -0,9555% после
дополнительной комиссии конечного закрытия. Все 20 его slope равны нулю. Incumbent — NoTrade.
Торговые credentials, orders, paper/live и promotion authority остаются false.

## 2. Объём и ограничения анализа

Проведена статическая инвентаризация всех 1784 JSON-файлов внутри Genetic_Primus:
ни одного файла не пропущено по установленному ограничению 2 MB. Создан индекс с путями,
SHA-256, схемами и извлечёнными метриками; большие массивы результатов намеренно сокращены.
Предметно сопоставлены ключевые development, validation, matrix, attribution, prospective,
calibration и execution-артефакты; прочитан соответствующий актуальный исследовательский код.

Проверены 23 связи previous-ledger SHA между 24 версиями журнала: несовпадений нет.
Это подтверждает целостность сохранённой цепочки, но не полноту всех когда-либо сделанных
ручных адаптаций, выборов признаков и просмотренных результатов.

Во время обзора не запускались модели, bootstrap, обучение, pytest, collector, сеть или trading runtime.
Старые NPZ/parquet/графики не пересчитывались независимо: старые метрики здесь — сохранённые
результаты, а не новый воспроизведённый backtest. Не изучалась длинная история чатов.
Текущий обзор не является аудитом всей системы Panteon или её реальных торгов.

Данные основного OHLCV-снимка: 4h, full8 — ADA, BNB, BTC, DOGE, ETH, LINK, SOL, XRP / USDT;
покрытие 2022-01-01 — 2026-09-28, 83 088 строк, 10 386 временных отметок.
Сохранённый data-integrity audit показывает ноль duplicates, missing-symbol bars,
off-grid timestamps и zero-volume bars, согласованные конечные положительные OHLC.
В последнем пилоте все 11 parquet-файлов также проверены по реальным byte hashes.
Число bars/assets нельзя принимать за число независимых торговых наблюдений.

## 3. Топ-5 сопоставимых результатов

Единый глобальный рейтинг по всем запускам был бы недостоверным: различаются периоды,
стоимости, горизонты, экспозиция и даже единица измерения. Поэтому основной топ-5 составлен
по одному общему непрерывному replay: **2026-09-01 — 2026-09-28, 27 дней**.
Ранжирование — по итоговому normal net; stress служит дополнительным наблюдением.
В этом артефакте normal/stress — 16/24 bps roundtrip, не новый сценарий 40 bps + latency.

| Место | Политика | Normal net | Stress net | Normal DD | Завершённые trades |
|---:|---|---:|---:|---:|---:|
| 1 | NoTrade | 0,0000% | 0,0000% | 0,0000% | 0 |
| 2 | GA_target_gate | -1,4301% | -1,8719% | 4,0467% | 45 |
| 3 | GA_entry_only | -1,4801% | -1,8811% | 3,7228% | 41 |
| 4 | EMA_long_only | -2,1519% | -2,6163% | 4,2349% | 48 |
| 5 | EMA_TrendConsensus | -6,9576% | -7,7631% | 8,8679% | 87 |

Источник: [baseline economics audit](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/baseline_economics_audit_v1_20261002/result.json).

Это рейтинг фактических архивных политик, не алгоритмов при равном search budget и не
независимый production leaderboard. GA_target_gate не получает разрешение на торговлю:
он лишь меньше теряет, чем EMA. Преимущество над GA_entry_only составляет около 0,05 п.п.
normal net — слишком мало для вывода о статистическом превосходстве по одной реализации пути.
У GA_entry_only несколько меньшая просадка. Все торгующие участники проиграли cash.

Экспозиции и durations этих пяти политик различаются. Более строгая отдельная атрибуция
сохраняет geometry позиций/оборот и сравнивает GA с 15 controls:

- GA normal -1,4301%, медианный control -1,1229%.
- GA stress -1,8719%, медианный control -1,5672%.
- GA лучше только 1 из 15 controls; paired LCB разницы составляет -5,1985 и -5,2863 bps/day.
- Сам artifact имеет historical_support_gate=false: допустимые перестановки ограничены.

Вывод: добавленная ценность именно GA-selection не подтверждена. Это не доказательство,
что никакой GA никогда не сможет работать, но оснований расширять поиск данного кандидата нет.
Источник: [matched attribution v3](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/episode_attribution_v3_20261002/result.json).

### Сильные положительные исторические эпизоды — отдельно от рейтинга

| Артефакт | Сохранённый результат | Почему это не production-победитель |
|---|---|---|
| EMA_long_only, reference validation в selection-development | +16,9286% normal; +16,3010% stress | Reference уже раскрыт; challenger не выбран, validation_result=null |
| GA_overlay_f7a4ae11, revealed August diagnostic | +15,6450% normal; +14,9313% stress | Engineering/development-only; paired HAC LCB vs EMA отрицателен |
| EMA_TrendConsensus, тот же August diagnostic | +11,0010% normal; +9,6560% stress | Результат не сохранился в сентябре/июньском screen |
| V3 development | +62,86 bps/trade, LCB +47,05 bps/trade | На validation mean -13,46, LCB -38,31 bps/trade |
| V4 strict development | Mean +14,60 bps/day | LCB -0,62 bps/day, DD 53,91%, нулевые entries в 2024 folds |

EMA reference validation относится к 2026-07-31 — 2026-08-30, определённым 20 training и
10 validation origins от 1 июня. August diagnostic состоит из 10 раскрытых origins / 30 дней.
Периоды частично пересекаются; эти положительные результаты нельзя считать независимыми подтверждениями.
Таблица намеренно не сортирует bps/trade, bps/day и portfolio net как одну метрику.

В August diagnostic normal paired HAC LCB GA−EMA равен -10,0431 bps/day при положительной
средней разнице +13,4090 bps/day. Выборка короткая и неопределённость велика.
Положительных origins относительно NoTrade только 4/10; normal HAC LCB относительно
NoTrade также отрицателен (-173,5984 bps/day). Положительный total net не равен пройденному gate.
Источники: [August summary](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/revealed_august_diagnostic_v1_20260904/summary.json),
[selection result](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/selection_development_v1_20261001/result.json),
[V3 validation](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/development_v3_100g_20260821/validation_metrics.json).

## 4. Актуальные алгоритмы и их статус

| Контур | Что делает | Размер/адаптация | Статус |
|---|---|---|---|
| NoTrade | Всегда cash, никаких позиций | 0 обучаемых коэффициентов | Текущий incumbent |
| EMA_TrendConsensus | Согласует EMA gap 12/48, slope 12/3, return_12 и efficiency>=0.30 | Фиксированные правила; legacy допускает обе стороны | Есть прибыльные и убыточные эпизоды, устойчивость не подтверждена |
| EMA_long_only | Использует только long-сигналы EMA | Фиксированная простая политика | Лучше двусторонней EMA в сентябре, но отрицательный net |
| Relative momentum | На origin выбирает top-2 по предыдущим 18 барам с фильтром положительного движения | Без fitted parameters; удержание по 3-дневному протоколу | Июньский training gate не пройден |
| Frozen GA-overlay f7a4ae11 | Binary allow/veto поверх EMA | Genome [0,0,1,1,-1]: 5 stored coefficients, 3 nonzero; веса заморожены | Fixed historical challenger, не production-модель |
| GA target gate | Overlay разрешает/запрещает целевую позицию, включая влияние на выход | Linear binary allow, без большого MLP | Уменьшает потери EMA, но не beats NoTrade/controls |
| GA entry-only | Overlay фильтрует входы; exit не превращается в постоянный повторный veto | Отдельная ablation-политика | -1,4801% в общем September replay |
| Ridge overlay, 2/4 features | Прогнозирует gross 3-day return среди EMA proposals; учитывает costs при переходах | 3/5 коэффициентов с intercept; uniform или half-life 90d + 20% anchor | Все четыре June arms не прошли training gate |
| Market-relative reversal | Один relative-return feature, pooled ridge, slope<=0, новые scaler/fit каждые 3 дня | 2 коэффициента, uniform; без EMA proposal filter | Последний пилот провален, slope=0 на всех origins |
| Future EMA viability | Отдельный fixed long-only baseline experiment, без GA или поиска | 60 origins/180 дней, единственный final analysis | Зарегистрирован, не оценён автоматически; не даёт trading authority |
| V8 Clean collector | Диагностический сбор/проверка frozen V7 LINK hypothesis, fail-closed safety | Transport/status/ledger, не новый доказанный learner | Не запускается; diagnostic root после UI crash не prospective evidence |

Legacy Momentum_return_1, MeanReversion_return_1 и RelativeMomentumRank_return_12 также
существуют в benchmarks_v4. Они отличаются от top-2/18-bar relative momentum: нельзя
объединять результаты под одним словом «моментум».

Исторические V1–V3 — самостоятельные long/flat/short neural policies. У V3 30 inputs,
hidden 32/16, 3 outputs: 1571 параметр. Прогон использовал population 64 и 100 generations.
V4 compact_16_8 имеет 659 параметров. Большая capacity/бюджет дали хороший development,
но не устойчивую validation. Современный sparse overlay существенно проще; простота сама
по себе не создаёт сигнал.

В rolling_v4 реализованы islands, warm-start, 20–30% random immigrants и ансамбли;
legacy ансамбли 3/5, modern branch 1/3. Это возможности кода, не подтверждение их alpha.
В September frozen prospective genetic_search_during_outer=false и genome_updates=false:
там **refit scaler**, а не полное повторное обучение GA. Это важное отличие от последнего
reversal evaluator, который действительно переобучал коэффициенты на каждом origin.

Источники кода: [binary policy](E:/Crypto_exchange/src/exia/genetic_primus/policy_v4.py),
[rolling search](E:/Crypto_exchange/src/exia/genetic_primus/rolling_v4.py),
[rolling ridge matrix](E:/Crypto_exchange/src/exia/genetic_primus/rolling_linear_matrix_v1.py),
[reversal fitter](E:/Crypto_exchange/src/exia/genetic_primus/market_relative_reversal_linear_v1.py),
[September contract](E:/Crypto_exchange/configs/genetic_primus_prospective_3d_v1.json).

## 5. Какие теории проверены

| Гипотеза | Что наблюдали | Вывод / предел доказательства |
|---|---|---|
| Больше GA generations/сложнее сеть → лучше модель | V3 strong development, затем отрицательная validation и DD ~49,99% | Научение development не равно переносимости |
| Compact MLP решит generalization | V4 659 params, положительный mean, отрицательный LCB, DD ~54%, empty 2024 folds | Capacity уменьшена, но проблема не решена |
| Frozen event thresholds найдут редкий качественный edge | V6/V7 terminal rejected: no event capacity | Нет достаточной поддержки для допуска; не надо компенсировать forced trading |
| Binary sparse GA лучше полной стратегии | Ниже оборот/убыток EMA в общем replay | Архитектурно разумно, но дополнительный prediction edge не подтверждён |
| Хороший August результат достаточен | GA +15,65%, но LCB vs EMA<0 и subsequent losses | Не достаточен; период чувствителен к режиму |
| Warm-start нужен при каждом обновлении | 10 August origins, cold/warm по 240 evaluations, 9 одинаковых winners, 0 screen passes | Warm-start не устранил отсутствие signal; paired search budget важен |
| Recency fitness улучшает адаптацию | Weekly five-cell matrix: все audit/stress LCB<0, promotion-pass fraction=0 | Нет подтверждённого улучшения этой реализации |
| Half-life 90d лучше uniform в ridge | В June matrix half-life arms abstain; uniform торгует с отрицательным net | Нулевой результат — cash, не прибыльный learner; нельзя называть его winner alpha |
| Relative momentum даст простой сигнал | June normal -1,9431%, stress -2,1303%, оба LCB vs NoTrade<0 | Fixed training gate не пройден |
| Market-relative reversal устранит рыночный bias | 20 fits, slope всегда 0; 36 cash days, затем full8 hold 24 days; отрицательный gross/net | Fixed negative-slope signal не подтверждён; модель свелась к intercept |
| GA помогает выбором trade episodes | Matched controls median лучше GA, GA beats 1/15; paired LCB<0 | Зафиксированное GA-selection преимущество не доказано |
| Проблема только в комиссиях | EMA, relative momentum и reversal имеют отрицательный gross на соответствующих June paths | Одного уменьшения costs недостаточно |

Weekly matrix, сохранённые audit paired LCB / stress LCB, bps:

| Вариант | Audit paired LCB | Stress LCB | Promotion fraction |
|---|---:|---:|---:|
| Expanding, uniform | -16,0960 | -15,2841 | 0 |
| Rolling 730d, uniform | -4,3124 | -4,6026 | 0 |
| Rolling 730d, half-life 180d | -4,8255 | -5,2380 | 0 |
| Rolling 365d, half-life 90d | -11,3947 | -9,8841 | 0 |
| Rolling 180d, half-life 30d | -14,9189 | -14,6442 | 0 |

Эти cell summaries имеют 5 comparable origins и legacy измерительные semantics.
Различия отрицательных величин не дают разрешение считать вариант победителем или
переносить эти units прямо на текущий causal 3-day evaluator.

June matrix normal net: EMA/2-feature uniform -8,5059%; 4-feature uniform -7,9706%;
оба half-life arms 0%, 0 trades. Challenger validation не открылась: selected_candidate=null.
Новый reversal pilot несопоставим с ними по exposure и exit semantics, поэтому меньший его
убыток нельзя объявлять победой другой гипотезы. Его отдельные normal/stress costs также иные.

Источники: [weekly cells](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/matrix_weekly_rolling730_uniform_outer6_20260826/cell_summary.json),
[warm-start](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/warmstart_august_v1_20260907/summary.json),
[June matrix](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/rolling_linear_matrix_v1_20261002/result.json),
[relative momentum](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/relative_momentum_development_v1_20261001/result.json),
[reversal report](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/reversal_calibration_pilot_v1_20261005/README.md).

## 6. Актуальная схема расчёта — последний причинный evaluator

Последний запуск — retrospective calibration, не production и не новый sealed OOS.

1. До fit фиксируются objective, один candidate, 20 folds, data/source/code hashes и cutoff.
2. Reader проверяет все snapshot bytes и ставит date predicates до parquet decoding.
   Каждый origin прочитал 35 240 свечей; будущие September/EMA цены не подавались.
3. Decision features строятся только по завершённым 13 предыдущим 4h candles.
4. Собираются завершённые трёхдневные training labels на сетке origins, rolling 730d.
5. Purge: label_end <= origin−7×4h. На этой 3-day grid latest label фактически
   закончилась за 72h до deployment; 242 origins × 8 assets = 1936 rows на fit.
6. Median/IQR scaler и ridge coefficients пересчитываются заново на каждом origin.
7. Решение фиксируется с model/scaler/data/folds/weights SHA **до valuation future candles**.
8. Ledger переносит cash, quantity и targets между origins; отдельно считает normal/stress.
9. Из NAV по 4h closes выводятся все календарные daily returns, включая cash days и убытки.
10. Один final analysis после всех origins; при отрицательном development net — NoTrade и STOP.

### Основные формулы и units

Relative feature: xᵢ,t = return₁₂ᵢ,t − medianⱼ(return₁₂ⱼ,t), для фиксированного full8.
Scaled feature: zᵢ,t = clip((xᵢ,t−median_train)/IQR_train, −8, 8).

Target: yᵢ,t = closeᵢ,t+3d−4h / openᵢ,t − 1. Это gross fractional return,
не win/loss flag, не cost-adjusted label, не только успешные исходы.

Fit минимизирует mean[(y−β₀−β₁z)²] + 0,5β₁² при β₁<=0.
Intercept не штрафуется. Проекция slope в ноль с пересчётом intercept корректна для
зарегистрированной constrained ridge; отсутствие signal — не автоматически ошибка численного solver.

Если позиция flat, long entry разрешён при forecast > (1+f)/(1−f)−1;
если позиция уже long, continuation threshold=0. При normal one-way f=0,0008
entry threshold ~0,00160128, то есть 16,0128 bps. При недостаточной temporal support,
нулевом IQR или недоступном прогнозе — abstain.

Buy quantity = cash/[open×(1+f)]; sell cash = quantity×open×(1−f).
Portfolio NAV — средняя стоимость восьми независимых sleeves, изначально по единице cash.
Daily return R(d) = NAV(d)/NAV(d−1) − 1; общий net = ∏(1+R(d))−1,
а не сумма «средних bps на сделку». Конечная liquidation-комиссия показана отдельно.

Historical recency weights: wᵢ = 0,2/N + 0,8×2^(−ageᵢ/h) / Σⱼ2^(−ageⱼ/h),
h=30/90/180 дней; ESS=(Σw)²/Σw². Все исходы, включая убытки, должны иметь такой же time weight.
В последнем reversal fit используется только uniform: новые half-life arms не запускались.

GA fitness_v4 умеет paired incremental LCB относительно NoTrade и baseline, worst-window
LCB, recency и penalties за DD, turnover, complexity, active features и instability.
Modern ranking lexicographic: coverage, structural failures, adjusted minimum paired LCB,
absolute/baseline LCB, recency/worst-window и далее risk/parsimony tie-breaks.
Не следует превращать штрафную сумму с произвольным масштабом в единственный критерий допуска.

Bootstrap samples общие для сравниваемых genomes. Audit seed bank отдельный от selection
bank, но оба используют одни и те же market observations: это не независимый market interval.
Пилот reversal использовал descriptive HAC lags 0/4/9/14, не promotion bootstrap;
sigma-envelope не confidence bound. Signal fraction 8/20 включает continuations,
а все восемь входов произошли в одном origin — это не восемь независимых signals.

Источники: [bounded reader](E:/Crypto_exchange/src/exia/genetic_primus/reversal_calibration_reader_v1.py),
[builder](E:/Crypto_exchange/src/exia/genetic_primus/market_relative_reversal_builder_v1.py),
[rolling evaluator](E:/Crypto_exchange/src/exia/genetic_primus/market_relative_reversal_rolling_v1.py),
[fitness](E:/Crypto_exchange/src/exia/genetic_primus/fitness_v4.py).

## 7. Проблемы и остаточные риски

### Критические для выводов о модели

1. **Нет абсолютного edge.** Уменьшение потерь EMA часто объясняется меньшей экспозицией.
   Даже matched controls не подтвердили дополнительный selection-эффект.
2. **Разные измерительные semantics.** В истории смешаны bps/trade, bps/day,
   closed-origin chaining и continuous NAV. September «9 origins + 1 post-period diagnostic»
   показывал GA -2,5075%, EMA -8,2186%, но имел qualification_allowed=false.
   Это другой контур, не цифры непрерывного 27-day replay в топ-5.
3. **Исторический horizon mismatch.** September contract имеет labels 7×4h,
   deployment 3 days и frozen genome. Это не обучение 72h-horizon модели каждые 3 дня.
   Последний evaluator исправляет horizon/refit, но не отменяет ограничений старых результатов.
4. **Adaptive multiple-testing не закрыт.** Hash-linked registry полезен, но неизвестные
   исторические decisions остаются. Номинальные LCB/Bonferroni локального screen не покрывают
   все смены objectives, thresholds, features, scopes и просмотренные validation references.
   Наличие global_multiple_testing.py не равно calibrated global DSR/PBO.
5. **Малая независимая temporal support.** Correlated full8 rows, continued positions,
   overlapping/revealed origins и длинные эпизоды существенно слабее такого же числа
   независимых временных сигналов. 1936 fitted rows — не 1936 независимых bets.
6. **Intercept заменил гипотезу.** При β₁=0 последний reversal learner стал оценкой средней
   будущей доходности рынка, но по-прежнему мог войти full8. Явный «feature edge отсутствует»
   gate не был частью этой гипотезы. Добавлять его задним числом для улучшения этих чисел нельзя.
7. **Stress смешивает две причины.** Более дорогие costs + задержка могут дать лучший путь
   цен. Требуется различать cost-only и latency stress; правило worst-case необходимо
   регистрировать заранее для будущих контрактов, а не выбирать удачную задержку постфактум.

### Код и исполнение

8. **Много legacy branches/defaults.** В promotion_v4 functional-logit drift может
   fallback на max_normalized_model_drift; legacy default 0.75 ещё присутствует.
   Это риск ошибочного переиспользования старого threshold, не доказанная причина последнего
   pilot loss: promotion в нём не исполнялся. Для нового registered режима missing explicit
   drift semantics/threshold должны fail closed.
9. **Fallback drawdown недостаточно строг.** promotion_v4 при отсутствии portfolio
   может брать максимальный origin drawdown, а не весь непрерывный NAV path. Нужен обязательный
   continuous portfolio для release gate; это предложение, код в обзоре не менялся.
10. **Support vs forced trading.** Legacy minimum_trades checks остаются в fitness_v4.
    Для abstention-веток нужно явно разделять opportunities/ESS и trades и фиксировать config,
    а не предполагать, что старый путь автоматически использует новые правила.
11. **Execution не сертифицирован.** OHLCV open/close, fixed bps и delay не дают order-book
    depth, spread, funding, queue/partial fills, latency distribution и liquidation conditions.
    Tape presence report содержит четыре архива, но execution_ready=false, full integrity
    и price/fill quality там не проверены. Архивы другого дня/актива не валидируют full8 June/September.
12. **Цена crash/resume.** В неактивном collector остаются wall-clock reconnect loops,
    asyncio waits и fsync status. Причина прежних UI/PC crashes не установлена; этот обзор
    collector не запускал. Offline research специально обходит его. Независимая проверка
    на другой машине отложена, не «пройдена».

Отсутствие сильного feature drift у раннего V3 не спасло validation: как возможное объяснение,
может меняться conditional relation return|features, а не только распределение features.
Это гипотеза, не установленная по одному manifest причинная атрибуция.

## 8. План дальнейших действий

### Этап A — закончить унификацию без нового model search

- Единая typed schema результата: objective/model/data/folds/code/bank hashes,
  train/inner/outer dates, revealed status, horizon, cadence, position carry, terminal liquidation,
  costs/funding scope, actual model refits и effective feature count.
- Разделить engineering smoke, development, inner-validation и sealed outer в catalogue.
- Мигрировать ключевые summaries на один continuous NAV ledger; старые числа не перезаписывать.
- Явно перечислить известные adaptive trials; неизвестные оставить unknown, не заполнять нулём.

Если старую историю восстановить нельзя, не выдавать DSR/PBO по неполному каталогу за global
correction. Ввести полностью учитываемый новый research epoch и заранее фиксированный budget
будущих независимых проверок; старые результаты оставить descriptive. Нельзя бесконечно
переоткрывать «новый outer» после failure без учёта последовательных испытаний. Свежая
пререгистрированная проверка и обучение на раскрытом прошлом — разные источники доказательства.

Критерий готовности: один и тот же committed decision schedule даёт одинаковый NAV,
fees и DD во всех wrappers; ни один отчёт не называет engineering результат prospective.

### Этап B — проверить наличие экономического сигнала, а не расширять GA

Новая гипотеза требует отдельного разрешения/регистрации. До её выбора полезна ограниченная
диагностика на уже раскрытом training/development, не на будущей EMA-ветке:

1. Отделить common market direction от cross-sectional asset selection.
   Например, asset-selection target — residual относительно market basket, а не абсолютный
   gross return, dominated общей intercept. Это новая гипотеза, не ретюнинг failed reversal.
2. Проверить знак/устойчивость связи на temporal blocks, число независимых opportunities
   и влияние каждого feature. Нулевые или нестабильные коэффициенты должны давать explicit no-edge.
3. Сравнивать с NoTrade, allow-all, fixed momentum и matched duration/exposure/turnover controls.
4. Сначала одна regularized linear specification; максимум два зарегистрированных arms
   uniform/half-life 90d. Не менять одновременно feature family, threshold, lookback и costs.
5. В цель включать и выигрыши, и убытки. Признаки/scaler/labels каждого origin — только доступные
   на этот момент; losses не исключать из обучения и fitness.

Критерий STOP: отсутствие положительного absolute net/LCB и стабильного сигнала — закрыть
гипотезу. Не запускать GA, не менять sign/threshold на том же validation для получения красивой цифры.

### Этап C — GA только после простого signal baseline

Если простой learner проходит registered training/inner gates, сравнить GA со случайным
поиском и линейным optimizer при **одинаковом** числе evaluations, data folds и budgets.
Небольшие islands/immigrants, 5–10 generations и sparse linear binary overlay уже доступны
как инфраструктура. Их применение должно решать конкретную задачу, а не заменять слабый signal.
Заранее выбрать ensemble/selection rule; не выбирать seed по outer результату.

Fitness/selection: одновременно absolute paired LCB vs NoTrade и incremental LCB vs
matched reference; opportunity support отдельно от trades; registered concentration,
hysteresis/cooldown и functional drift. Admission при NoTrade — absolute risk budget,
а не требование DD<=0. Численный risk budget нового learner отдельно согласуется;
10% ниже относится только к уже зарегистрированному EMA-контракту.

### Этап D — три temporal уровня и единственный outer analysis

Внутри development можно использовать temporal validation при поиске generations,
но тогда она становится частью обучения **процедуры выбора**. Independent outer должен
оставаться непросмотренным. На outer воспроизводится весь заранее заданный pipeline:
refit scaler/model каждые 3 дня, purge, continuous state, одинаковые costs и model SHA на решение.
Inner выбирает только одного победителя; outer не выбирает замену среди проигравших кандидатов.

Пользовательская цель — неубыточный итог при допустимых временных просадках. Для торгующего
агента это нельзя гарантировать. Проверяемый net должен включать все relevant costs;
положительный observed net недостаточен без support/uncertainty/risk checks.
Нулевой design effect не даёт конечного размера выборки для высокой power у строгого
positive-mean теста: нужен отдельный положительный scenario effect для size planning,
не обещанная доходность. Текущая market power не калибрована.

### Существующий EMA prospective — не менять и не читать результаты заранее

Уже зарегистрировано 2026-10-05 — 2027-04-03, 180 дней / 60 origins.
Fixed long-only EMA, normal 16 bps/no delay; stress 40 bps/1 bar delay.
4096 bootstrap samples, block 5 calendar days, alpha=0.05,
primary/audit seeds 20261011/20261012, minimum 20 signal origins.
Gate: positive net и paired LCB vs NoTrade в обоих банках/сценариях,
4 из 6 positive stress slices, DD<=10%, positive origin PnL concentration<=40%.
Недостаточная поддержка — inconclusive без продления, failure закрывает текущую EMA-overlay ветку.
PASS разрешает только отдельный последующий GA research objective, не реальные деньги.
automatic_evaluation=false; контракт сам по себе не означает запущенный фоновой расчёт.

Этот интервал нельзя переиспользовать для подбора learner из этапов B/C.
Новые cost-only/latency gates предлагаются для будущих контрактов; текущий замороженный
EMA-контракт не менять в процессе ради улучшения результата.
Источник: [EMA viability contract](E:/Crypto_exchange/configs/genetic_primus_future_ema_viability_v1.json).

### Этап E — инженерная готовность отдельно от alpha

Пока будущий interval не завершён, допустимо проектировать offline synthetic проверки:
causal event ordering, bid/ask execution, fee/funding timestamps, missing/duplicate/out-of-order
data, bounded crash/resume, stale model, partial fills, risk limits и fail-closed blocks.
Подключать реальные tape/сеть только после отдельного разрешения; текущий анализ его не даёт.
Paper/shadow и затем ограниченный canary требуют отдельного контракта и разрешения,
статистически прошедшего кандидата, реалистичного execution и заданного денежного loss budget.
Сейчас этот переход не оправдан. На данном этапе главное — не новый большой train,
а проверяемая гипотеза и честное решение STOP при отсутствии signal.

## 9. Проверка этого отчёта

- Топ-5 взят из одного конкретного economics artifact; units — проценты portfolio net/DD.
- June и August не объединены с September; bps/trade не ранжируются вместе с bps/day.
- Все 24 ledger versions найдены, все 23 hash links проверены.
- Полный JSON inventory: 1784/1784, parsing/size skips=0; binary artifacts не переисполнены.
- SHA последнего calibration result и future EMA contract сохранены; эксперименты не изменены.
- Запускался только static inventory builder: его py_compile и direct pure-function checks прошли.
- Новых fits, кандидатных trials, market evaluations, statistical tests, разрешений или orders — 0.

Сопоставимый ranking также сохранён в ranking.json рядом с отчётом.
Подробный локальный evidence_index_v2.json содержит текущие byte hashes и извлечённые поля;
это объёмный служебный индекс, не новый registry и не подписанное подтверждение provenance.

Ближайший рекомендуемый шаг: **A — единая measurement/result schema и небольшой закрытый
набор regression invariants**, затем отдельная регистрация гипотезы B. Не переобучать
проигравший reversal и не объявлять GA победителем по тому, что он меньше потерял, чем EMA.
