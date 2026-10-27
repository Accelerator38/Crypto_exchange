# Общий offline portfolio replay v1

2026-09-10. Отдельное инженерное продолжение release/execution v1. Старые
контракты, matching kernel, tests/receipts от 8 сентября и prospective не
изменяются. Ключи, ордера, сеть, collector и торговый runtime не используются.

## Назначение

У трёх контрфактических агентов — GA, EMA и NoTrade — одинаковые рынок,
капитал, комиссия, funding, правила инструментов, задержка, защитная цена и
ограничения риска. Каждый агент получает собственную копию кошелька и
ликвидности: результаты не зависят от порядка запуска конкурентов.

GA — прежний continuous binary overlay `(0,0,1,1,-1)`, не новое обучение.
EMA определяется из тех же четырёх RAW features по зафиксированному правилу
`measurement_lab_v2`: согласованные знаки gap/slope/return и efficiency ≥0.30.
Направление не передаётся отдельно для каждого агента. NoTrade всегда flat.
Отсутствующие признаки дают abstention; неверный формат чисел — ошибка,
а не незаметное превращение всех агентов в NoTrade.

## Семантика капитала и решений

- Начальный капитал делится на **8 равных виртуальных долей**, по одной на
  инструмент. Funding, fees и realized PnL относятся к своей доле. Сумма
  balances/equities сверяется с общим futures ledger на каждом event time.
- Неактивные доли не перераспределяются. Положительный PnL одного инструмента
  не увеличивает размер следующего входа другого. Общий equity знаменатель
  одинаков у GA/EMA/NoTrade, idle cash не исключается из доходности.
- Новый вход рассчитывается по собственному marked sleeve equity с резервом
  комиссии. Same-side решение удерживает фактически исполненную quantity:
  частичная позиция не доливается и не перебалансируется каждый бар.
- При нулевом target создаётся reduce-only выход, в том числе если собственный
  equity неположителен. При противоположном target — один net-delta IOC;
  неполный разворот остаётся реальной промежуточной позицией.
- Защитные лимиты рассчитываются по свежему mark, известному **на момент
  решения**, плюс/минус явно зарегистрированная fraction. Будущие ask/bid
  не участвуют ни в sizing, ни в выборе защитной цены.
- Решения допускаются только на закрытых 4h-границах, полной сеткой full8.
  Сначала записываются все решения, затем создаются все intents, затем
  обрабатываются последующие книги. TTL строго меньше 4h: приказ не переносит
  устаревшее намерение в следующий цикл решения.
- Scaler/source manifest меняется по трёхдневным origins с purge 7 баров.
  **Позиции и деньги при смене origin не сбрасываются.** Новых fees или
  бесплатных fills на границе модели нет.
- Ни окончание файла, ни следующий origin не закрывают позицию принудительно.
  Остаток, pending/expired orders и stale marks сохраняются в результате.
- Риск учитывается на счёте и при новом fill в собственной доле. После общего
  risk latch новый риск блокируется; reduce-only закрытие планируется на
  следующей штатной 4h-границе, только при наличии свежих данных/ликвидности.
  Это **не intrabar аварийный риск-контроллер** и не гарантия maximum loss.

Виртуальные доли — правило учёта/размера позиции, **не биржевой isolated
margin**. Гэпы могут привести к отрицательному equity. Margin/liquidation
по-прежнему не квалифицированы; автоматических выдуманных ликвидаций нет.

## Идентичность и output

`configs/genetic_primus_portfolio_replay_v1.json` фиксирует семантику и SHA
трёх замороженных зависимостей. CLI сверяет эти файлы **до импорта** проектного
кода, затем проверяет транзитивные source pins исходного release contract.

Каждый source model manifest восстанавливается через `bind_model`: проверяются
scaler, cutoff, train/data/folds/resample/audit hashes и SHA реализации.
Поверх него создаётся отдельная role identity. У EMA/NoTrade нет фиктивных
GA weights или scaler в policy; их source manifest — только общий контекст
сравнения. На каждой decision остаются role/model SHA и execution binding.

Выход содержит:

- три independent execution ledgers и фактически сгенерированные приказы;
- cash/equity path после полного набора событий каждой временной метки;
- balances/equities по восьми долям, fees, funding, notional и fills;
- общий счётчик ненулевых EMA opportunities и отдельные target decisions;
- paired equity differences `GA−EMA` и `GA−NoTrade` на общем времени;
- SHA inputs, implementation, contracts, role models, paths и всего результата.

`fills` — исполнения, **не число независимых сделок**. Событийные equity marks,
разные символы и частичные fills не считаются независимыми наблюдениями для
LCB/ESS. CI, DSR/PBO, promotion gates и ranking победителя не вычисляются.

`within_replay_candidates_comparable=true`: агенты сравнимы **внутри нового
общего replay**. `historical_research_results_comparable=false`: результаты
нельзя напрямую присоединять к старым OHLCV next-open/terminal-close ретротестам.
При одинаковых фактических targets GA и EMA должны иметь одинаковую экономику;
это проверяется регрессией, а не гарантируется для разных решений.

## Формат и ограничения запуска

Bundle: `exia.genetic_primus.portfolio_bundle/1`.
Поля: `settings`, `rules` для всех 8 символов, `models`, `profile`, `events`,
`implementation_sha256`. Settings/rules — прежняя явная схема execution v1.

Profile содержит `protection_fraction`, `max_input_events` и
`feature_dataset_sha256`. `settings.dataset_sha256` — canonical SHA
`{rules, events}`; общий файл действительно проверяется, а не только его имя.
Внешний feature dataset SHA пока фиксирует заявленное происхождение: его
содержимое и causal feature extraction ещё требуют отдельной проверки.

Signals содержат только `id`, `kind=signal`, `ts`, `symbol`,
`feature_cutoff_ms=ts`, `features`. Признаки — четыре decimal strings либо
явные null. Входные приказы, per-agent proposal и будущий cutoff запрещены.
Все input IDs уникальны; точная повторная доставка идемпотентна, конфликт —
ошибка. Отсутствующий символ на decision boundary отклоняет весь сегмент,
а не приводит к неявному изменению весов.

Порядок общего входа: funding → marks → signals → books → clock при одном
`ts`; время не убывает. Signals/создаваемые intents обрабатываются по символам
в фиксированном порядке. Прочие same-time market events сохраняют входной
порядок. Risk/drawdown видят последовательные наблюдения; они не получают
будущие marks соседних событий.

Ограничения: input JSON ≤2 MiB, до 1024 market/signal events. Kernel budget
должен заранее покрывать эти events плюс все возможные generated intents.
При недостающих rules/fees, неправильном cutoff или исчерпании бюджета —
fail closed, без частичного output-файла с «успешным» статусом.

CLI без аргументов выполняет только проверку контрактов:

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/run_genetic_primus_portfolio_replay_v1.py
```

Прямые синтетические тесты — отдельными короткими командами с
`tools/check_genetic_primus_portfolio_replay_v1.py --group policy`,
`comparison`, `execution`, `accounting` или `safety`; pytest не нужен.

Создать **новый** пример и затем отдельно выполнить replay:

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/check_genetic_primus_portfolio_replay_v1.py --group accounting --demo Reports/Exia/Genetic_Primus/portfolio_replay_v1_20260910/example_new.json
```

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/run_genetic_primus_portfolio_replay_v1.py --input Reports/Exia/Genetic_Primus/portfolio_replay_v1_20260910/example_new.json --output Reports/Exia/Genetic_Primus/portfolio_replay_v1_20260910/example_result_new.json
```

Существующие артефакты не заменяются. Restart этого bounded controller —
повторение исходного immutable bundle с точным result SHA. Старый kernel
checkpoint **не** считается checkpoint всего portfolio controller: его
использование для нового слоя явно запрещено. Incremental/live recovery
отдельно не реализованы и не квалифицированы.

## Что ещё не доказано

Все цифры примера синтетические. Нового обучения, подбора objective или
финансовой валидации нет. Нет проверенного комплекта recorded book + mark +
funding + актуальные rules/fees; не квалифицированы latency, impact,
исполнимость отмен на бирже, feed completeness и causality генератора features.

Локальный funding manifest
`Retrodate/bitget_funding_history_v1_20260727/integrity_manifest.json` указывает
`recent_diagnostic_screen_only` и общее покрытие 2026-04-28 16:00 UTC —
2026-07-27 08:00 UTC для восьми инструментов. Поиск в `Retrodate` по именам
`*book*`, `*depth*`, `*mark*` не дал файлов; это ограниченная инвентаризация,
не доказательство отсутствия записей где-либо ещё. OHLCV не превращается в
придуманный стакан. CSV funding на этом этапе не исполнялся и не импортировался.

Следующий технический этап — read-only importer/validator предоставленного
согласованного recorded dataset и causal feature/scaler provenance, затем
один ограниченный replay GA/EMA/NoTrade. Только после калибровки исполнения —
зарегистрированный независимый performance audit. Все safety-флаги false;
допуск к paper/shadow/live этим инструментом не предоставляется.
