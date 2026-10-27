# V8 Clean — единый preflight перед historical smoke

Дата: 2026-10-07. Результат: **NO-GO для fit и рыночной финансовой оценки**.
Metadata preflight завершён; причина — незарегистрированный engineering-протокол
и неподтверждённая интеграция нового контракта с market pipeline, не отсутствие
нового sealed интервала.

## Подтверждено

`--run` и отдельный `--verify` прошли; receipt/result совпадают,
**176 source/metadata identities** неизменны. Проверены только metadata и
исходники. Новых synthetic experiments=0, parquet reads=0, scaler/model fit=0,
market financial evaluation=0. Девять safety flags false. Сеть, collector,
pytest, фоновые процессы и runtime не запускались.

Существующая цепочка подтверждает:

- качество 11 разрешённых исторических OHLCV-файлов, 83 088 строк без выявленных
  нарушений полей, плюс ранее проверенную timestamp/symbol сетку;
- синтетический origin/prefix adapter: 739 assertions;
- синтетические завершённые labels и X/y manifest: 776 assertions;
- синтетический net-accounting adapter: 124 assertions.

Эти counts относятся к разным локальным наборам. Они не означают число
независимых экспериментов, statistical significance или готовность модели.
Оригинальное arrival evidence и подлинность источника ими не доказаны.

## Требования разделены по назначению

В старом immutable preregistration остаются **30 требований**:

- **22 относятся к протоколу historical smoke:** 3 timing, 8 data, 10 costs
  плюс executable threshold rule. Все 21 поля основного timing/data/cost
  registration сейчас null; threshold также не зарегистрирован.
- **8 относятся к prospective/promotion:** drawdown budget, concentration/
  turnover/drift gates, confidence/multiplicity, две новые sealed-даты,
  folds/resample/audit banks, global trial count, power/block design.

Восемь последних **не нужны как предварительное условие не-доказательного
инженерного smoke**. Но они остаются обязательствами для соответствующих
statistical/prospective/promotion этапов. Смешивание этих назначений могло
создать бесконечный барьер перед инженерной проверкой.

Отдельно draft availability-контракт оставляет null для ingestion lag,
arrival evidence и revision policy. Это перекрывающиеся требования к
availability, а не ещё три независимых статистических теста.

Для engineering sensitivity допустимы заранее объявленные допущения вместо
измерений, если это **отдельный immutable протокол** и результат нельзя
использовать для доказательства edge или promotion. Допущения не становятся
измеренной задержкой/комиссией по факту их утверждения.

## Основная техническая граница

Не все оставшиеся проблемы решаются заполнением чисел:

| Интеграция | Что есть в проверенной цепочке | Чего она ещё не подтверждает |
| --- | --- | --- |
| Market feature/label matrix | SYN-only builder, SHA и timing gates | Full8 market matrix под новой closed-row семантикой |
| Price → cost adapter | Проверенная synthetic-mid арифметика | Основание считать OHLCV-open mid/fill и применять конкретные cost assumptions |
| Continuous portfolio ledger | Арифметика одной позиции | Сквозные cash/quantity, marks, carry и единая terminal policy |
| Causal fit/model lineage | Scaler-input и X/y manifests | Реально fitted scaler/model на каждом origin и model SHA каждого решения |

Статус всех четырёх — **not demonstrated in current audited pipeline**.
Это не утверждение об отсутствии legacy-реализаций во всём репозитории.
Существующий offline обучающий код следует адаптировать/перепроверить, а не
автоматически объявить совместимым с новым target и accounting.

OHLCV-open не доказан как mid; синтетический cost adapter прямо отклоняет
этот price kind. Поэтому нельзя просто вычесть из старых gross targets
16/24/40 bps и назвать результат проверенным net objective. Эти budgets
агрегированные; fee/slippage по сторонам ими не идентифицированы.

## Зафиксированный объём следующей работы

Старый план не изменён: один **исторический инженерный** origin
2026-09-22 → номинально 2026-09-25, shifted exit может требовать данные до
2026-09-25 08:00 UTC. Это уже исторический период, не prospective.

Кандидаты: linear overlay, `return_12` + `efficiency_12`, 3 коэффициента,
без hidden layer/GA; только uniform и half-life90 с anchor20. Максимум
2 planned pipeline runs, включая failures. **Authorized market runs under
current record=0**; это не заявление об отсутствии прошлых попыток в репозитории.
Целевой экономический floor остаётся 0 после stress costs относительно NoTrade;
гарантия отсутствия убытков не даётся.

Практическая последовательность:

1. Отдельно зафиксировать engineering protocol: price semantics, ingestion/
   execution delay, training budget, revision policy, явные cost/funding/borrow
   assumptions, decision rule, terminal policy, manifest/cutoffs и бюджет попыток.
   Старые frozen записи не редактировать. Нельзя выдавать предположения за calibration.
2. Адаптировать существующий **offline market path** под новый origin/prefix/label
   contract и сквозной ledger. Начать с data/feature adapter; не создавать новый
   search, дополнительные seeds или ещё одну исследовательскую матрицу.
3. Проверить реальные входные hashes/схему и интеграционные границы в пределах
   явно разрешённого исторического набора; не открывать новый sealed период.
4. Только после регистрации протокола и отдельного разрешения fit выполнить
   один ограниченный historical smoke. По нему не выбирать лучшую модель и
   не делать заявлений о превосходстве над NoTrade.
5. Statistical promotion/prospective готовить отдельным этапом после исправной
   market-интеграции, а не требовать их до инженерной проверки.

Следующий содержательный шаг — market data/feature adapter и protocol draft,
а не увеличение числа локальных synthetic experiments. Если нужен иной
price/cost/sizing objective, это отдельная версия контракта, не тихая подмена.

## Артефакты

`result.json`: полный список 30 требований, разделение scopes, availability,
четыре integration obligations, frozen scope и go/no-go.
`completion_receipt.json`: identities и safety. Этот Markdown вне receipt.
Добавлены только preflight helper/checker/documentation; предыдущие файлы
не менялись. Публикация, commit/push и реальные/бумажные торги не выполнялись.
