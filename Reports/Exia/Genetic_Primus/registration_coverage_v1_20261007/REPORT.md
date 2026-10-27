# Genetic Primus: покрытие timing / execution / data регистрации

Дата: 7 октября 2026. Выполнено сопоставление **всех 21 полей** research execution
contract с **13 выбранными локальными source/metadata файлами**.
Полные значения, единицы, locators и byte SHA находятся в
[coverage.json](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/registration_coverage_v1_20261007/coverage.json).

Это scoped audit, не полный поиск по истории репозитория и не проверка самих
архивов/parquet. Старые metadata читаются как заявления соответствующих receipts.
Значения контрактов не изменены; автоматическая регистрация запрещена.

## Главное

**0 из 21 поля зарегистрированы.** У 15 полей найдены частичные, инженерные или
несовместимые источники; у 6 нет нужной регистрации в inspected sources.
Это не 15 подтверждённых доказательств и не новый торговый результат.

Найдены четыре существенных ограничения:

1. Старый OHLCV manifest содержит file hashes, full8 universe и coverage
   **2022-01-01 → 2026-09-28 UTC exclusive**, но columns только
   timestamp/symbol/OHLCV. Arrival/revision semantics этим manifest не доказаны.
   Его файл совпадает с SHA, закреплённым в старом latency config; сами parquet
   здесь не читались и не хешировались заново.
2. Recorded-admission package имеет `evidence_kind=synthetic`. Его успешные
   admission tests не являются доказательством пригодности реального market data.
3. BTC derivative ticker за 2024-12-01 имеет complete coverage receipt с
   заявленными 163447 rows и проверкой gzip CRC. Но source authenticity,
   content semantics и funding settlements в receipts остаются **false**.
   Присутствие funding timestamps/rates не доказывает фактические settlements.
4. Последний **inspected** L2 receipt `l2_coverage_009.json` имеет
   `complete=false`; заявленный exchange range заканчивается
   2024-12-01 10:48:54.571 +03:00. Он не подтверждает полный день или пригодность
   order book replay. Другие возможные источники вне curated bank не исключаются.

## Покрытие 21 поля

| Поле | Что найдено / почему недостаточно |
| --- | --- |
| timing.training_budget_ms | Не выбран; proposal содержит null |
| timing.execution_delay_ms | Старые 0/1 bar sensitivity; не measured latency, 1 bar вне нового delay < bar scope |
| timing.budget_evidence_sha256 | Нет основания зарегистрированного timing budget |
| data.manifest_sha256 | Старый immutable manifest другого контракта; не manifest новой выборки |
| data.timestamp_spec_sha256 | Columns/timeframe и receive-time правило IOC; dataset-specific semantics не подтверждены |
| data.availability_evidence_sha256 | Нет arrival в OHLCV columns; один BTC ticker day не покрывает full8 train |
| data.revision_policy_sha256 | Fail-closed правило есть; dataset-specific revision policy не зарегистрирована |
| data.universe_sha256 | Старый full8 и futures symbol map; новый instrument/universe basis не выбран |
| data.feature_definition_sha256 | Есть проверенные synthetic формулы; actual prefix/availability ещё не привязаны |
| data.training_prefix_manifest_sha256 | Нет actual per-origin training-prefix manifest; archive SHA включает post-train данные |
| data.snapshot_cutoff_ms | Старый coverage end не является новым availability cutoff |
| costs.adapter_sha256 | Snapshot IOC source существует, но несовместим с OHLCV open sensitivity |
| costs.normal_fee_bps_per_side | Composite round-trip budgets; fee schedule evidence null |
| costs.normal_slippage_bps_per_side | Composite round-trip budgets не выделяют slippage |
| costs.stress_fee_bps_per_side | Composite stress scenarios не выделяют fee |
| costs.stress_slippage_bps_per_side | Нет зарегистрированного per-side stress slippage основания |
| costs.calibration_evidence_sha256 | Старый latency experiment без model refits, calibration evidence null |
| costs.funding_evidence_sha256 | Ticker metadata есть; authenticity/settlements не подтверждены |
| costs.borrow_basis_sha256 | Applicability/N/A нового instrument не зарегистрирована |
| costs.terminal_policy_sha256 | Accounting code задаёт interval-end close; новые terminal time/price availability не привязаны |
| costs.instrument_spec_sha256 | Старый Bitget USDT linear perpetual target; instrument_rules evidence null |

Разбивка evidence statuses: missing registration — 6; legacy metadata — 3;
unverified metadata — 3; engineering definitions — 2; incompatible adapter — 1;
legacy composite cost budgets — 4; legacy diagnostics — 2. Всего 21.

## Почему нельзя автоматически заполнить контракт

- **16/24/40 bps — aggregate round-trip scenarios**, а не отдельно fee и
  slippage per side. Деление общей суммы на 2 не определяет эти составляющие.
- Задержка одного 4h бара — conservative sensitivity, не измерение latency;
  старый эксперимент не переобучал коэффициенты.
- OHLCV planned open не становится fresh snapshot IOC fill. Старый release
  contract прямо запрещает OHLCV fallback и требует mark/funding.
- SHA инженерного кода или metadata файла не доказывает реальные inputs,
  settlement semantics, source authenticity или existence успешного fit.
- Формулы return12/efficiency12 готовы, но synthetic feature availability
  обозначает declared decision-asof, а не measured compute availability.
- Source terminal rule полезен, но не обосновывает цену/availability будущего
  global-end liquidation. Нельзя закрывать позицию на каждом retrain ради labels.

Важно разделить **декларацию исследовательских assumptions** и **подтверждение
реалистичного исполнения**. Первая позволяет определить будущий research protocol,
но сама по себе не создаёт вторую. Требование доказанного live-fill качества не
следует заменять dummy values; исследовательскую sensitivity не следует ошибочно
выдавать за failed или passed release qualification.

## Проверки и сохранность

Последовательно выполнены compile одного нового модуля, прямой pure cost-units
check, compile checker и его metadata regression:

- **81 assertions passed**;
- **81 source/metadata/doc inputs неизменны** до/после, включая predecessor bank;
- повторный `--verify` подтвердил receipt и два JSON output;
- checks отвергают hash substitution, synthetic -> observed laundering,
  изменение baseline funding/complete claims и попытки открыть training authority.

Source accounting/execution читался как текст и не импортировался. Нет pytest,
collector, сети, raw archive/parquet reads, финансовых evaluation или fits.
Никакие старые outputs/contracts не перезаписывались. Git commit/push не выполнялся.
Report создан после receipt и не входит в input bank.

## Следующий содержательный шаг

Подготовить **один preregistration decision record**, а не очередной toy stage:

1. Зафиксировать предложенный research scope: universe/product, доступную
   local history и intended cutoff, без чтения нового sealed interval.
   Старый full8/Bitget mapping — кандидат для обсуждения, не автоматически выбранный
   новый contract. Не смешивать spot-like OHLCV notation и perpetual funding basis.
2. Для timing и costs отдельно указать planning/sensitivity assumptions,
   единицы, основание и что ещё не измерено. Не присваивать старые total budgets
   новым fee/slippage fields; не считать неизвестный funding/borrow нулевым.
3. Переиспользовать определения features/accounting только через новую versioned
   регистрацию с явной семантикой. Actual prefix, revisions, label availability и
   instrument/fee/settlement evidence закрывать отдельно, когда доступны данные
   и соответствующая авторизация работы с ними.
4. После регистрации, достаточных opportunities/ESS и отдельной авторизации
   ограниченного fit — один causal origin, затем заранее определённые uniform и
   half-life90d варианты. До этого не запускать GA и не повторять январскую настройку.

Предыдущие pre-fit sample блокеры не изменились: synthetic fixture содержит 7
opportunities вместо minimum 64, Kish ESS около 7 вместо 40. Это не оценка размера
реального multi-year dataset: последний здесь не исследовался.

Конечная цель — неубыточный net после costs при допустимых просадках. Этот аудит
не улучшил доходность модели и не доказал превосходство над NoTrade. Все training,
network, paper/live, execution и promotion полномочия остаются false.
