# Coverage audit регистрации forward-research

Аудит сопоставляет 21 незаполненное поле отдельного research execution contract
с 13 выбранными локальными source/metadata файлами. Это не полный анализ всей
истории репозитория, не чтение рыночных рядов и не измерение исполнения.
Не меняет существующие contracts, models, intervals или trading authority.

## Evidence discipline

`coverage.json` содержит для каждого поля status, source path, byte SHA, locator,
source value/units, reason и required action. Значение регистрации остаётся `null`,
auto-registration запрещена даже при наличии SHA пригодного старого файла.

Старый manifest можно проверить по SHA из latency config без чтения parquet.
Однако byte hash manifest не доказывает корректность underlying data, arrival,
revisions или принадлежность новой выборке. Метки `verified` в старых receipts
читаются как claims, а не становятся новой независимой аттестацией.

Status categories не складываются в score модели и не разрешают fit:

- `missing_registration`: нужное основание/политика не представлены выбранными источниками;
- `legacy_metadata_only`: старый manifest, universe или coverage end;
- `unverified_metadata_only`: timestamp/availability/funding claims;
- `engineering_definition_only`: формулы features либо accounting terminal rule;
- `incompatible_adapter`: snapshot IOC не заменяется OHLCV open sensitivity;
- `legacy_budget_only`: composite round-trip budgets не идентифицируют per-side fee/slippage;
- `legacy_diagnostic_only`: sensitivity не является measurement/calibration.

Не объявляется отсутствие файла во всём репозитории: выводы ограничены явно
перечисленными inspected sources. Изменённые baseline claims требуют нового
аудита; старый audit нельзя использовать для автоматического повышения статуса.

## Проверки

Только workspace Python `-X utf8`, одна короткая локальная команда за раз.
Сначала компиляция одного нового модуля, затем чистая cost-units функция.
Checker выполняет metadata-only regression, включая отсутствие auto-mapping,
подмены hashes, synthetic -> observed laundering и попытки открыть authority.
Source files читаются как текст; accounting/execution code не импортируется.
Нет pytest, collectors, сетевых вызовов, archive/parquet reads или fits.

Связывается predecessor receipt и его input/output bank, все inputs повторно
хешируются перед публикацией. JSON-артефакты создаются в новой директории
через exclusive create. `--verify` повторяет checks и сравнивает receipt/results,
не перезаписывая их. Markdown report создаётся после receipt и в его bank не входит.

## Следующий шаг должен закрывать содержание, а не добавлять mock stages

Нужен один preregistration decision record: отдельно research assumptions,
известные определения features/accounting и реальные неизвестные данные.
Он должен выбрать universe/product и доступный local dataset scope, а также
обоснованные timing/cost assumptions, не переименовывая их в measured evidence.
Для исполнимого release дополнительно нужны actual instrument/fee specs,
совместимые book/mark/funding данные и calibration. Их нельзя заменить
параметрами synthetic samples или SHA любого похожего файла.

Полнота registration не решает opportunities/ESS, causal retraining,
statistical budget, независимый audit и sealed-period qualification.
Все полномочия на обучение, сеть, paper/live и promotion остаются false.
