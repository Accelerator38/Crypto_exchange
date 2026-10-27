# Genetic Primus: boundary stub adapter v1

Дата: 2026-10-05. Продолжение `GENETIC_PRIMUS_BOUNDARY_ENGINEERING_V1.md` без изменения предыдущего контракта.

## Завершённая область

Проверен call protocol нового инженерного adapter: явная передача параметров и построение synthetic decision ledger с двумя model identities. Это **не интеграция с реальным evaluator** и не финансовый эксперимент. Запрещены произвольные callees и данные: adapter принимает только точный тип встроенной `RecordingStub` и точные зарегистрированные synthetic payloads.

Новый модуль `src/exia/genetic_primus/boundary_stub_adapter_v1.py` импортирует только stdlib `copy` и `math`. API предыдущего pure boundary contract передаётся явно через `runpy`, без импорта research package `__init__`. Существующие model/evaluator/collector/runtime modules читаются как bytes/AST, но не исполняются; pure metadata builders предыдущих этапов используются для сверки provenance.

## Явные параметры

По статической сигнатуре текущего `simulate_cash_targets` подтверждены девять имён: `bars`, `targets`, `symbols`, `start_timestamp`, `end_timestamp`, `timeframe_ms`, `round_trip_cost_bps`, `force_flat_timestamps`, `record_details`. Adapter задаёт каждый keyword явно. Universe и пустой force-flat передаются tuple, `record_details=true`; start/end/timeframe/cost задаются из bound engineering plan.

Заглушка записывает значения, размер и semantic hashes входов. Она не рассчитывает NAV, quantity, fees, trades, fills, returns или drawdown. Переданные `bars`/`targets` — synthetic списки; пригодность pandas DataFrame adapter и фактическая совместимость с simulator не подтверждены совпадением имён параметров.

## Fixtures и ledger

Три случая: carry (неизменные противоположные направления для двух artificial symbols), retarget (направление → flat → противоположное направление), abstain (NoTrade). Каждый проверен при reference normal 16 и stress 24 bps: всего шесть комбинаций. Все payloads создаются кодом, не загружаются из market data или обученных models.

В каждом случае три полных 3-дневных origins, свечи 4 часа, две synthetic symbols, 108 target rows. Всего опубликовано 648 synthetic decision records и 12 terminal expectations. Origin open намеренно отличается от previous close: эти границы/цены нельзя отождествлять.

Для каждого решения:

- `decision_model_sha256` соответствует synthetic payload активной модели этого origin.
- `position_entry_model_sha256` сохраняется при carry; меняется при новой позиции и становится null при flat.
- Retrain не порождает принудительный выход или resize. Target change описывается exit/enter semantic events, не настоящими fills.
- Terminal содержит final close timestamp/ожидаемую цену/ожидаемый выход, но `actual_fill_verified=false`. Fees при выходе требуются декларативно, не рассчитаны.

Ledger проверен отдельной элементарной state-machine oracle, которая не вызывает transition function контракта. Инварианты модели и terminal rules проверяются на всех строках. Это не проверка quantity/cash carry или экономической доходности.

## Статический source bank

Пять scoped source fingerprints: два pure fixture modules и три ранее рассмотренных accounting/helper reference files. Для каждого сохранены byte SHA, размер и AST import facts. Явная injection edge adapter → boundary specification зарегистрирована отдельно.

Этот bank **не полный transitive dependency graph**. Импорты reference files, dynamic imports, external distribution versions, package initializers и actual loaded-runtime identity не считаются разрешёнными/подтверждёнными. Перед `runpy` нового adapter AST проверяется на незарегистрированные imports; predecessor code сверяется с receipt до загрузки.

`completion_receipt.json` имеет тип `synthetic_adapter_only_not_financial_launch`. Связывает predecessor receipt, input hashes, builder/adapter source bytes и output hashes. Наблюдаемые Python path/version не заменяют полный environment fingerprint. Старые accounting gaps остаются неизвестными.

## Выполненные проверки

По одному новому файлу выполнен py_compile, затем pure fixture check, preflight, **4145 детерминированных checks**, публикация и отдельный artifact checker. Число checks включает повторяемые построчные assertions на шести фиксированных комбинациях, не независимые эксперименты и не statistical testing budget.

Проверены: explicit parameters, input immutability, deterministic repeat, payload/active/entry SHA, carry/retarget/NoTrade, final terminal expectations, late/malformed/duplicate/missing data rejection до stub call, unknown scenario rejection, повторное использование stub и запрет произвольного evaluator/subclass. В процессе regression runner не импортировались exia/numpy/pandas/websockets/requests.

Только `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8`, команды отдельно:

- `tools/check_genetic_primus_boundary_stub_adapter_v1.py`.
- `tools/build_genetic_primus_boundary_stub_adapter_v1.py --verify-only`.
- Builder без флага публикует в новый exclusive каталог, не перезаписывая существующий.
- `tools/check_genetic_primus_boundary_stub_adapter_artifacts_v1.py`.

Результаты: `Reports/Exia/Genetic_Primus/boundary_stub_adapter_v1_20261005/{fixtures.json,source_bank.json,summary.json,completion_receipt.json}`. Markdown-документация не входит в hashed output bank.

## Оставшиеся gates

`protocol_stub_verified=true`; `real_evaluator_adapter_verified=false`, `runtime_dependency_graph_verified=false`. Все flags credentials/orders/paper/live/execution/promotion/training/financial evaluation/network остаются false. Новых fits и финансовых evaluations — 0. 1784 исторических JSON и старые контракты не изменены.

Следующая содержательная проверка должна касаться бухгалтерского исполнения на минимальных synthetic fixtures: DataFrame schema, сохранение quantity/cash, fees и final-close liquidation, с отдельным oracle. Нельзя считать их доказанными по записанному stub call. До допуска фактического callee нужны проверка его изолированности и hashes зависимостей; новый финансовый расчёт по market data остаётся отдельным решением. Frozen/prospective контракт и testing budget этим этапом не переопределяются.
