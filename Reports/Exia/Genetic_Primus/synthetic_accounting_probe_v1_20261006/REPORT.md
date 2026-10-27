# Проверка бухгалтерского исполнения: результаты

6 октября 2026 года.

## Выполнено

Впервые в этой последовательности после metadata/stub аудитов проверена сама текущая pinned accounting-функция на маленьких artificial pandas DataFrames. Original source SHA подтверждён; package initializer и accounting `.pyc` обойдены. Source/runtime/model файлы не изменены.

Восемь фиксированных cases: carry/retarget/NoTrade в двух reference cost scenarios, минимальный 18-bar NoTrade и forced-open-reset negative control. Рыночные данные, обучение и подбор objective не использовались.

## Результат

- 17 356 oracle comparisons согласованы; maximum absolute error **4.44e-16** при tolerance **1e-10**.
- 129 дополнительных deterministic regression assertions прошли.
- Carry сохранил quantity; cash/NAV, fees и realized/unrealized PnL по snapshot phases совпали с rational oracle. Нет retrain-induced выходов при неизменном target.
- Закрытые trades сохранили entry-model SHA при смене активной модели решения.
- Final-close liquidation и actual-notional fees согласованы с oracle; после terminal cash равен equity.
- NoTrade: equity 1, ноль fills/trades/fees/drawdown.
- Negative control доказал open-price exit для `force_flat_timestamps`, не previous-close liquidation.
- Независимая сверка подтвердила receipt, source/environment/output hashes. 1784 исторических JSON и прежние контракты не изменены.

Количество сравнения относится к строкам/полям этих же восьми fixtures, не независимым экспериментам. Toy PnL не является результатом реального агента; ранжирование старых моделей не изменилось.

## Ограничения и найденный пробел

В actual decision ledger нет SHA модели входа позиции: он присутствует только в закрытых trades. Derived oracle attribution записана отдельно и не выдаётся за исходное поле. Полная forward contract integration остаётся **неподтверждённой**.

Идентичность именно исполненного accounting source проверена, но NumPy/pandas versions и on-disk init hashes не закрывают compiled/transitive dependency graph. Historical runs не получают это подтверждение задним числом. Intrabar риски, funding/borrow и realistic exchange execution не доказаны.

При dependency import guard заблокировал local hostname metadata request. Первоначальный assert «нет blocked events» был уточнён; запрос остался заблокированным, никаких соединений/процессов не разрешено. Guard не заменяет OS sandbox.

Новых market strategy evaluations и model fits — 0. Credentials/orders/paper/live/execution/promotion выключены. Collector, pytest, supervisor и торговый runtime не запускались. Положительный результат unit-проверки не доказывает прибыльность или допуск к реальным деньгам.

## Дальше

Реализовать отдельный forward wrapper со строгой input/model binding и entry-model identity sidecar для каждого решения. Проверить на уже зафиксированных synthetic fixtures без открытия нового периода. Затем закрывать оставшиеся dependency/launch gates; не начинать новые обучения на прежнем отклонённом objective.

Техническое описание: `docs/GENETIC_PRIMUS_SYNTHETIC_ACCOUNTING_PROBE_V1.md`. Машиночитаемые результаты рядом: `fixtures.json`, `environment.json`, `summary.json`, `completion_receipt.json`.
