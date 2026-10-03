# Future EMA: реализация финального offline evaluator

Дата: 2026-10-03. Реализация: локальный commit `8b455a4`.

## Итог

Реализован воспроизводимый расчёт зарегистрированного EMA baseline против NoTrade. Это инженерный этап, не новое обучение и не оценка доходности на реальном будущем периоде. Параметры зарегистрированного эксперимента не изменены; GA не запускался.

Контракт: `configs/genetic_primus_future_ema_viability_v1.json`, SHA256 `11b83d2e3ecbdda572307dd902b73c9c2d8ae108499be85e22b988f193b227b6`.

Фиксация реализации: `configs/genetic_primus_future_ema_implementation_v1.json`, SHA256 `654477fad14c36bef92ef3ea584bb759f3243d98be9fe1872d13fbbba5b3832a`. Закреплены SHA evaluator, runner и проверок происхождения данных. Основные зависимости уже закреплены исходным контрактом.

## Что добавлено

- Строгая проверка полного 4h-панеля восьми активов: сетка времени, дубликаты, пропуски, конечные значения и согласованность OHLC.
- Причинные решения по завершённым барам; ledger с SHA политики и временем доступности признаков. Позиции переносятся между трёхдневными origins.
- Два сценария: 16 bps без дополнительной задержки и 40 bps с задержкой на полный 4h-бар.
- Зарегистрированные primary/audit block-bootstrap banks, 60 origins, шесть 30-дневных slices, drawdown и концентрация положительного результата. Минимум — 20 разных origins с предложением сделки, а не 20 сделок.
- Команда фиксирует SHA полного входного сегмента. Перед итоговым расчётом этот manifest должен быть неизменным файлом в HEAD. Перед анализом создаётся exclusive reservation; ошибка также сохраняет reservation и не разрешает автоматический повтор.
- До конца интервала runner останавливается до чтения рыночных файлов. Никакого автоматического запуска после наступления даты нет.

## Проверки

`py_compile` отдельно для evaluator и runner: успешно.

Прямой запуск `tests/test_genetic_primus_future_ema_evaluator_v1.py`: **5 tests, OK**, 1.538 s. Проверены дефекты сетки/OHLC, независимость решений от будущих цен, ранний запрет анализа, концентрация в единственном успешном origin и полный синтетический pipeline. Синтетический успешный gate не является доказательством торгового преимущества.

Runner `preflight`: `REGISTERED_AWAITING_NEW_OFFLINE_DATA`.

Runner `evaluate` от 2026-10-03: `PENDING_INTERVAL_COMPLETE`, `market_files_opened=false`, `outcome_metrics_computed=false`. Reservation не создан.

## Финальный offline workflow

Период оценки остаётся **2026-10-05 00:00 UTC — 2027-04-03 00:00 UTC**, 60 трёхдневных origins. Полный добавляемый сегмент начинается 2026-09-28, включает семь дней прогрева и должен содержать 8976 строк. Не использовать этот период для настройки challenger.

После 2027-04-03 и получения полного локального сегмента:

1. Проверить `preflight` с закреплённой реализацией.
2. Выполнить `E:\Crypto_exchange\.venv\Scripts\python.exe tools/run_genetic_primus_future_ema_viability_v1.py commit --segment <workspace-relative.parquet>`.
3. Проверить и закоммитить только созданный `Reports/Exia/Genetic_Primus/future_ema_viability_v1_20261005/data_manifest.json`, сохранив все пользовательские изменения.
4. Выполнить `E:\Crypto_exchange\.venv\Scripts\python.exe tools/run_genetic_primus_future_ema_viability_v1.py evaluate` один раз. Не удалять reservation ради повторного анализа.

## Границы результата

Реальных новых PnL нет; прежние отрицательные оценки и запреты GA/promotion остаются в силе. Даже успешный future gate разрешает лишь отдельное позднейшее зарегистрированное исследование, не реальные торги. OHLCV delay/cost sensitivity не подтверждает биржевые fills: funding и синхронные tapes не проверены, глобальные DSR/PBO не откалиброваны. Все credentials/orders/paper/live/promotion flags остаются false.
