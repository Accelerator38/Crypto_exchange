# Этап A — нормализация результатов, 5 октября 2026

Этап выполнен как **добавочный каталог**, без переписывания исторических JSON,
переобучения, collector, pytest, сети или торгового runtime.

## Получено

- 1784 исходных артефакта из фиксированного inventory; все хэши проверены до и
  после сборки. Отдельная проверка повторно сверяет оригиналы и completion receipt.
- Разделены результаты, origin steps, модели, scalers, manifests, регистрации,
  ledger, checkpoints, search traces, engineering checks и promotion gates.
- Поддержаны 41 источник, 90 evaluations, 233 явно определённых поля метрик.
  В этих полях нет отсутствующих значений; это **не означает** полноту контекста.
- 46 финансовых файлов имеют неподдержанную форму: метрики не угаданы.
  Ещё 253 артефакта оставлены с неизвестным типом.
- У 40 evaluations неизвестен точный период, у 50 — задержка, у 40 — costs.
  Горизонт и retrain cadence в текущих адаптерах неизвестны у всех 90: требуется
  проверенное присоединение hash-bound contracts/registrations, а не догадка.
- По 49 детерминированным регрессиям проверены единицы, missing vs zero,
  неверные значения, ambiguous JSON, типы, hash tampering и блокировка сравнения.

## Что предотвращено

LCB на сделку не смешивается с LCB на календарный день. Trades, fills и transitions
разделены. Drawdown по закрытиям баров не назван intrabar drawdown. Marked NAV и
гипотетическая итоговая ликвидация различаются. Августовская цепочка закрытых
origin windows не приравнивается к continuous NAV. Reported hash claims не
выдаются за проверенную lineage или цифровую подпись.

Разрешено лишь описательное сравнение внутри одного ранее проверенного
сентябрьского baseline economics audit, отдельно для normal и stress:

| Политика | Normal net, % | Stress net, % |
|---|---:|---:|
| NoTrade | 0.0000 | 0.0000 |
| GA_target_gate | -1.4301 | -1.8719 |
| GA_entry_only | -1.4801 | -1.8811 |
| EMA_long_only | -2.1519 | -2.6163 |
| EMA_TrendConsensus | -6.9576 | -7.7631 |

Период этого источника: 1–28 сентября 2026 UTC, конец исключён. Costs: normal
16 bps, stress 24 bps; это не новый reversal stress 40 bps + один бар задержки.
Сортировка не является exposure-matched attribution, paired significance test
или promotion gate. Лучший результат в этой сопоставимой группе — NoTrade.

## Следующий шаг

Добавить проверяемые sidecar joins для точных периодов, horizon/cadence, hashes
данных/folds/banks/model и execution semantics. Затем объединить определения
метрик и журнал попыток, сохраняя неизвестную часть adaptive-testing history.
Не открывать новый training/GA objective на основании одной этой нормализации.

Замороженный future EMA contract и continuous-NAV расчёт не изменены.
Никакого допуска к реальным деньгам или новой независимой валидации нет.

Файлы этой ревизии: `catalog.json`, `summary.json`, `completion_receipt.json`.
Первая metadata-сборка сохранена в `result_catalog_v1_20261005`; это история
инженерного уточнения каталога, не дополнительная торговая гипотеза.

Контракт и ограничения: `docs/GENETIC_PRIMUS_RESULT_CATALOG_V1.md`.
