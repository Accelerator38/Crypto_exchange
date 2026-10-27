# Следующий этап: связи результатов с контрактами

5 октября 2026. Выполнен отдельный metadata-only слой поверх каталога этапа A.
Обучение, collector, pytest, сеть, торговый runtime и анализ будущего интервала
не запускались. Исходные результаты и финансовые метрики не изменены.

## Результат

Связаны 23 источника и 67 из 90 evaluations: reversal calibration — 42,
rolling linear matrix — 15, сентябрьский baseline economics — 10.

| Поле | Неизвестно до | Неизвестно после |
|---|---:|---:|
| Начало/исключённый конец периода | 40 | 23 |
| Deployment horizon | 90 | 33 |
| Retrain cadence | 90 | 36* |
| Round-trip costs | 40 | 23 |
| Delay bars | 50 | 48 |
| Схема accounting | 42 | 42 |

\* В этих 36 три fixed-EMA evaluations имеют cadence «не применимо»; остальные
33 остаются неизвестными. Отсутствующий delay не считается нулевым.

Reversal и обучение линейной матрицы: **1 июня — 31 июля 2026 UTC, конец
исключён**, 60 дней / 20 трёхдневных origins. Это уже открытый development,
не новый sealed test. Для reversal подтверждены horizon/cadence 3 дня, purge 7,
declared rolling window 730 дней, normal 16 bps / delay 0, stress 40 bps / delay 1.
Для линейных arm подтверждены трёхдневные model origins и cutoffs; stress там
**24 bps**, не reversal 40. Задержка линейной матрицы остаётся неизвестной.

Сентябрьский baseline economics: **1–28 сентября 2026 UTC, конец исключён**,
27 дней, normal 16 / stress 24 bps. Его горизонт/cadence данным слоем не доказаны.

## Проверки

- 53 детерминированные регрессии: units/types, UTC, semantic vs byte SHA,
  duplicate JSON, unknown vs zero, conflicting context, нарушенная цепочка,
  подменённые model SHA, cutoff и stress delay.
- Проверены семантические хэши регистрации, folds, итогового reversal result,
  20 steps/states и 100 model payloads.
- В reversal совпадают 160 явных per-decision model SHA. В линейной матрице
  проверены 80 parent decision batches: индивидуальных model SHA в их decision
  items нет, и такая проверка не заявляется.
- Manifest joins сверены по байтам. Raw market data, scaler/weights и bootstrap
  bank payloads не перепроверялись; соответствующие SHA остаются reported claims.
- Независимая реконструкция слоя подтвердила outputs/receipt/provenance и
  неизменность базовых метрик. Повторно проверены все 1784 оригинала, старый
  каталог и замороженный future EMA contract.

## Что это значит для модели

Прибыльность не улучшилась и не пересчитывалась. Допуск к торговле не изменился.
Предыдущее описательное первенство NoTrade в сопоставимой сентябрьской группе
сохраняется. Связанные contracts не доказывают правильность исторического fit,
raw-data lineage, funding/real fills или независимость валидации.

Общая автоматическая рейтинговая таблица разных запусков по-прежнему блокируется:
accounting неизвестен у 42 evaluations, сохраняются gaps по данным, costs/delay,
экспозиции и adaptive-testing history. Новых торговых гипотез или обучений этот
metadata этап не добавляет.

## Дальше

Объединить определения метрик и идентичности objective/attempt, отделив
инженерные проверки от финансовых trials. Закрывать accounting/valuation и
provenance gaps с проверяемыми ссылками, не угадывая недостающую историю.
Только после этого обсуждать корректный общий ranking и новый research objective.

Артефакты: `context_links.json`, `summary.json`, `completion_receipt.json`.
Подробный контракт: `docs/GENETIC_PRIMUS_RESULT_CONTEXT_LINKS_V1.md`.
