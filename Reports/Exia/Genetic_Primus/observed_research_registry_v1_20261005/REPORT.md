# Общий словарь метрик и наблюдаемый реестр исследования

5 октября 2026. Следующий инженерный этап выполнен **без нового обучения,
финансового evaluator, collector, pytest, сети и торгового runtime**.
Исходные результаты, предыдущие каталоги и контракты не изменены.

## Получено

- 14 определений метрик, 26 наблюдаемых aliases, все 233 нормализованных поля
  каталога. Финансовые значения не пересчитывались.
- 41 группа явных claims об objective ID, включая девять проверенных по байтам
  objective-registry snapshots. Это не 41 независимый алгоритм или trial.
- 24 ledger snapshots; 23 записанные previous-ledger ссылки совпали по байтам,
  циклов не обнаружено. Это подтверждает локальную связность записанных версий,
  но не полноту истории поиска.
- 65 типизированных наблюдений ledger. Сохранены источники, поля, роли, исходные
  objective/status/config claims и reported budgets.
- 15 result references подтверждены по байтам; 50 наблюдений не имеют result path.
  Получены 15 verified result-artifact groups, без конфликтующих ID в этих группах.
  Синтетические проверки доказывают дедупликацию повторов по байтам; фактические
  15 ссылок в этом наборе ведут к 15 разным result bytes.

| Категория наблюдения | Количество |
|---|---:|
| New declared attempts | 7 |
| Historically reported attempts | 4 |
| Diagnostics | 4 |
| Engineering checks / updates | 11 |
| Objective declarations | 33 |
| Registrations | 2 |
| Review / draft / deferred work | 4 |

Сумма 65 — число записей, **не** число statistical trials. Регистрации, reviews,
origins, поколения, model/scaler files и инженерные ревизии не считаются
дополнительными независимыми испытаниями автоматически.

## Обнаруженные ограничения

1. У 52 финансовых артефактов нет подтверждённой ledger-ссылки на result bytes.
   Они перечислены отдельно, а не превращены в вымышленные trials.
2. В обработанных attempt-записях нет явного `attempt_id`; сохраняются оригинальные
   записи и проверяемые result-artifact identities, а не придумывается исторический
   глобальный run ID.
3. Из contract claims четырём соответствует byte-verified config из контекстного
   слоя; остальные claims о configs не означают повторную проверку их содержимого.
   У одной objective-группы contract identity неизвестна. Несовпадающих contract
   hash claims под одним ID в этом наборе не найдено, но semantics не доказаны.
4. У 1774 артефактов нет собственного root `objective_id`. Многие из них — scalers,
   модели и вспомогательная метаинформация: это не 1774 ошибок или новых trials.
   Родительские связи нельзя заменять догадкой по имени файла.
5. Historical statistical trial count остаётся `null`; unknown adaptive decisions
   — `true`, global DSR/PBO и global immutable-registry verification — `false`.
   Целостная цепочка JSON не восстанавливает незаписанные решения отбора.

## Что исправлено в измерительной основе

Определения не смешивают trade/day LCB, marked/terminal return, виды drawdown,
trades/fills/transitions и fees относительно initial capital. Missing не становится
нулём. Config variants одного objective не объединяются в одну semantics.
Byte SHA и semantic SHA различаются; конфликтующий hash блокирует связь.
Reported budgets не суммируются в фиктивный знаменатель multiple-testing.

## Проверки и финансовый статус

55 детерминированных регрессий и независимая реконструкция outputs/receipt/
provenance прошли. Проверены дубли, несовместимые hashes/units, missing/zero,
objective variants и циклы ledger. Все 1784 оригинала и predecessor inputs,
включая замороженный future EMA contract, сохранили прежние байты.

Этот этап не улучшил и не пересчитывал прибыльность. Предыдущее первенство NoTrade
в сопоставимой сентябрьской группе не изменилось. Accounting по-прежнему неизвестен
у 42 evaluations из контекстного отчёта. Нового cross-source ranking, независимой
валидации, DSR/PBO-калибровки и допуска к торговле нет; все safety flags false.

## Следующий шаг

Закрыть accounting/valuation и provenance для небольшого явно выбранного
сопоставимого набора. Проверить capital/compounding/position-carry/terminal rules
по hash-bound реализации и контракту, а не по похожим именам метрик.
Новый training/GA objective не запускать на основании одной нормализации.
При отдельной будущей авторизации закреплять явные attempt IDs, contracts,
budgets и append-only launch/completion receipts до запуска, не задним числом.

Артефакты: `metric_dictionary.json`, `observed_registry.json`, `summary.json`,
`completion_receipt.json`. Контракт: `docs/GENETIC_PRIMUS_OBSERVED_RESEARCH_REGISTRY_V1.md`.
