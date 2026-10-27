# Итоги dependency readiness v1 — 6 октября 2026

Инженерная целостность ограниченного набора зависимостей подтверждена.
Готовность к новому рыночному расчёту и реальным деньгам не подтверждена.
Обучение и сравнение доходности моделей в этом этапе не выполнялись.

## Результаты

| Измерение | Результат |
|---|---:|
| Module entries в сохранённом snapshot | 612 |
| File-hashed / built-in-frozen / unresolved entries | 539 / 67 / 6 |
| Уникальные файлы | 1009 |
| Прочитано bytes | 59 053 144 |
| Source / cache / extension файлы | 470 / 468 / 68 |
| Interpreter binaries / core DLL candidate | 2 / 1 |
| Прежние искусственные outputs без изменений | 6 из 6 |
| Детерминированные assertions | 43, прошли |
| Исходные исторические JSON без изменений | 1784 |
| Новые model fits / market strategy evaluations | 0 / 0 |

Проверены сохранённые JSON, receipt, привязки исходников и SHA файлов. Проверка
артефактов не повторяла численный replay и не требовала совпадения module snapshot
разных процессов. 612 entries — ключи `sys.modules`, возможны aliases.

## Единый readiness record

| Gate | Статус | Ограничение |
|---|---|---|
| Artificial input/model payload binding | verified_scoped | Только три фиксированных искусственных model banks |
| Entry sidecar | verified_scoped | 648 строк, 12 закрытых сделок; не online |
| Сохранение accounting outputs | verified_scoped | Шесть зарегистрированных комбинаций |
| Observed dependency disk identity | verified_scoped | Не loaded-memory identity |
| Полный native/runtime closure | unverified | DLL, bytecode selection, generated globals, future imports |
| Dependency identity старых accounting runs | unverified | Новые хэши не аттестуют старые запуски |
| Full forward market contract integration | unverified | Инженерный контракт не заменяет замороженные |
| Реальные data/labels/scalers/folds/resamples/cutoff | not_evaluated | Рыночные данные не загружались |
| Causal training latency / online open positions | not_evaluated | Только искусственные закрытые интервалы |
| Новый objective / testing budget | not_authorized | Нового search или обучения не было |
| Statistical net after stress-costs | not_evaluated | Искусственный PnL не доказывает доходность модели |
| Fills/funding/borrow/intrabar risk | not_evaluated | Только offline cash ledger |
| Paper/shadow/fault injection/canary | not_authorized | Runtime и сеть не разрешены |

Итого: 4 ограниченно подтверждённых и 9 незакрытых gate. Никакого допуска по
числу пройденных проверок: `market_research_launch_ready=false`,
`real_money_ready=false`, `full_runtime_dependency_graph_verified=false`.

Disk SHA не доказывает, какие bytes уже находятся в памяти. Cache files
захэшированы как доступные файлы, без доказательства их использования;
Python core DLL — candidate. Есть 6 unresolved module entries. Native DLL
граф, builtin/frozen memory, extension memory и будущие imports не аттестованы.
Ephemeral runpy sources связаны receipt отдельно, не полным module snapshot.

## Вывод и дальнейшие действия

Этот этап не меняет рейтинг торговых моделей и не доказывает преимущество над
NoTrade. Все flags credentials/orders/execution/paper/live/network/promotion/
training/financial_evaluation оставлены false. Collector, pytest, сеть и
торговый runtime не запускались. Commit/push не выполнялись.

Следующий шаг: подготовить отдельный forward research contract proposal и
проаудировать data/label/scaler lineage до нового обучения. Зафиксировать
причинность, execution assumptions, сравнения и testing budget; не переиспользовать
sealed interval автоматически. Цель пользователя — неубыточный итог после costs
при допустимых просадках; это требует независимого статистического подтверждения,
а не дополнительных успешных искусственных сценариев.

Методика: `docs/GENETIC_PRIMUS_DEPENDENCY_READINESS_V1.md`.
Машинные результаты: `summary.json`, `readiness.json`,
`dependency_fingerprint.json`, `environment.json`, `completion_receipt.json`.
Этот Markdown — пояснение; проверенная публикация и её receipt состоят из JSON.
