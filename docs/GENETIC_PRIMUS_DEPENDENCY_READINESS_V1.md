# Genetic Primus: observed dependency readiness v1

Дата: 2026-10-06. Только локальная инженерная проверка, без рыночных данных,
обучения, collector, сети и торгового runtime.

## Назначение

Этот этап связывает результаты шести ранее зарегистрированных искусственных
сценариев с ограниченным snapshot файлов наблюдаемых зависимостей и собирает
единый fail-closed readiness record. Он не меняет objective, исторические
результаты, замороженные контракты или допуск к реальным деньгам.

## Реализация

- `src/exia/genetic_primus/observed_dependency_readiness_v1.py`: ограниченный
  сбор, потоковые SHA-256, валидация fingerprint и derivation readiness.
- `tools/build_genetic_primus_dependency_readiness_v1.py`: проверка provenance,
  повтор шести фиксированных искусственных комбинаций, snapshot, эксклюзивная
  публикация JSON и completion receipt.
- `tools/check_genetic_primus_dependency_readiness_v1.py`: положительные и
  отрицательные детерминированные проверки, включая fail-closed условия.
- `tools/check_genetic_primus_dependency_readiness_artifacts_v1.py`: проверка
  сохранённого snapshot, файлов и receipt без численного replay.

Лимиты: 1600 module entries, 1200 файлов, 128 MiB на файл, 384 MiB суммарно;
хэширование блоками 512 KiB. Проверяются scope путей и неизменность размера и
mtime во время чтения. Лимиты не являются доказательством OS sandbox.

## Проверенный результат

В `Reports/Exia/Genetic_Primus/dependency_readiness_v1_20261006` опубликованы
`dependency_fingerprint.json`, `readiness.json`, `environment.json`,
`summary.json`, `completion_receipt.json`.

- 612 module entries: 539 file-hashed, 67 built-in/frozen, 6 unresolved.
  Entries означают ключи `sys.modules`, а не обязательно уникальные модули.
- 1009 уникальных файлов, 59 053 144 bytes (около 56,32 MiB).
- 470 Python sources, 468 cache files, 68 extension binaries,
  2 interpreter binaries и 1 Python core DLL candidate.
- Шесть искусственных outputs совпали с предыдущим этапом.
- 43 детерминированные проверки прошли; это не финансовые испытания.
- Сохранённые артефакты независимо проверены; 1784 исходных исторических JSON
  остались неизменными.

## Границы доказательства

Disk SHA не доказывает идентичность уже загруженной памяти. Наличие и SHA
`.pyc` не доказывают, что процесс использовал именно этот cache. Core DLL
указана как candidate, а не как аттестованный загруженный бинарник.
Не закрыты native DLL dependencies, встроенный/frozen код в памяти,
выбор bytecode, память extensions, будущие dynamic imports и ephemeral
runpy namespaces. Для последних исходные файлы отдельно связаны receipt,
но это не превращает snapshot в полный runtime dependency graph.

Artifact checker проверяет сохранённый snapshot, а не предполагает равенство
`sys.modules` нового процесса и процесса публикации. Новые файловые хэши
не восстанавливают dependency identity старых рыночных запусков.

## Readiness

Четыре gate имеют статус `verified_scoped`: artificial input/model payload
binding; sidecar для 648 строк и 12 закрытых сделок; сохранение шести outputs;
идентичность наблюдаемых disk files. Sidecar — offline attribution, не
доказательство причинного online поведения и не вход для обучения.

Девять gate остаются незакрытыми: полный native/runtime graph; исторические
dependencies; полный forward market contract; реальные данные/labels/scalers/
folds/resamples/cutoff; причинное обучение и открытые позиции; зарегистрированный
objective/testing budget; статистический net после stress-costs; реалистичные
fills/funding/borrow/intrabar risk; paper/shadow/fault injection/canary.

`market_research_launch_ready=false`, `real_money_ready=false`,
`full_runtime_dependency_graph_verified=false`. Все authority/safety flags false;
новых model fits и market strategy evaluations — 0.

## Следующий содержательный этап

Отдельное предложение forward research contract и статический аудит происхождения
данных, причинности labels и обучения scaler до нового обучения. Зафиксировать
objective, бюджет сравнений, метрики, execution assumptions и правила независимого
финального анализа. Не открывать sealed interval автоматически и не считать
новые искусственные проверки доказательством преимущества над NoTrade.
