# Приёмка записанных данных — 2026-09-11

Выполнен следующий локальный этап: импортёр нормализованного recording package
и проверки качества входа **до** portfolio replay. Старый release/execution/
portfolio код и результаты не изменены; новые файлы добавлены отдельно.

## Что добавлено

Проверяются SHA и размеры файлов, локальные пути без выхода из package,
строгий JSON с явным пределом вложенности 64, полная ожидаемая full8/4h-сетка,
свежесть и покрытие book/mark, наблюдаемый стакан после latency до expiry,
funding по явно заданному расписанию каждого символа, rule/fee payloads и
сроки их действия. Проверяется также соответствие declared feature/scaler
lineage: values/source/model hashes, cutoff, availability и label-end bounds.

Важное отличие от предыдущего шага: **полностью пропущенная 4h-граница** теперь
не исчезает незаметно из тестового периода. Отсутствующий funding не становится
нулевой стоимостью. Стакан не синтезируется из OHLCV, missing observations не
заполняются будущими значениями.

## Проверки и артефакты

- **18 финальных прямых регрессий passed:** 8 coverage + 10 provenance/IO/integration.
- Актуальные receipts: `coverage_tests_v2.json`, `provenance_tests.json`.
  Проверены checksums обоих receipts и **8 соответствий SHA исходников**.
- `coverage_tests.json` сохранён как исторический receipt до исправления
  JSON-depth guard. Он не относится к финальной версии и не суммируется с 18 тестами.
- При первой расширенной проверке выяснилось, что нельзя полагаться на
  исключение JSON-парсера при глубокой вложенности. Добавлен явный предварительный
  счётчик глубины, учитывающий строки/escape sequences. Финальные группы повторены
  отдельно, без перезаписи исторического receipt.
- Файловый smoke: `package_manifest.json` + `package_bundle.json` +
  `package_provenance.json` → `admission_report.json` и `admitted_bundle.json`.
  Проверено точное равенство экспортированного и входного bundle как JSON-объектов.

В пакете **56 синтетических событий**, две full8-границы решений и восемь
funding settlements. Классификация осталась `synthetic`.
`admitted_for_offline_replay=true`, **`qualification_eligible=false`**.

Canonical admission body SHA:
`8939651ab23c51198e31f417deb570278794252d1f8c769e5a9542198c4f6315`.

Сам импорт не запускал matching или портфельный расчёт — это проверено тестом,
запрещающим `Replay.ingest`. Отдельная короткая synthetic integration regression
проверила совместимость экспортированного bundle со старым общим portfolio.
Никакой новой рыночной доходности в этом отчёте нет.

В начале работы подтверждены 40 source-hash checks по portfolio receipts от
10 сентября. Каждый импорт дополнительно проверяет замороженный receipt и
его зависимости до импорта проектного кода. Pytest, collector, обучение,
live-сеть, фоновые процессы и торговый runtime не запускались. Ключи/ордера
не затрагивались, все торговые разрешения остались false.

## Граница доказанного

Проверена согласованность предоставленного package, **не подлинность записи
биржи**. `source_authenticity_verified=false`, `feature_computation_verified=false`,
`scaler_recomputed=false`, `calibration_verified=false`. Полная история bars
и scaler arithmetic не пересчитывались: timestamps сопоставлялись с заявленным
lineage. Coverage thresholds и funding schedule тоже требуют внешней проверки.

Импортёр принимает нормализованный JSON, не произвольные сырые WS/CSV/Parquet
дампы. Общий лимит — 2 MiB и до 1024 input events: короткая инженерная проверка,
не многодневный streaming backtest. Допуск к импорту не равен допуску к paper/live,
независимому statistical audit или подтверждению преимущества GA.

## Следующий реальный блокер

Подтверждённый согласованный recorded dataset не предоставлен. Ограниченный
поиск book/depth/mark/recorded-manifest файлов в `Retrodate` и
`Reports/Exia/Genetic_Primus`, исключая benchmark-артефакты, не дал подходящих
JSON/Parquet записей. Это не утверждение об отсутствии дампов под другими
именами или в иных каталогах.

Для следующего **рыночного**, а не синтетического прогона нужен путь к реальным
book/mark/funding-записям, rules/fees и происхождению features/scaler. Если
доступны только raw dumps, сначала понадобится согласовать их read-only mapping.
Получение данных через сеть отдельно не разрешалось и не выполнялось.
Новое обучение и ещё один synthetic profit этот блокер не устраняют.

Формат и команды:
[GENETIC_PRIMUS_RECORDED_ADMISSION_V1.md](../../../../docs/GENETIC_PRIMUS_RECORDED_ADMISSION_V1.md).
