# Future EMA: среда и read-only разбор артефактов

Дата: 2026-10-03. Реализация: локальный commit `19f883e`.

## Итог

Добавлены immutable environment fingerprint, отдельный read-only checker и инструкция ручного разбора повреждённых артефактов. Исходный контракт, evaluator, runner и закреплённые implementation hashes не изменены. Никакие пользовательские зависимости не установлены, не обновлены и не удалены.

Environment registration: `configs/genetic_primus_future_ema_environment_v1.json`, SHA256 `8f82114b946f09afcd4aa8b6e5dcd721fb7c2f303a09f1c2dac3a0fc51137360`.

Operator procedure: `docs/GENETIC_PRIMUS_FUTURE_EMA_OPERATIONS_V1.md`.

## Проверки

- `py_compile` для module, checker и тестового файла: успешно.
- Прямая проверка чистой функции сравнения environment: успешно.
- Итоговый синтетический unittest-набор: **15 tests, OK**, **0,056 s**. Первые 13 тестов прошли отдельно до добавления двух negative CLI regressions; итоговый прогон связан с расширением покрытия, не с новой модельной попыткой.
- Фактический CLI `environment`: **ENVIRONMENT_MATCH**, differences=[], market_files_opened=false, runner_launched=false.
- Фактический CLI `inspect`: **AWAITING_INPUT_COMMIT**, artifact_sha256={}, issues=[]; read_only=true, rerun_authorized=false.

Тесты используют только stdlib и маленькие файлы во временном каталоге. Не открывались market datasets, future outcomes, collector или trading runtime; pytest не запускался.

## Поведение

Checker обнаруживает отличие Python build/path/архитектуры, python.exe/pyvenv.cfg SHA и версий/METADATA/RECORD numpy, pandas, pyarrow, python-dateutil, six, tzdata; также появление optional bottleneck/numexpr. Сверяет source SHA исходного контракта и implementation lock. Подмена environment registration либо drift отклоняются.

Инспектор читает только четыре JSON финального workflow. Различает отсутствие input, manifest без подтверждения HEAD, прерванный анализ, сохранённую ошибку, итоговый результат и несогласованное состояние. Отклоняет malformed JSON, duplicate keys, nonfinite values, oversized artifacts, symlink, неверные lineage hashes, timestamp и safety. Не читает segment paths, не выводит PnL, не пересчитывает метрики, ничего не исправляет и не разрешает повторный анализ.

## Ограничения и следующая работа

Это operator preflight, а не новый встроенный gate замороженного runner: оператор должен выполнить его перед финальным запуском. Package payload/DLL integrity, offline wheel bundle, восстановление среды на другом ПК и power-loss durability **не проверены**. Fingerprint не подменяет эти доказательства. Автоматического восстановления `.venv`, удаления reservation и повторного анализа нет.

Практический следующий инженерный этап — отдельный проверенный offline archive/wheel bundle и восстановление в изолированной среде, если доступны локальные установочные файлы. Для скачивания потребуется отдельное разрешение на сеть. Новые модельные прогоны на раскрытых периодах вместо этой работы не предусмотрены.

Новых fits/genomes/statistical attempts/PnL нет. Все модельные gates остаются закрыты, NoTrade сохраняется. Единственный future analysis полного зарегистрированного периода остаётся не ранее **2027-04-03 00:00 UTC**. Сам этот этап не является evidence прибыльности или допуска к реальным деньгам.
