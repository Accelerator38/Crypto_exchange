# Future EMA: проверяемый package handoff

Дата: 2026-10-04. Реализация: локальный commit `c46e8c0`.

## Результат

Создан архив для передачи ранее проверенного offline package bundle. Это пакет зависимостей, **не релиз модели/торгового runtime**. Артефакт сохранён локально:

`Reports/Exia/Genetic_Primus/future_ema_handoff_v1_20261004/genetic_primus_future_ema_packages_v1.zip`.

Размер **50 616 905 bytes**; SHA256 **58561bc839529cba381bb23a8a40b274238e37ced49df7b42d6e944ac3359759**.

Внутри 19 payload-файлов и index: шесть wheels, package evidence, исходный manifest, requirements, environment/cache provenance, прошлый restore smoke, README и stdlib-only verifier. HTTP envelopes, установленный target, модельные веса и credentials отсутствуют. Исходный bundle SHA сверялся с закреплённым значением; package/wheel SHA проверены до экспорта и во время копирования.

## Проверки

- `py_compile` нового archive tool: успешно.
- Прямая проверка чистой функции safe-name: успешно.
- Синтетические архивные регрессии: **8 tests, OK**, **0,040 s**. Проверены source/archive/member SHA mismatch, exclusive create, traversal/Windows names, symlink, duplicate JSON и bounded central directory; payload не исполняется.
- Фактический export: **ARCHIVE_VERIFIED**, 19 payload-файлов, 50 610 989 payload bytes.
- Фактическая распаковка в новый локальный `extraction_check`: **ARCHIVE_EXTRACTED_AND_VERIFIED**, без overwrite и исполнения.
- Отдельное чтение с диска: index совпадает с архивом; SHA всех 19 записанных payload-файлов совпали.
- Source hashes зарегистрированного evaluator и его dependencies остались неизменными.

Ничего не устанавливалось и не обучалось; рыночные datasets/future outcomes, сеть, collector, pytest и trading runtime не запускались. NoTrade и закрытые модельные gates сохраняются.

## Передача и оставшаяся граница

Инструкция: `docs/GENETIC_PRIMUS_FUTURE_EMA_HANDOFF_V1.md`. В Git фиксируются checker/tests, checksums и metadata, **не бинарный ZIP или распакованные wheels**. Сам ZIP нужно передавать отдельно и сверять с checksum из независимого источника.

Архив и проверочная распаковка находятся на том же E: и ПК. Это доказательство согласованности упаковки/копирования, **не независимой backup-копии или disaster recovery**. Восстановление на другой машине, отдельный Python/OS runtime, power-loss durability и upstream package authenticity не проверены.

Следующий существенный шаг требует конкретного независимого носителя/другого ПК и согласованного места передачи. Эти параметры не заданы; произвольные сетевые передачи, выбор облака, запись на неизвестный диск или выключение ПК не выполняются. Дополнительные копии на том же диске не будут выдаваться за независимое доказательство.

Будущий statistical analysis не переносится и не открывается раньше срока: полный зарегистрированный interval заканчивается **2027-04-03 00:00 UTC**, автоматического запуска нет. Инженерный handoff не разрешает GA, paper/live или promotion.
