# Future EMA: проверенный локальный offline package bundle

Каталог: `Reports/Exia/Genetic_Primus/future_ema_offline_bundle_v1_20261003`.

Шесть wheels восстановлены из локального кэша pip. HTTP-cache envelopes не копируются в bundle: сохраняется только ZIP payload. Обработка legacy `cc=4` не десериализует кэш. Все архивные файлы проверяются по CRC и SHA256 из RECORD; каждый payload-файл, кроме самого RECORD, сравнивается с исходным установленным файлом в `.venv/Lib/site-packages`.

Состав: numpy 2.4.2, pandas 3.0.0, pyarrow 25.0.0, python-dateutil 2.9.0.post0, six 1.17.0, tzdata 2025.3. CPython 3.12 / Windows x64. Wheels, их SHA и исходные cache SHA описаны в шести package JSON; корневой `bundle_manifest.json` фиксирует SHA этих JSON и smoke-результата.

В каталоге уже сохранены `wheels` и `isolated_target`. Это локальные бинарные артефакты, не добавленные в Git. Клон репозитория содержит metadata/code, но не восстановит эти бинарные файлы сам по себе. Для переноса нужно отдельно сохранить комплект wheels вместе с manifest и проверить их SHA. Не архивировать HTTP headers или весь pip cache: они не нужны для восстановления.

## Выполненная проверка

Установка была выполнена разрешённым `E:\Crypto_exchange\.venv\Scripts\python.exe -m pip` с `--isolated --no-index --only-binary=:all: --no-deps --no-compile --require-hashes --no-cache-dir` в новый `isolated_target`, не поверх `.venv`. Requirements содержат SHA всех шести wheels. Первичная проверка отсутствия target обязательна; не повторять установку в тот же каталог автоматически.

Smoke запускался этим же Python с `-I -S -B`. Исходные site-packages исключены; все шесть модулей импортированы из target. Проверены NumPy dot, pandas EWM и Arrow roundtrip в памяти на трёх синтетических строках. Рыночные данные, evaluator и обучение не запускались. `restore_smoke.json` фиксируется эксклюзивно, без overwrite.

После установки отдельный environment checker подтвердил неизменность исходной среды. Старый environment registration не обновлён: его исторический флаг `offline_reinstallation_verified=false` описывает состояние на момент регистрации. Новое package-target evidence хранится отдельно в bundle manifest и research ledger.

## Ограничения

Это восстановление package target на том же ПК с общим исходным Python 3.12.4, не восстановление отдельного интерпретатора, всей ОС или машины. Python DLL/runtime не архивировались. Исходная `.venv` не заменена. PyPI authenticity независимо не проверена: доказано совпадение package payload с уже установленной baseline и внутренним RECORD. Power-loss durability не доказана.

Нельзя превращать это инженерное evidence в разрешение на paper/live или promotion. Не удалять reservation/result финального анализа и не открывать промежуточные outcomes. Восстановление на другой машине, применение к общей среде или принудительное отключение питания требуют отдельной задачи и безопасной изоляции.
