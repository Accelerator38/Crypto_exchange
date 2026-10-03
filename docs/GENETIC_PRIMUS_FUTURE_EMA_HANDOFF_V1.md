# Future EMA: package handoff archive

Это перенос пакетов, не релиз модели, Python runtime или торговой системы.

Локальный архив: `Reports/Exia/Genetic_Primus/future_ema_handoff_v1_20261004/genetic_primus_future_ema_packages_v1.zip`.

Размер: **50 616 905 bytes**. SHA256:

```text
58561bc839529cba381bb23a8a40b274238e37ced49df7b42d6e944ac3359759
```

Сохранить checksum независимо от ZIP — например, вместе с зафиксированным Git commit отчёта. SHA защищает от несовпадения bytes с зарегистрированным экземпляром, но не подтверждает авторство PyPI или сохранность независимого резервного носителя.

Состав: шесть wheels, шесть package evidence JSON, bundle manifest, environment registration, cache sources registration, requirements с wheel SHA, offline restore smoke, README и stdlib-only verifier. Внутренний index связывает SHA/размеры 19 файлов payload; сам index дополнительно защищён SHA всего ZIP. HTTP envelopes, API-ключи, торговые configs/runtime, модельные веса и установленный target не включены.

## Проверка перед переносом

По одной локальной команде из `E:\Crypto_exchange`:

```powershell
E:\Crypto_exchange\.venv\Scripts\python.exe tools/package_genetic_primus_future_ema_handoff_v1.py verify --archive Reports/Exia/Genetic_Primus/future_ema_handoff_v1_20261004/genetic_primus_future_ema_packages_v1.zip --expected-sha 58561bc839529cba381bb23a8a40b274238e37ced49df7b42d6e944ac3359759
```

Не запускать повторный `export` или `extract` поверх существующих артефактов: overwrite запрещён. Порча источника/архива требует разбора и отдельной версии, а не удаления evidence для скрытого повторения.

На другом ПК сначала проверить ZIP SHA штатным `Get-FileHash -Algorithm SHA256 -LiteralPath <archive.zip>` и сравнить с независимо сохранённым значением. Не исполнять полученный verifier до проверки этого внешнего SHA. Verifier остаётся ограничен утверждённым workspace Python по пути `E:/Crypto_exchange/.venv/Scripts/python.exe`; отдельного Python installer в архиве нет. При отсутствии такого интерпретатора нужна отдельная согласованная процедура подготовки среды. Не переносить `.venv` поверх чужой рабочей среды автоматически.

## Что проверено локально

Восемь синтетических регрессий: source/archive/member SHA mismatch, запрет overwrite, path traversal/Windows reserved names, symlink, duplicate JSON keys, ограничение central directory и неисполнение payload.

Фактический export сверил source manifest/package evidence/wheel SHA. Локальная распаковка проверила CRC/member SHA; отдельное чтение записанных файлов с диска подтвердило SHA всех 19 payload файлов и совпадение index с архивом. Содержимое ZIP не исполнялось, packages не устанавливались, market datasets не открывались.

## Непройденные gates

Архив и extraction-check находятся на том же ПК и диске. Независимая backup-копия, восстановление на другом ПК, отдельный Python runtime, power-loss durability и независимая authenticity upstream packages **не проверены**. Бинарный ZIP не включён в Git: один клон репозитория не заменяет этот комплект.

Следующий существенный шаг требует доступного независимого носителя/ПК и явно заданного места передачи. Не создавать видимость внешней проверки дополнительными копиями на том же E:. До этого остаётся сохранённый локальный handoff, а не подтверждённый disaster recovery.

Model/evidence gates остаются закрыты, NoTrade не меняется. Этот архив не разрешает paper/live, GA или promotion и не открывает будущие outcomes; единственный зарегистрированный анализ полного окна — не ранее 2027-04-03 00:00 UTC.
