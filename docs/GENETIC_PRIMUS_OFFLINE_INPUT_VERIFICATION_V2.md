# Offline input verification: исправление schema guard до публикации результата

Пользователь разрешил только bounded чтение 11 historical parquet из origin_09,
без fit, финансового backtest, сети, цен в stdout и нового sealed interval.
Каждая команда `--file N` читает один файл, только timestamp/symbol projection.

Первый подготовительный запуск v1 остановился на schema первого файла: physical
schema имела служебный integer `source_rows`, а manifest.columns описывали семь
нормализованных OHLCV columns. Byte size/SHA совпали. Это слишком строгий guard
проверяющего кода, не доказательство испорченных данных или label leakage.
V1 не имеет completion receipt и остаётся диагностическим. Его preparation и
полные source texts обоих новых скриптов сохранены до исправления в
`offline_input_verification_v1_20261007/diagnostic_source_snapshot.json`.
Ни один законченный predecessor artifact/source не изменён.

V2 допускает physical core columns по именам (порядок не является семантикой)
и только дополнительный integer `source_rows`. Physical и normalized schemas
сохраняются отдельно. Extra values не декодируются и не становятся features.
Иные columns, включая future labels, duplicate names или отсутствующий core,
отклоняются. Extra source_rows semantics/value quality этим не подтверждаются.
Допущение зафиксировано до остальных file reads; objective/labels/модель не менялись.

## Изоляция чтения

Parquet <= 8 MiB, <= 200000 projected rows, <= 64 row groups. SHA и size проверены
до/после каждого чтения. Native decoder получает BufferReader проверенных bytes,
не filename/remote filesystem. Price values не декодируются; compressed bytes
ценовых columns присутствуют в файле и участвуют только в byte SHA.

PyArrow import guard блокирует numpy/pandas/exia/pytest/WebSocket/HTTP clients;
socket/process launch audit events запрещены. `use_threads=false`, prebuffer=false.
Четыре decoder files имеют SHA; полная native closure/loaded-memory identity не
аттестуются, это не OS/native sandbox proof. Нет package install или сторонних Python.

## Проверки

Compile одного changed helper -> pure schema/key check -> compile tool -> prepare.
Далее по одной команде file1..11, с оценкой результата между ними; finalize/verify
читают только audit JSONs, не перечитывают несколько parquet за один шаг.
Полные 8-symbol presence bitmaps выявляют межфайловые duplicates и global gaps.
Timestamp int64 трактуется как epoch-ms только для проверки reference grid/range;
arrival/исходная timestamp semantics из этого не следуют.

V2 artifacts создаются exclusive в новой директории. Source/metadata bank
проверяется до/после каждого шага и включает сохранённую v1 диагностику.
SHA исходного parquet относится к его отдельному чтению: source changes после
него не исключаются новым одновременным snapshot. Receipt не заявляет reread всех
source bytes при finalize или при последующем artifact verify.

## Предел результата

Проверяются file identities, compatible schema и timestamp/symbol grid.
Не проверены finite/positive OHLCV, high/low consistency, arrival/revisions,
authenticity/venue/product, actual opportunity sample/ESS, costs/funding/borrow
или доходность. Успех не разрешает training, paper/live, orders или promotion.
Следующее содержательное действие должно закрывать price/source/availability
unknowns и research assumptions, а не повторять synthetic evidence.
