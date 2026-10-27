# Авторизованная offline-проверка старого manifest origin_09

Пользователь явно разрешил bounded offline verification historical parquet.
Разрешение относится только к 11 paths из старого immutable manifest, не к fit,
financial evaluation, runtime, network или новому sealed interval.
Ранее опубликованный preregistration record не изменяется и не используется
как разрешение: дополнительная авторизация записана отдельно в preparation.

## Минимальное чтение

Одна команда `--file N` читает **один** approved parquet <= 8 MiB, до 200000 rows.
Проверяются exact path/size/SHA/PAR1, schema и только timestamp/symbol projection.
File bytes содержат сжатые OHLCV columns, но price values не декодируются,
не печатаются и не сохраняются в outputs. Формат timestamp — только int64;
единицы не угадываются автоматическим conversion. Epoch-ms interpretation
проверяется по reference range и 4h grid, но не объявляется доказательством
исходных timestamp/arrival semantics.

PyArrow читает BufferReader уже проверенных immutable bytes: нет Arrow pathname/
remote filesystem. `use_threads=false`, `pre_buffer=false`, без pandas metadata.
Import guard запрещает numpy/pandas/exia/pytest/websockets и HTTP clients;
socket/process audit events запрещены. Это не OS/native sandbox attestation.
Четыре decoder files имеют byte SHA, полная native closure/loaded memory не
аттестуется. Native libraries не используются для торговли или model fitting.

## Сопоставление файлов

Per-file artifact содержит только type descriptions, aggregate key counters и
компактную presence bitmap по reference symbols. Проверяются null/type/range/grid,
intra-file duplicates и per-symbol internal gaps. Regressions в row order
диагностические: file row order не равен arrival order и может быть нормализован.

`--finalize` читает только созданные audit JSONs, не parquet. Проверяются union
8-symbol grid, expected row/timestamp counts, cross-file overlaps и global gaps.
SHA source bytes проверен до/после **каждого отдельного чтения**, не заново всех
файлов одновременно в finalize. `--verify` тоже проверяет artifacts/source code
bank без reread parquet. Изменение исходных данных после их audit не исключается
как новая независимая snapshot guarantee; receipt фиксирует проверенную identity.

## Последовательность

Только workspace Python `-X utf8`, одна короткая команда, затем оценка результата:
compile одного нового helper; pure key-function; compile tool; `--prepare`;
`--file 1` ... `--file 11`; `--finalize`; `--verify`.
`--prepare` не читает parquet; выполняет bounded pure key regressions и фиксирует
predecessor/source bank. Outputs создаются exclusive, не перезаписываются.
Нельзя вызывать два file indices одной командой или фоновые worker processes.

## Что эта проверка не подтверждает

Не проверены OHLCV price quality, arrival/revisions, source authenticity,
venue/product equivalence, fees/funding/borrow, реальные fills/latency,
число market opportunities/ESS или доходность. Успех grid/hash audit не включает
training authority и не доказывает edge над NoTrade.
Все credentials/orders/execution/paper/live/promotion flags остаются false.
Следующие проверки должны закрывать реальные unknowns, а не повторять mock tests.
