# Ограниченный L2 format smoke

Это локальная инженерная проверка BTC-архива 2024-12-01, а не новый
ретротест стратегии за август–сентябрь 2026. Snapshot-проверки повторно
используют frozen `market_fragment_v1`; старый исходник не изменяется.

## Чтение

Caller открывает файл read-only. `bounded_gzip_csv_v1` читает только начало:
512 CSV data rows, максимум 64 KiB compressed и 128 KiB inflated, line cap
2048 bytes. Inflate chunk ≤4096 bytes; после достижения row budget в памяти
могут остаться непарсенные строки из последнего chunk, но общий byte cap
никогда не расширяется. Loop hard cap 1024, gzip members ≤32; retries нет.
Начало stream обязано быть byte 0. CSV fields сохраняются исходными строками.
Нет pandas/dataframe, workbook import, распаковки полного файла или fsync.

Compressed/uncompressed PREFIX и parsed row bank имеют разные SHA256.
Это не полный hash архива. При prefix stop весь gzip CRC/trailer не проверен;
ограничение явно записано, а не скрыто под успешным чтением CSV. На маленьких
синтетических gzip дополнительно проверены CRC, EOF, truncation и concat.

## Реконструкция

Формат: exchange, symbol, timestamp, local_timestamp, is_snapshot, side,
price, amount. Clocks — integer UTC microseconds, без округления в ms.
В будущем clock adapter должен отдельно решить совместимость с ms IOC;
нельзя терять source ordering у событий внутри одной миллисекунды.

Группа сообщения выводится из local timestamp, exchange timestamp и
snapshot flag. Это inferred identity, не wire packet/venue sequence ID.
Публикация только после появления следующей группы. Последняя неполная
или даже выглядящая полной группа в prefix отбрасывается без flush.

Snapshot содержит ровно зарегистрированные 15 bid и 15 ask levels и
полностью заменяет старое observed state. Это observed subscribed depth,
не вся книга venue. Синтетический delta-контракт: absolute replacement,
zero delete, не сложение; реальные deltas должны отдельно подтвердить
семантику. State не обрезается незаметно: bounds breach блокирует книгу.
При atomic group проверяются duplicate prices, positive snapshot qty,
nonnegative delta qty, nonempty sides, отсутствие crossed/locked book.

Malformed rows, регресс clocks, неоднозначные same-receipt группы, gap/
disconnect и silence threshold инвалидируют state. Deltas после разрыва
не восстанавливают книгу; нужен новый полный snapshot. Exchange watermark
не опускается после ошибочного snapshot. Silence threshold — консервативное
инженерное правило, не доказательство потери пакета и не калибровка latency.

Без venue sequence невозможно надёжно обнаружить тихо потерянный delta.
Это показано отдельной регрессией: обе версии входа без формальной ошибки,
но состояние различается. Поэтому venue sequence/full-depth/availability
qualification всегда false, даже когда 31/31 checks и real prefix проходят.

Checkpoint сохраняет bounded accepted row prefix, source/policy/result hashes.
Restore повторяет prefix и сверяет результат. После source faults checkpoint
не выдаётся: для продолжения нужен заново обоснованный сегмент. Нет pickle,
авторестарта, reconnect или фонового процесса.

## Доказательства и дальнейшая интеграция

Реальный prefix и синтетические faults разделены. Если в prefix только
snapshots, это не называется эмпирической проверкой delta feed. Сравнение
со старым snapshot-only валидатором проверяет сами observed levels/clocks,
но не рыночную прибыльность. График содержит только BBO и provider delivery
lag на коротком фрагменте; order latency и исполнение стратегии не измерялись.

`local_timestamp` — время получения provider, не доказанное время доступности
данных нашему агенту. Нет observed order/fills/rules/fees qualification.
Этот архив не регистрируется как full8 bundle для 2026, не строит экономические
labels, не переобучает модели и не меняет старые rankings/LCB/прибыли.

Следующий data шаг: отдельные format/preflight проверки trades/derivative
источников и общей timebase, затем реальный full8 bundle с проверенными
coverage, sequence/gaps, availability, historical rules/fees и settlement
marks для validation и causal training scope. Включать funding на terminal
границе. До этого financial replay и training остаются выключены.

Все 9 safety flags false, никакой сети/runtime/collector и никакого pytest.
