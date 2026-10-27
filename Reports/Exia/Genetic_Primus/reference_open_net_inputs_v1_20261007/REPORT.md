# Reference-open net-accounting: слой готов, costs пока не зарегистрированы

Дата: 2026-10-07. Статус: arithmetic и входной bridge проверены; реальные net-targets не вычислялись. Обучение не запускалось.

## Результат этапа

Добавлен отдельный accounting для OHLCV reference-open. Прежний immutable synthetic-mid kernel не изменён и не используется для маскировки реальных рыночных цен под синтетический инструмент.

К новому формату подключены все 4 568 ранее полученных samples. Каждый input связан с byte SHA/номером строки исходного gross-label JSONL, canonical SHA sample и identity строки признаков. Сохранены все symbols, outcomes и proposal/abstention состояния. Causal next-open, выход через 18 баров и maturity/purge повторно проверены.

Во всех 4 568 inputs net-, stress-net- и paired incremental targets остаются **None**, а не 0. Вызовы accounting на реальных рыночных samples: **0**. Сформирован лишь заблокированный входной формат для следующего этапа. Размер артефакта: 6 815 580 bytes; он содержит локальные исторические reference-open цены, не выводившиеся в чат.

Закреплены 223 input identities, hashes data/result и completion receipt. Повторное построение и проверка receipt прошли.

## Математика, реализованная и проверенная на fixtures

- Quantity рассчитывается из выделенной суммы с резервированием entry fee внутри allocation; leverage не подразумевается.
- Entry/exit комиссии применяются отдельно к respective adjusted notionals.
- Adverse price adjustment задаётся явно относительно reference-open. Он не является автоматически восстановленным mid, half-spread или observed fill.
- Quantity фиксировано до зарегистрированного выхода; денежный остаток после входа и recovered cash после выхода считаются явно.
- Cash-flow путь согласуется с gross минус price drag, обе fees, signed funding и borrow в пределах Decimal precision tolerance.
- Funding требует полного явно заданного списка marks/rates и порядка settlements строго внутри интервала. Положительная ставка debit Long, отрицательная credit. Событие ровно на boundary без sequence evidence отклоняется.
- Borrow — отдельный явно заданный quote debit, не автоматический 0.
- NoTrade не имеет позиций, reference prices или costs. Paired actions требуют одинакового капитала, символа и интервала; Long/Long также одинаковых observed inputs.

Арифметика использует Decimal precision 50; входные числа ограничены по размеру и точности без скрытой quantization. Это не подтверждение venue rounding, liquidity, latency или fills. Per-interval cash accounting не является непрерывным full8 portfolio ledger и не рассчитывает промежуточный drawdown.

## Что проверено

Успешно выполнены:

1. py_compile только нового чистого helper.
2. Прямой вызов gate: неизвестные параметры блокируют net и fit.
3. `--check-code`, `--verify-code`: fabricated cash recovery, asymmetric fees/drag, signed funding, borrow, stress monotonicity, NoTrade, paired differences, согласованность при изменении ambient Decimal precision.
4. Guards неизвестных fees/capital/borrow, неполного funding, ambiguous/duplicate settlements, подмены reference kind, горизонта, paired capital, sample identity/gross value и purge.
5. `--build`, `--verify`: causal bridge для 4 568 реальных samples, сохранение неизвестных параметров, byte identities и receipt.

Code checks используют вымышленные цены/rates. Они хешируют предыдущие производные файлы, но не разбирают их числовые значения. Build/verify читают только уже сформированные gross samples, не raw parquet. Нативный decoder и numpy/pandas не импортировались.

Эти проверки доказывают engineering consistency и воспроизводимость. Они не являются независимым аудитом данных биржи или подтверждением прибыльности.

## Блокирующие регистрации

Отдельный v1 контракт фиксирует структуру accounting, но остаётся draft. Семь полей относятся к costs/exposure/calibration:

- normal per-side scenario;
- stress per-side scenario;
- общий normalization quote capital;
- allocation rule;
- funding policy и evidence;
- borrow basis;
- market-cost evidence.

Ещё два поля относятся к выбору модели/обучению: decision threshold rule и net training objective. Все девять пока None. Это сгруппированные нерешённые поля, а не девять новых статистических promotion gates. Старые aggregate round-trip bps не раскладывались молча по сторонам.

Draft намеренно не принимает cost-регистрацию через изменение завершённого файла. Следующая регистрация должна быть отдельной immutable записью с provenance и явным статусом measured/assumed. Unit-test rates нельзя использовать как рыночные normal/stress costs по умолчанию.

## Следующий шаг

Нужно выбрать способ регистрации:

1. Только измеренные costs: установить локальные источники комиссии, funding/marks и reference-to-execution evidence; недостающие данные не заменять нулями.
2. Явно условный инженерный scenario: отдельно зарегистрировать параметры и предположения, без claims реальности, обучения, выбора лучшей модели или допуска к торгам.

После регистрации costs остаются задачи непрерывного portfolio accounting, причинного fit/scaler и независимой temporal validation. В текущем этапе нет результатов модели или сравнения её прибыли с NoTrade.

Raw parquet reads: 0. Net target values: 0. Fit: 0. Финансовая оценка моделей: 0. Все девять safety-разрешений false. Сеть, collector, runtime, paper/live, ключи и ордера не запускались. Исторический интервал не становится prospective.

## Артефакты

- `configs/genetic_primus_reference_open_net_contract_v1.json`
- `src/exia/genetic_primus/reference_open_net_accounting_v1.py`
- `tools/build_genetic_primus_reference_open_net_inputs_v1.py`
- `docs/GENETIC_PRIMUS_REFERENCE_OPEN_NET_INPUTS_V1.md`
- В этом каталоге: `data/blocked_net_inputs.jsonl`, `result.json`, `completion_receipt.json`.
- Fixtures/check receipt: соседний каталог `reference_open_net_code_v1_20261007`.

Пояснительный отчёт находится вне immutable receipt bank и не меняет его содержимое.
