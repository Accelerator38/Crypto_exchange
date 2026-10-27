# Исторические causal labels: материализация завершена

Дата: 2026-10-07. Статус: реальные исторические samples сформированы; receipt и повторное построение проверены. Обучение не запускалось.

## Подтверждённый сценарий

Пользователь подтвердил отдельное от data availability допущение: вычисление решения мгновенное, вход строго на первом следующем 4h open после готовности решения, выход через 18 баров. Ingestion lag остаётся 0. При decision на границе бара вход сдвинут на один бар, а не совпадает с decision open.

Это инженерный сценарий. Фактическая задержка вычисления/доставки, fills и исполнимость reference-open не доказаны. Предыдущий draft-контракт и результаты прошлых этапов не изменялись; отдельная preparation фиксирует подтверждение пользователя только для диагностического чтения labels.

## Что теперь получено

- Девять ранее закреплённых источников прочитаны последовательными короткими командами. SHA и размер проверены до и после каждого чтения.
- Из 82 800 включённых исходных OHLCV-строк сохранены 4 576 уникальных наблюдений open, необходимых для входов/выходов.
- Существующая сетка содержит 572 candidate origins; 571 проходит maturity/purge, один поздний origin исключён. Два контекстных события не являются training samples.
- Получено 4 568 samples для восьми символов. Каждый содержит два прежних признака, proposal-флаг, входное/выходное наблюдение и диагностический gross open/open outcome.
- Исходы не фильтруются ни по знаку, ни по proposal: 2 320 отрицательных, 2 247 положительных, один нулевой.
- Размер JSONL samples: 8 517 665 bytes. Артефакт содержит исторические цены и labels и предназначен для локальной машинной обработки, не для вывода в чат.
- Закреплены 207 input identities, preparation, последовательная цепочка частей, hashes open-артефактов, samples, result и completion receipt.

Это распределение исходов reference-цен, **не** PnL стратегии, не win rate торгового агента и не сравнение с NoTrade. 4 568 строк не равны 4 568 независимым наблюдениям: символы внутри каждого origin коррелированы.

## Временные границы

Первый допустимый decision: 2022-01-10 00:00 UTC. Последний: 2026-09-16 00:00 UTC. Последний label exit: 2026-09-19 04:00 UTC. Последняя выходная строка доступна после её закрытия в 08:00 UTC при lag 0.

Training as-of: 2026-09-21 20:00 UTC. Purge cutoff: 2026-09-20 16:00 UTC, то есть training as-of минус семь 4h-баров. Availability каждой label не позже cutoff. Выход остаётся ровно через три дня после входа. Deployment labels не читаются/не присоединяются; хвостовые источники Sep22–27 не открывались.

Исторический интервал не переобъявляется prospective.

## Lineage и проверки

Для каждого sample сохранены feature file byte SHA и номер строки, canonical SHA строки признаков, prefix SHA, timestamps и символ обеих price dependencies, canonical SHA полной исходной OHLCV-строки и byte identity её parquet-источника. Canonical row SHA относится к декодированному представлению; это не оригинальный exchange-message hash и не аутентификация биржи.

Успешно выполнены:

1. py_compile нового чистого helper.
2. Минимальный прямой вызов gross-функции на вымышленных loss/zero примерах.
3. `--check-code` и `--verify-code`: вымышленные zero/negative/positive исходы, abstention без удаления sample, guards next-open и source provenance.
4. `--prepare`, девять отдельных `--file N`, `--finalize`.
5. `--verify`: реконструкция samples из производных opens и прежних признаков, сверка byte identities и receipt без повторного чтения parquet.

Повторное построение использует тот же алгоритм и доказывает воспроизводимость/целостность, а не независимую верификацию торгового результата. Вычисление gross использует Decimal с precision 50 на десятичном представлении исходных чисел. Backend может декодировать поздние строки внутри batch; выходные записи ограничены разрешённым prefix и списком необходимых dependencies.

## Ограничения и следующий этап

Outcome = exit reference open / entry reference open − 1. Это диагностический gross label, **не net training target**. Costs, funding, borrow, исполнение, portfolio ledger и paired incremental return пока не добавлены в эти samples. Значения сохранены для проверки causality и будущего accounting, но fit на этом gross objective запрещён.

Следующий этап: согласовать reference-price bridge и costs/accounting, затем превратить samples в причинные net targets с нормализацией на одинаковый капитал и сравнимыми NoTrade/incumbent actions. Нельзя молча считать OHLCV open mid/fill или переносить старый агрегированный round-trip cost на отдельные стороны исполнения. До fit требуется зафиксировать оставшиеся поля контракта и измерительный objective.

Fit: 0. Финансовая оценка моделей: 0. Все девять safety-разрешений false. Сеть, collector, trading runtime, ключи, paper/live и ордера не запускались. Никакого допуска к реальным деньгам этот этап не даёт.

## Артефакты

- `src/exia/genetic_primus/market_gross_label_materialization_v1.py`
- `tools/materialize_genetic_primus_market_gross_labels_v1.py`
- `docs/GENETIC_PRIMUS_MARKET_GROSS_LABELS_V1.md`
- В этом каталоге: `preparation.json`, девять `part_NN.json`, `data/opens_NN.jsonl`, `data/gross_label_samples.jsonl`, `result.json`, `completion_receipt.json`.
- Проверка кода: соседний каталог `market_gross_label_code_v1_20261007`.

Этот отчёт находится вне immutable receipt bank и не меняет его содержимое.
