# Accounting/valuation audit: итоги

Дата: 5 октября 2026 года.

## Вывод

Арифметика опубликованных итогов двух выбранных наборов согласована, но полного исторического доказательства бухгалтерской семантики нет. Найден потенциальный конфликт terminal scope: закрытие в конце каждого origin заявлено контрактом, но обоими runner явно не задаётся. До устранения неоднозначности результаты нельзя превращать в основание допуска к торгам или общего рейтинга моделей.

## Проверенный объём

- Линейная матрица: 4 варианта и EMA reference, каждый в 3 cost scenarios; 15 оценок, 60 дней на связанном периоде 2026-06-01 — 2026-07-31, правая граница исключена.
- Сентябрьский baseline economics audit: 5 политик в normal/stress; 10 оценок, 27 дней 2026-09-01 — 2026-09-28, правая граница исключена.
- 100 арифметических проверок consistent, 15 unknown, inconsistent не обнаружено.
- 48 детерминированных регрессий прошли. Independent artifact checker подтвердил receipt, provenance и сохранность base metrics.
- Проверены совпадающие byte SHA двух producer и одного model module; статически рассмотрены три текущие зависимости. Полностью исторически подтверждённых accounting profiles: **0**.
- 1784 исходных файла и старые контракты не изменены. 42 исходных accounting-unknown оценки остаются unknown; этот audit не скрывает пробелы происхождения.

## Что мешает интерпретации

1. **Terminal boundary.** Оба callsite передают полный интервал, без `force_flat_timestamps`. Текущий simulator обязательно закрывает только на global end. Это два потенциальных mismatch с `origin_final_candle_close`, но не доказательство исторического переноса каждой позиции: SHA импортированной зависимости при расчёте отсутствует.
2. **Учёт не равен exchange execution.** Текущий cash-sleeve ledger имеет фиксированное количество до выхода и fees по simulated notional. Funding, borrow и реальные fills не подтверждены; drawdown snapshots не покрывают внутрисвечную траекторию.
3. **Gross нельзя переносить между сценариями.** Для 15 оценок матрицы отсутствует gross PnL той же бухгалтерской траектории. Zero-cost результат не является net + fees другого сценария из-за изменения sizing и compounding.
4. **Daily aggregation.** Сохранённое compounding и количество days согласованы; полнота raw grid и zero-return days отдельно не доказаны. Origin blocks baseline описывают одну непрерывную траекторию, а не независимые liquidation runs.

## Что не изменилось

Новые доходности не рассчитывались; обучение и market replay не выполнялись. Прибыльность моделей не улучшена самим audit. Ранее полученные убыточные сентябрьские итоги остаются убыточными; NoTrade остаётся описательным ориентиром внутри этого набора, не универсальным межпериодным рейтингом. Cross-source comparison authority, execution и promotion остаются false. Сеть, collector, pytest и торговый runtime не запускались.

## Следующее действие

Разделить retrain cadence и liquidation boundary в новом forward engineering contract, явно зарегистрировать carry/terminal policy и закрепить dependency SHA в launch receipt. Проверить эти правила на небольших искусственных fixtures до нового финансового расчёта. Старые frozen контракты и prospective interval не переопределять; нового обучения на отклонённом objective не начинать.

Реализация и ограничения описаны в `docs/GENETIC_PRIMUS_ACCOUNTING_SEMANTICS_AUDIT_V1.md`. Машиночитаемые доказательства находятся рядом: `accounting_audit.json`, `summary.json`, `completion_receipt.json`.
