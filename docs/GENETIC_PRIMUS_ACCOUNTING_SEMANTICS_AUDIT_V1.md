# Genetic Primus: accounting/valuation audit v1

Дата: 2026-10-05. Только offline-аудит сохранённых чисел и AST исходного кода. Торговый evaluator, collector, обучение и сеть не запускаются.

## Область проверки

Проверяются два набора: `rolling_linear_matrix_v1_20261002` (15 оценок, включая EMA reference) и `baseline_economics_audit_v1_20261002` (10 оценок). Общий frozen inventory содержит 1784 JSON-файла; их размер и SHA перепроверяются до и после публикации.

Уровни доказательств разделены:

1. **Заявлено контрактом**: капитал, комиссии, моменты сигнала/исполнения, terminal rule.
2. **Наблюдается в текущем коде**: статически проверенные реализации sizing, cash ledger, snapshots и terminal close.
3. **Исторически закреплено**: только файлы, чьи byte SHA записаны в исходном результате или связанной конфигурации. Совпадение SHA producer не доказывает версию его импортированных зависимостей при старом запуске.

Историческая идентичность accounting/helper dependencies не подтверждена. Поэтому audit не заполняет неизвестные поля исходного каталога и не разрешает cross-source ranking или promotion.

## Наблюдаемая модель учёта

Текущая реализация использует synthetic cash ledger с collateralized short, начальным equity 1 и равными начальными sleeves. Между sleeves нет ребалансировки. Entry notional равен `cash / (1 + fee_rate)`, количество фиксировано до выхода; fees начисляются на фактический simulated fill notional. NAV определяется cash и signed quantity, оценённой по mark. Это не доказательство реалистичного исполнения биржевых фьючерсов: funding, borrow и реальные fills не подтверждены.

При неизменном target перенос позиции разрешён. По умолчанию `force_flat_timestamps` пуст, обязательное закрытие выполняется в конце всего интервала. Drawdown учитывает initial/open-mark/after-fills/close-mark/terminal-fills snapshots, но не внутрисвечную траекторию.

Контракт указывает `terminal_exit=origin_final_candle_close`, а оба проверенных callsite не передают `force_flat_timestamps`. Audit фиксирует **potential terminal scope mismatch**, а не доказанное историческое нарушение. Нельзя заключить, что каждая позиция действительно пересекала границу origin.

## Арифметические проверки

В пределах каждого сохранённого сценария проверяются compounding daily returns, число daily entries относительно связанного периода, summary/economics net и fees, contributions по symbols и compounding непрерывных origin blocks. Baseline blocks являются разбиением одной непрерывной бухгалтерской траектории, а не независимыми сделками с обязательным закрытием.

Сравнение использует абсолютный tolerance `1e-10`. Пустые, отсутствующие или нечисловые значения остаются unknown; отсутствие данных не превращается в нулевую доходность.

Итог: **100 consistent, 15 unknown, 0 inconsistent**. Все 15 unknown относятся к отсутствующему same-trajectory gross PnL линейной матрицы. Нельзя подставлять `gross_zero_cost` из другого сценария: fees меняют cash, sizing и последующие PnL. Тождество gross − fees = net применимо только к одной бухгалтерской траектории.

Текущие daily helpers используют 24-часовые bins относительно первого portfolio close. Полнота исходной сетки, сохранение всех zero-return days и историческая версия helpers этим audit не доказываются. Совпадение количества опубликованных дней не заменяет проверку raw grid.

## Безопасное воспроизведение audit

Каждая команда выполняется отдельно с оценкой результата; только `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8`.

- `tools/check_genetic_primus_accounting_semantics_audit_v1.py`: 48 детерминированных арифметических/AST-регрессий; без импорта simulator.
- `tools/build_genetic_primus_accounting_semantics_audit_v1.py --verify-only`: preflight без записи.
- Builder без флага: exclusive-публикация в новый каталог; существующие результаты не перезаписывает.
- `tools/check_genetic_primus_accounting_semantics_artifacts_v1.py`: независимая сверка receipt, input/output SHA, base metrics и provenance.

Артефакты: `Reports/Exia/Genetic_Primus/accounting_semantics_audit_v1_20261005/{accounting_audit.json,summary.json,completion_receipt.json}`. Receipt связывает входы, код и машиночитаемые outputs; поясняющие Markdown-файлы не входят в этот output bank.

## Следующий этап

Подготовить новый forward engineering contract с однозначной политикой carry/terminal и единицей капитала/compounding; отдельно определить retrain boundary и liquidation boundary. Не редактировать старый frozen контракт, результаты или запечатанный prospective interval.

В будущий launch receipt включить SHA всех бухгалтерских/helpers dependencies и полный набор параметров callsite. Перед любым новым финансовым расчётом проверить boundary semantics на малых детерминированных fixtures. Это не разрешение на новый objective search, сеть или торги. Execution и promotion authority остаются false.
