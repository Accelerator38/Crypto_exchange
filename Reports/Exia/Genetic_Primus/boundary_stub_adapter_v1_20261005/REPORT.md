# Boundary adapter с заглушкой: итоги

5 октября 2026 года.

## Итог этапа

Реализован отдельный restricted adapter для synthetic call protocol. Он явно передаёт все девять параметров рассмотренного simulator и сохраняет различие SHA модели решения и SHA модели входа позиции. Заглушка только записывает вызов: реальный evaluator и accounting simulator не запускались.

## Результаты

- 3 случая — carry, retarget и NoTrade — в 2 reference cost scenarios: **6 фиксированных комбинаций**.
- 648 synthetic decision records, 12 terminal expectations; 4145 детерминированных checks прошли. Checks включают assertions на каждой строке, это не 4145 независимых финансовых экспериментов.
- Сопоставлены девять argument names; tuple universe/force-flat, interval/timeframe/cost/details передаются явно.
- Проверена независимая state-machine oracle для ledger; input hashes сохраняются, повреждённые synthetic payloads и произвольные callees отклоняются до вызова.
- Carry сохраняет entry-model SHA при смене decision-model SHA; NoTrade остаётся без позиции. Gap fixture различает previous close и origin open.
- Создан bank из 5 source fingerprints с AST import facts: 2 pure fixture modules, 3 reference files без импорта.
- Independent artifact checker подтвердил receipt, source bank, output hashes и ledger identities. 1784 исторических файла и старые контракты не изменены.

## Граница доказательств

Совпадение signature names не подтверждает DataFrame compatibility, quantity/cash carry, фактические fees, NAV или fills. Terminal entries — ожидаемые действия с `actual_fill_verified=false`, не результаты исполнения. Static bank не является полным transitive или loaded-runtime graph.

Protocol stub gate подтверждён; real evaluator adapter и runtime dependency graph gates остаются false. Receipt является synthetic engineering receipt, не финансовым launch receipt. Обучения, market replay, сети, collector, pytest и торгового runtime не было. Credentials/orders/paper/live/execution/promotion выключены. Прибыльность моделей не изменилась и отсутствие убытков не доказано.

## Дальше

Перейти от call protocol к минимальной независимой проверке бухгалтерской семантики на artificial inputs: DataFrame schema, carry quantity/cash, actual-notional fees, final-close liquidation. Сначала проверить изолированность callee и закрепить dependency hashes; не возвращаться к обучению или market backtest до этих проверок. Старые/prospective результаты не перезапускать автоматически.

Подробности: `docs/GENETIC_PRIMUS_BOUNDARY_STUB_ADAPTER_V1.md`. Машиночитаемые доказательства рядом: `fixtures.json`, `source_bank.json`, `summary.json`, `completion_receipt.json`.
