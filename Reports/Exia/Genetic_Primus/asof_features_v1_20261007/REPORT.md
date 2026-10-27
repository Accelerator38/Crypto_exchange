# Итоги as-of features — 7 октября 2026

Реализовано причинное построение двух deploy-признаков и long proposal из
доступных к asof **искусственных** свечей. Будущие свечи не требуются и не входят
в hashes feature candidate. Реальные данные, обучение и trading не запускались.

## Проверенный результат

| Проверка | Результат |
|---|---:|
| Direct engineering assertions | 126 passed |
| Artifact assertions | 21 passed |
| Основной видимый prefix | 96 candles / 2 symbols |
| Невидимый future tail | 16 candles |
| История на symbol | 48 bars |
| Registered artificial cases | 5 |
| Numeric fields против rational oracle | 40 |
| Max absolute error / tolerance | 1.54e-15 / 1e-12 |
| Source/metadata input bindings | 32 |
| Новые fits / market evaluations | 0 / 0 |

Cases: mixed rising/falling, rising, falling, flat, alternating. Это engineering
fixtures и assertions, не доходность, не модельные trials и не независимая выборка.

## Что подтверждено в этом scope

Результат целиком, включая hashes, одинаков на prefix-only и на prefix с future
tail; сохраняется при перестановке input rows, изменении будущих OHLCV и добавлении
поздней revision старого бара. Аналогичное prefix-only сравнение прошло для
четырёх дополнительных patterns. При продвижении asof новая доступная свеча
включается; при явно принятом новом видимом prefix features меняются.

Недоступная/пропущенная последняя свеча не заменяется более старой: exact grid
отвергает input. Visible revision mismatch, duplicate, changed pinned past,
invalid OHLCV и malformed timestamps также отвергаются. Late revision должна
сосуществовать в versioned archive с прежней записью, а не перезаписывать её.
Input envelopes/caps проверяются даже для будущего хвоста; invariant не означает
приём произвольно повреждённого/неограниченного файла.

Независимый exact Fraction oracle сверил return12, efficiency12, EMA gap и
three-bar slope; flags proposal/flat проверены отдельно. Исторический pandas
module не исполнялся, поэтому equivalence с ним не объявляется.

## Явное отличие и незакрытые ограничения

При flat path новый kernel задаёт нули и abstention. Старая реализация давала
NaN и отбрасывала такие examples. Это зафиксированная engineering policy, не
скрытая смена objective и не разрешение перенести её в frozen experiment.

History seed фиксирован; bounded 48-bar warmup не аттестует полноценную market
history. Feature availability timestamp объявлен как decision-asof — compute
latency не измерена. Real provenance/revisions и real-indicator equivalence false.

С causal payload guard подтверждена только row schema. Full training/model/scaler
bundle не подключён; модельных predictions/решений/entry ledger нет. Старый
evaluator не изменён. Market launch и real-money readiness остаются false,
все authority flags false. Нет pytest/collector/network/runtime, fitting, PnL,
commit или push. Публикация и receipt проверены, прежний metadata/source bank
unchanged; historical inventory и native graph целиком не переаттестованы.

## Следующий шаг

Причинное построение completed training labels и синтетическая origin integration
с train-asof purge и новым isolated deploy builder. Нужны согласованные feature
definition и hashes, а не привязка прежнего manual model к новым features.
Сначала без обучения; sealed interval не открывать.

[Методика и API](E:/Crypto_exchange/docs/GENETIC_PRIMUS_ASOF_FEATURE_BUILDER_V1.md).
Рядом опубликованы `synthetic_candles.json`, `kernel_definition.json`,
`feature_candidate.json`, `pattern_results.json`, `summary.json`, `completion_receipt.json`.
