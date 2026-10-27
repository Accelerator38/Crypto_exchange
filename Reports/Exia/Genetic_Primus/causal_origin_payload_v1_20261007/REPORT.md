# Результат causal payload этапа — 7 октября 2026

Training labels и deploy features теперь имеют отдельные strict schemas и
валидаторы. Добавлена проверка причинного порядка объявленных timestamps и
согласованности training/model/scaler hashes. Это новый изолированный компонент;
старый evaluator не переписан и не получил допуск к запуску.

## Проверенный результат

| Проверка | Итог |
|---|---|
| Direct metadata regressions | 80 assertions passed |
| Published artifact checks | 21 assertions passed |
| Completed training fixture | 8 rows: 4 отрицательных, 4 положительных outcomes |
| Label-free deploy fixture | 2 rows, terminal prices не нужны |
| Source/metadata input bindings | 23 files, unchanged before/after |
| Model/scaler | Manual fixtures, не fitted |
| Новые fits / market evaluations | 0 / 0 |

Никакого PnL или рейтинга моделей на этом этапе не рассчитано. Counts — инженерные
assertions, не независимые финансовые испытания. Fixture epochs логические,
не реальное окно рынка; 8 rows не закрывают minimum opportunities/ESS gates.

## Какие ошибки теперь отвергает компонент

- Future/незавершённый label либо availability за purged train cutoff.
- Cutoff от deploy start вместо training-asof.
- Future/unclosed feature bar или feature availability после решения.
- Labels, future outcomes, terminal close и любые extra columns в deploy.
- Подмена revision, origin, universe или rows даже с пересчитанным self-hash.
- Несогласованные train payload, scaler input/payload и active model hashes.
- Model availability после решения или невозможный порядок fit timestamps.
- Missing/duplicate/unordered fixed-universe groups, пропущенные training origins,
  invalid types/nonfinite scalars, malformed hashes/JSON.

Deploy payload можно построить и проверить отдельно от training table. Это
устраняет зависимость **нового интерфейса** от future labels, но не доказывает,
что внешний feature producer использует только доступные на тот момент свечи.

## Что намеренно не заявляется

Manual scaler и coefficients не выдаются за обученные. Времена fit — объявленные
synthetic metadata, не измеренная latency. Trusted fixture pins проверяют
coherence, но не происхождение реальных OHLCV/revisions. Grid hash относится к
искусственной таблице, не биржевой полноте свечей. Entry model SHA для открытых
позиций и accounting integration ещё не проверены этим компонентом.

Все соответствующие actual verification fields в `guard_result.json` false.
`market_research_launch_ready=false`, `real_money_ready=false`. Все credentials,
orders, execution, paper/live, promotion, training, financial evaluation и network
flags false. Рыночные данные не читались; pytest, сеть, collector, старые модели
и торговый runtime не запускались. Commit/push не выполнялись.

Публикация receipt связывает пять JSON outputs и explicit source/metadata bank,
включая предыдущий static audit и его зафиксированные contracts. Исторический
inventory целиком и native dependencies заново не проверялись.

## Следующий шаг

Построение deploy features строго as-of и prefix-invariance проверки на небольших
искусственных свечах. Будущие additions/perturbations не должны менять старый
payload; missing/revised inputs должны явно fail closed. После этого — отдельная
интеграция с causal evaluator, не новый search/training/sealed run прямо сейчас.

[Методика и API](E:/Crypto_exchange/docs/GENETIC_PRIMUS_CAUSAL_ORIGIN_PAYLOAD_V1.md).
JSON рядом: `synthetic_snapshot.json`, `fixture_pins.json`, `payloads.json`,
`guard_result.json`, `summary.json`, `completion_receipt.json`.
