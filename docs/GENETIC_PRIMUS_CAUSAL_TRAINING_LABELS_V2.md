# Completed training labels и origin integration v2 — 7 октября 2026

Текущая версия этого этапа — **v2**. Первая v1 сохранена как диагностический
результат с известным дефектом shared snapshot identity. Финальное исправление
не переписывает v1 receipt, исходники и outputs.

## Реализовано

- `src/exia/genetic_primus/causal_training_labels_v1.py`: bounded completed-label
  validation, source grid, historical asof features и gross label reference.
  Его shared training/deploy binding в `build_origin_inputs` не использовать
  для training identity.
- `src/exia/genetic_primus/causal_training_labels_v2.py`: отдельный training-only
  constructor и origin integration с независимыми source snapshots.
- `tools/build_genetic_primus_causal_training_labels_v2.py`: exclusive publication
  искусственного примера, provenance, completion receipt.
- `tools/check_genetic_primus_causal_training_labels_v2.py`: direct metadata,
  identity-invariance, negative-control и artifact checks.

Используются только caller-supplied trusted pure APIs с source hashes в receipt.
Market runtime, package `exia`, numpy/pandas и collector не импортируются.

## Причинные labels

Origin starts передаются явно, sorted/unique/contiguous, по 3 дня. Нельзя
запросить label, заканчивающийся позже
`training_asof - 7 * 4h`. Нет silent selection по знаку outcome.

Для каждого symbol требуются все 18 свечей label path, полный grid и pinned
revision. Gross engineering label:

`close последнего бара origin / open первого бара origin - 1`.

Availability label — максимум availability всех 18 path candles; должен быть
не раньше label end и не позже train cutoff. Missing/delayed path candle,
duplicate, revision mismatch и mutation pinned source дают отказ.
Это labels по gross prices, не исполнение, costs, funding, borrow или net PnL.

Training feature kernel тот же, что у deploy. Исторический feature-asof:
`label_start - 4h + pinned synthetic offset`. В fixture offset=2500 ms;
raw availability=bar end+1000 ms. Это искусственная временная convention,
не утверждённый latency budget. Поздняя историческая свеча не используется
задним числом, даже если к training-asof архив уже содержит её данные.

## Исправленный дефект v1

В v1 numerical training rows корректно фильтровались, но `data_file_sha256`/
`data_content_sha256` training payload ссылались на общий snapshot с deploy
OHLCV после training-asof. Поэтому изменение deploy prices меняло training
payload SHA без изменения его features или labels. 95 assertions v1 и 21
artifact check этого не заметили. Они не доказывают полной причинности identity.

V2 разделяет `training_snapshot.json` и `deploy_snapshot.json`.
`build_training_inputs` принимает strict training-only request **без** deploy
source pin и может работать на 244 training candles без deploy data. Изменение
deploy OHLCV, даже видимых к decision-asof, не меняет целиком training record,
source hashes и payload SHA. V1 counterexample сохранён в v2 checker.

В training guard остаётся пустой compatibility slot `sha([])` для deploy rows;
это не допуск deploy и не связь model bundle. Общими остаются pinned feature
definition, universe/revision и плановые context timestamps, а не будущие цены.

## Scaler-input scope

Labels содержат весь fixed universe: 8 rows. Для будущего overlay-fit отдельно
подготавливаются все proposal-long opportunities: 7 rows, включая 2 убытка.
Ни коэффициенты, ни median/IQR не обучались.

Прежний training guard считает scaler-input hash по **всем** training rows;
candidate input hash — только по opportunity rows. Эти scopes явно раздельны.
`full_old_model_bundle_compatible=false`: нельзя использовать общий old model
guard как доказательство корректности будущего actual scaler fitting.

## Проверки и результаты

- V1: 95 checks passed, oracle 8 labels, max error 9.71e-17; затем выявлен
  causal identity defect. Только диагностика, не текущий admission path.
- V2: 63 checks passed; 8 Fraction oracle comparisons, max error 9.71e-17,
  tolerance 1e-12.
- V2 artifacts: 21 checks passed; 50 source/metadata input bindings, five outputs.
- Fixture: 296 artificial candles, training source 244, deploy source 278;
  4 completed origins, 8 labels (2 отрицательных / 6 положительных),
  7 opportunities (2 отрицательных / 5 положительных), 2 deploy rows.
- Minimum opportunities=64 не выполнен; ESS gate не оценён. Fits=0,
  market evaluations=0, model_ready=false.

Проверены standalone training без deploy, invariance training identity при
post-training price mutation, независимые snapshots, future-tail invariance,
отсутствие зависимости training от missing deploy candle, future-label отказ,
delayed/revised source и сохранение losses. V1 negative control должен
воспроизводить дефект; успешное его воспроизведение не делает v1 корректной.

Все времена и цены искусственные, не реальный market interval. Real revision
provenance, execution labels, measured compute latency и entry accounting
остаются unverified. Flat-path abstention policy из asof kernel — явное отличие
от legacy NaN/drop; проект исследования не превращён в executable contract.

## Артефакты и безопасность

`Reports/Exia/Genetic_Primus/causal_training_labels_v2_20261007`:
`synthetic_candles.json`, `training_snapshot.json`, `deploy_snapshot.json`,
`origin_inputs.json`, `summary.json`, `completion_receipt.json`.
Ничего не перезаписывается. Markdown — пояснение вне JSON receipt.
Старые inputs/contracts проверены unchanged в explicit bank; global historical
inventory и native runtime graph заново целиком не проверялись.

Python только `E:/Crypto_exchange/.venv/Scripts/python.exe -X utf8`, одна короткая
локальная команда за раз. Нет pytest, collector, WebSocket, сети, training,
order runtime, API keys, paper/live, promotion или финансовой оценки.
Все authority flags false; market research launch и real-money readiness false.

Следующий шаг: pre-fit origin admission adapter, использующий правильный
opportunity-filtered scaler scope, явные sample/timing gates и fail-closed
причины отказа. На этой маленькой fixture ожидается запрет fit, не вынужденные
сделки. Sealed interval не открывать.
