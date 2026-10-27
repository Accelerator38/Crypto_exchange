# Causal origin payload v1 — 7 октября 2026

Реализован отдельный чистый stdlib контракт для **синтетических метаданных**.
Он не меняет старые evaluator/модели, не обучает scaler или коэффициенты и не
разрешает market evaluation. Проект исследования от 6 октября остаётся draft.

## API и разделение

`src/exia/genetic_primus/causal_origin_payload_v1.py` содержит:

- `validate_training(payload, pins)`: только completed labels и training features.
- `validate_deploy(payload, pins)`: только deploy features; работает отдельно,
  без training labels и terminal prices.
- `validate_model(payload, pins, train_info)`: согласованность manual model/scaler
  fixture, hashes и объявленных времён.
- `validate_bundle(bundle, pins)`: совместная проверка isolated payloads и
  active decision model SHA; без prediction и accounting.

Schema строгая, с exact whitelist на каждом уровне. Deploy row содержит только
symbol, revision, feature bar open/end/availability, два features и proposal.
Любые labels, returns outcomes, terminal close или неизвестные поля отклоняются,
даже если caller пересчитал checksum. Признаки `return_12` и `efficiency_12` —
прошлые features; колонка `gross_return_3d` разрешена только в training rows.

`tools/build_genetic_primus_causal_origin_payload_v1.py:make_deploy` принимает
pins, feature rows и model SHA. Training/labels/future prices аргументов нет.
Это изоляция интерфейса, **не** доказательство причинности upstream indicators.

## Временные guards

Все времена — typed integer UTC epoch milliseconds, без bool. Длительность
бара 4h, origin 3d, purge 7 bars. Cutoff:

`train_cutoff = training_asof - 7 * bar_duration`.

Каждый training label имеет длительность 3d и удовлетворяет:

`label_end_exclusive <= label_available_at <= train_cutoff`.

Feature bar должен быть целым и закрытым, а доступность — не раньше его конца.
Training feature available не позже label start и training-asof; deploy feature
available не позже decision-asof. Объявленная модель должна удовлетворять:

`training_asof <= fit_started <= fit_completed <= model_available <= decision_asof
<= deploy_start`.

Это проверки **объявленной** последовательности. Реальную compute latency,
submit/fill timestamps и costs этот этап не измеряет и не подменяет нулями.

## Hashes и revisions

Caller передаёт заранее зафиксированные fixture pins, отдельно от проверяемого
payload: context, snapshot bindings, ожидаемые training/deploy row hashes.
Проверяется revision каждой строки, universe и origin, точные hashes строк,
payloads, actual selected scaler-input feature rows, scaler payload, model payload
и decision model. Самостоятельного self-hash недостаточно для смены pin identity.

Binding содержит file/content/grid/timestamp/feature-definition/label-definition/
universe SHA. Для fixture file SHA соответствует точным bytes опубликованного
`synthetic_snapshot.json`. Pins — доверенный вход caller, не магическое доказательство
источника. Grid SHA относится к таблице искусственных labels, не полному OHLCV grid.
Real revision provenance и feature calculation здесь не аттестованы.

Training rows упорядочены и уникальны; на каждом training origin есть весь
фиксированный universe, origins идут без пропусков/overlap по 3d. Неполные groups,
NaN/nonfinite, нетипизированные proposal, invalid scaler denominator и неизвестные
поля отклоняются. Лимиты — 512 training rows и 24 synthetic symbols.

## Проверенная fixture

Два `SYN_*` symbols, четыре completed training origins: 8 строк, из них
4 отрицательных и 4 положительных искусственных outcomes. Deploy — 2 строки.
Это логические синтетические timestamps, не исследуемый рыночный период.
Рыночных цен нет. Scaler median/IQR и 3 coefficients заданы вручную;
`manual_fixture_not_fit` обязательно, fits=0. 8 строк не означают выполнение
minimum opportunities=64 или ESS gate из draft. Прибыльность не рассчитывается.

80 direct metadata assertions прошли; отдельно 21 artifact assertion прошёл.
Тестируется отказ на будущих labels/features, cutoff от deployment вместо asof,
revision mismatch, смене pinned rows, несовпадении model/scaler hashes,
model availability после решения, extra columns, missing/duplicate groups,
malformed JSON и invalid scalars. Constructor isolation — не real feature
prefix-invariance test; latter остаётся отдельной задачей.

## Публикация и границы

`Reports/Exia/Genetic_Primus/causal_origin_payload_v1_20261007`:
`synthetic_snapshot.json`, `fixture_pins.json`, `payloads.json`,
`guard_result.json`, `summary.json`, `completion_receipt.json`.
Публикация exclusive, без перезаписи; receipt связывает 23 source/metadata inputs
и пять outputs. Проверен explicit input bank предыдущего static audit, включая
прежние contracts. Global historical inventory и native dependency graph заново
не аттестованы. Markdown поясняет, но не входит в JSON receipt.

Запускались только `.venv/Scripts/python.exe -X utf8`, по одной короткой команде:
py_compile одного файла, pure SHA check, bounded builder preflight, direct checker,
publication и artifact checker. Pytest, collector, сеть, package `exia`, numpy,
pandas, WebSocket, training и runtime не запускались/не импортировались.

Actual feature causality, scaler fitting, training latency, revision provenance,
market grid, entry-model accounting integration остаются false/unverified.
Все authority flags false; market research launch и real-money readiness false.

## Следующий шаг

Bounded as-of feature builder и prefix-invariance проверки на искусственных
свечах: добавление/изменение данных после decision-asof не должно менять прежний
deploy payload; отдельно пропуски и revisions. До этого не подключать новый guard
к legacy examples для запуска обучения и не открывать sealed interval.
