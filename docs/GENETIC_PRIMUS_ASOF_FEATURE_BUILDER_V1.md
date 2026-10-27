# As-of feature builder v1 — 7 октября 2026

Изолированный stdlib builder признаков на искусственных OHLCV, без модели,
labels, fitting и execution. Старые evaluator/feature modules не изменены
и не исполнялись. Исследовательский контракт остаётся draft, не launch-ready.

## Реализация

- `src/exia/genetic_primus/asof_feature_builder_v1.py`: strict request,
  выбор доступного prefix, exact grid и revision/prefix pins, вычисление признаков.
- `tools/build_genetic_primus_asof_features_v1.py`: five registered artificial
  cases, row-schema compatibility с прежним causal guard, exclusive publication.
- `tools/check_genetic_primus_asof_features_v1.py`: metamorphic checks,
  negative controls, независимый рациональный oracle и artifact validation.

Публичный `build_asof_features(candles, request)` не принимает training labels,
model, target либо terminal prices deployment окна.

## Причинный cutoff

При 4h bars `feature_end = floor(decision_asof / bar_duration) * bar_duration`.
Сохраняются только candles из pinned history interval до feature_end, закрытые
к этому времени и доступные не позже decision-asof. Поздняя revision старого бара
тоже исключается по её собственному availability timestamp. Нужен versioned
archive: поздняя запись не должна уничтожать исходную доступную версию.

До вычислений проверяются exact fixed-universe grid, отсутствие duplicates,
OHLCV domains, revision ID и SHA **сохранённого** prefix. Missing или delayed
последняя свеча дают отказ — builder не подменяет её более старой. Доступная
revision с другим ID отвергается; price mutation с прежним ID отвергается
against immutable prefix pin. Caller pins — доверенный внешний ввод, не доказательство
биржевого происхождения или полноты истории.

Envelope всех input rows должен быть корректным, а объём bounded. Future OHLCV
не участвуют ни в индикаторах, ни в результатных hashes. Даже nonfinite значение
в невидимой будущей цене не читается при вычислении; это не означает проверки
качества всего future dataset. Invalid envelopes и превышение общего cap
по-прежнему отклоняются. Более ранние rows до pinned history start исключаются.

## Kernel definition

- `return_12 = close[t] / close[t-12] - 1`.
- `efficiency_12 = abs(close[t] - close[t-12]) / sum(abs(diff(last 13 closes)))`.
- EMA12/48 seeded первым close fixed history, recursive adjust=false.
- Proposal long: EMA gap > 0, EMA12 three-bar slope > 0, return12 > 0,
  efficiency12 >= 0.30.

При нулевом ценовом пути return12=0, efficiency12=0, proposal=false. Это
**новая явно описанная engineering policy**, не эквивалентность прежнему
pandas NaN/drop поведению. Clamp efficiency к 1 допустим только в пределах
1e-12 floating roundoff; более крупное нарушение отвергается.

Minimum history — 48 bars. Seed/history start фиксируется request и должен
оставаться неизменным при продвижении asof, иначе EMA определяется иначе.
Применение этой bounded warmup policy к полноценной market history не проверено.
Лимиты: 1024 input rows, 512 retained rows, до 4 synthetic symbols.

Derived `feature_available_at` равен объявленному decision-asof. Это синтетическая
metadata convention, **не** измеренная compute latency. Происхождение timestamps,
revisions и показателей реального источника остаётся unverified.

## Проверенный результат

126 direct assertions passed. Metamorphic проверки: prefix-only против rows
с будущим хвостом; reverse order; изменение future OHLCV; late revision старого
бара; четыре дополнительных pattern prefixes. Полный result/hash прежнего asof
сохранился. Отдельный control показывает, что принятый новый видимый snapshot
меняет features; advancing asof действительно включает новую доступную свечу.

Пять artificial cases: mixed rising/falling, rising, falling, flat, alternating.
Два symbols и 48 retained bars каждый; основной input содержит 96 retained и
16 future candles. 40 numeric indicator fields сверены с exact Fraction oracle:
max absolute error 1.54e-15, tolerance 1e-12. Proposal/flat flags проверены отдельно.
Oracle — проверка engineering formulas на этих fixtures, не запуск исторического
pandas evaluator и не доказательство эквивалентности всех индикаторов/edge cases.

Отдельно 21 artifact assertion passed. Source/code/metadata inputs — 32 files,
пять JSON outputs и completion receipt. Predecessor input bank, включая прежние
contracts, unchanged. Historical inventory и native dependency graph целиком
заново не аттестованы. Никаких статистических финансовых trials в этих counts нет.

## Публикация

`Reports/Exia/Genetic_Primus/asof_features_v1_20261007`:
`synthetic_candles.json`, `kernel_definition.json`, `feature_candidate.json`,
`pattern_results.json`, `summary.json`, `completion_receipt.json`.
Exclusive publication, no overwrite; Markdown не часть JSON receipt.

С прежним guard проверена только deploy **row schema**, не full trained bundle:
нельзя связать старый manual scaler/model с новым вычисленным feature definition
и объявить это обучением. Candidate не содержит решений о торгах/model SHA.
Все credentials/orders/execution/paper/live/promotion/training/financial_evaluation/
network flags false; market research launch и real-money readiness false.
Python только `E:/Crypto_exchange/.venv/Scripts/python.exe -X utf8`;
pytest, collector, сеть, numpy/pandas, package `exia`, training и runtime не запускались.

## Следующий шаг

As-of builder завершённых training labels и синтетическая origin integration:
label end/availability, train-asof purge, causal deploy features, согласованные
feature definitions и hashes. Сначала без fitting. Нельзя считать этот этап
основанием нового market run, silent objective change или открытия sealed interval.
