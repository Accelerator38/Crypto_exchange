# Итоги completed labels и origin integration — 7 октября 2026

Текущий результат — v2: training labels и deploy features используют один
feature kernel, но **независимые snapshots/hashes**. Training можно построить
без deploy данных. Обучения, market evaluation и trading не было.

## Проверенный результат

| Проверка | Результат |
|---|---:|
| V2 direct assertions | 63 passed |
| V2 artifact assertions | 21 passed |
| Labels против Fraction oracle | 8 |
| Max absolute error / tolerance | 9.71e-17 / 1e-12 |
| Completed training origins / rows | 4 / 8 |
| Отрицательные / положительные labels | 2 / 6 |
| Proposal opportunities | 7, включая 2 убыточных |
| Training / deploy source candles | 244 / 278 |
| Source/metadata input bindings | 50 |
| Fits / market evaluations | 0 / 0 |

Все candles, prices и timestamps искусственные. Gross labels — open-to-last-close
на 3-дневном origin, не net PnL и не результаты торгов. Assertions не являются
статистическими trials.

## Что исправлено и подтверждено

Cutoff вычисляется от training-asof с purge 7 четырёхчасовых bars. Label доступен
только после конца пути и доступности всех 18 candles, до cutoff. Исторические
features строятся по собственному asof, тем же kernel, что deploy. Поздние
свечи нельзя использовать задним числом. Все signs outcomes сохраняются.

При дополнительной проверке v1 обнаружилась metadata leakage: training hash
ссылался на общий snapshot с deploy prices после training-asof. V1 сохранена
как диагностическая; её 95 прошедших проверок недостаточны для causal identity.

V2 имеет отдельный training-only request без deploy source pin и два независимых
source snapshots. Полный training record/hash неизменен при изменении видимых
к deployment, но будущих к training-asof цен. Missing deploy candle не мешает
standalone training builder. В checker есть negative control, воспроизводящий
дефект v1, и отдельная проверка, что v2 его устраняет.

## Незакрытые условия

Minimum opportunities=64 не выполнен: здесь 7. ESS gate не оценён, actual scaler
и coefficients не fitted. Подготовленный scaler-input — 7 opportunity rows;
old guard hash относится ко всем 8 training rows. Эти scopes не смешиваются,
full old model bundle compatibility=false.

Модель, accounting/entry SHA, real execution costs и labels, real provenance и
measured latency не проверены. Синтетический offset 2500 ms не является реальным
latency budget. Новый flat-path abstention policy не выдаётся за legacy equivalence.
Market launch/real-money readiness и все authority flags остаются false.

Рыночные данные не читались; pytest, collector, сеть, runtime, API keys и торги
не запускались. Frozen contracts и previous explicit bank unchanged; historical
inventory/native graph целиком не переаттестованы. Commit/push не выполнялись.

## Следующий шаг

Pre-fit admission adapter: корректный opportunity-filtered scaler scope,
sample/timing gates и явные fail-closed причины. На этой fixture ожидается
запрет обучения, а не попытка создать сделки ради minimum trades. Новый sealed
interval и новые market runs на этом шаге не нужны.

[Методика и API](E:/Crypto_exchange/docs/GENETIC_PRIMUS_CAUSAL_TRAINING_LABELS_V2.md).
Рядом: `synthetic_candles.json`, `training_snapshot.json`, `deploy_snapshot.json`,
`origin_inputs.json`, `summary.json`, `completion_receipt.json`.
