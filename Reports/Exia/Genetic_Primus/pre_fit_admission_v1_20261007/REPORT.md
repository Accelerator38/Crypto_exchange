# Итоги pre-fit admission — 7 октября 2026

Реализован fail-closed pre-fit plan с opportunity-filtered scaler scope,
uniform/recency weights, sample/ESS и declared timing gates. Два варианта
**отклонены**. Это ожидаемый инженерный результат, а не неудача обученной модели.

## Результаты

| Вариант | Opportunities / minimum | Kish ESS / minimum | Loss weight mass | Fit |
|---|---|---|---:|---|
| Uniform | 7 / 64 | 7.0 / 40 | 28.5714% | denied |
| Half-life 90d, anchor 20% | 7 / 64 | 6.996668 / 40 | 28.8728% | denied |

Все 7 opportunities, включая 2 убыточных, сохраняются в weight bank. Sign-flip
контроль подтвердил: recency weight не зависит от того, выигрышный outcome или
убыточный. Scaler-input SHA пересчитан по opportunity rows, а не по всем 8 labels.

66 direct assertions и 13 artifact assertions прошли. Rational weight/ESS
control и numerical underflow control прошли. 64 equal-age toy rows — проверка
арифметики, не способ увеличить реальную выборку и не разрешение на fit.

Kish ESS измеряет веса, не независимость рынка. Здесь не считались bootstrap,
LCB, DSR/PBO, PnL, fitness и модельные rankings. Fits / market evaluations — 0 / 0.

## Явные причины запрета

Кроме недостатка n и ESS: proposal остаётся draft, training authority false,
данные artificial, actual provenance/latency не аттестованы; training budget и
execution delay null; execution-aligned label и normal/stress cost contract
не зарегистрированы. Неизвестные значения не заменены нулями.

NoTrade fallback — название административного no-fit/no-orders решения. Runtime
не получает команд; позиции, ордера, API keys и текущие реальные агенты не затронуты.
Прохождение отдельных численных unit gates не открывает authority.

## Артефакты и scope

Опубликованы `admission.json`, `summary.json`, `completion_receipt.json`;
59 source/metadata input bindings и два output bindings проверены. Snapshot
training/deploy остаются раздельными. Explicit previous contracts/source bank
unchanged; historical market inventory и native dependency graph целиком
заново не аттестованы. Source feature/label builders заново не запускались.

Нет обучения/scaler fitting, pytest, collector, сети, runtime, paper/live,
promotion, financial evaluation, commit/push. Все authority flags, market launch
и real-money readiness false.

## Дальше

До нового fit требуется явный timing/execution/data research contract с
проверенными bindings. Не снижать thresholds под 7 artificial opportunities
и не открывать sealed interval. Завершённый admission — диагностический план,
не production authorizer и не доказательство преимущества над NoTrade.

[Методика и API](E:/Crypto_exchange/docs/GENETIC_PRIMUS_PRE_FIT_ADMISSION_V1.md).
