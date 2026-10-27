# Pre-fit admission v1 — 7 октября 2026

Реализован изолированный admission **plan** для проверенных synthetic origin
inputs v2. Он пересчитывает scaler-input scope, outcome weights и sample gates;
возвращает явные причины отказа. Fitter отсутствует, fit authority не открывается.
Это не production authorizer и не новая торговая стратегия.

## Файлы

- `src/exia/genetic_primus/pre_fit_origin_admission_v1.py`: pure stdlib validation,
  uniform/90d weight bank, Kish ESS, declared timing gate и denial plans.
- `tools/build_genetic_primus_pre_fit_admission_v1.py`: чтение существующих verified
  synthetic JSON, receipt/source bindings, exclusive publication.
- `tools/check_genetic_primus_pre_fit_admission_v1.py`: bounded direct regressions,
  rational weight controls и независимая проверка published artifacts.

Старые origin/label/feature builders заново не исполняются. Только pure payload
guard и новый admission helper загружаются через runpy, без package `exia`.
Предыдущий explicit metadata/source bank проверяется по bytes/SHA; это не
повторная аттестация historical market inventory или native runtime memory.

## Scaler-input scope

Training guard проверяет completed rows и их pinned identity. Затем admission
сам выбирает `proposal_long=true`, **не** фильтруя знак gross label. Scaler input —
два feature values каждой выбранной opportunity в стабильном training order.
Counts, vectors, labels signs и SHA пересчитываются; caller counters не доверяются.

Hash от всех training rows нельзя заменить им: в этой fixture 8 training rows
против 7 opportunities. Old guard scaler scope остаётся явно all-training-rows;
admission scope — opportunity-filtered. Существующий old model bundle не
объявляется совместимым. Median/IQR и coefficients здесь не fit.

## Outcome weights и ESS

Age считается от **training-asof** до label end, не от deployment и не от
знака прибыли. Завершённость/availability проверяются по purged cutoff.

- Uniform: `w_i = 1/n`.
- Half-life 90d: `w_i = 0.20/n + 0.80*r_i/sum(r)`,
  `r_i = 2^(-(age_i-min_age)/90)`.

Вычитание min-age сохраняет normalized ratios и предотвращает обнуление всех
exponents на очень старой искусственной истории. Каждый исход, включая убыток
и zero, получает положительный вес. Weight bank связывает ordered row IDs,
symbols, completed labels, asof, variant и weights SHA.

`Kish ESS = 1 / sum(w_i^2)`. Это effective count **весов**, не независимых
market observations: temporal/cross-symbol correlation здесь не оценивается.
Нет bootstrap, LCB, PnL, fitness ranking, multiple-testing correction или DSR/PBO.

Sample gates из pinned draft: opportunities >= 64 и weight ESS >= 40.
Minimum trades не используется для принуждения к сделкам.

## Timing и authority

Declared timing window — `decision_asof - training_asof`. Зарегистрированный
положительный training budget должен помещаться в него; execution delay должен
быть явно задан неотрицательным числом. Execution delay не вычитается из compute
window: он относится к последующему fill, а model должен быть доступен к decision.
Это проверка scheduled metadata, не измерение compute/model/fill availability.

В текущем draft оба budgets null; timing gate denied. Draft status, disabled
training authority, synthetic provenance, незакрытая actual latency,
execution-aligned labels и normal/stress-cost contract также явно перечислены
как blockers. Численное прохождение отдельного weight/timing unit control не
открывает fit. Все plans имеют `fit_allowed=false`.

Fallback `NoTrade_no_fit_no_orders_not_a_trading_decision` — административный
запрет работы fitter, не команда runtime и не изменение текущих позиций.

## Проверенный результат

66 direct assertions и 13 artifact assertions прошли. Проверены строгие types/
schemas, незаработанные gate flags, scope substitution, loss omission, будущие
weighted labels, веса независимо от outcome sign, rational 90-day control,
numerical underflow control и declared timing boundaries.

Для двух outcomes с возрастной разницей ровно 90 дней exact Fraction weights
19/30 и 11/30 сверены с результатом, включая ESS. 64 equal-age toy rows — только
положительный unit control весов, **не** новые реальные opportunities/model trials
и не разрешённая training dataset.

| Variant | Opportunities | Kish ESS | Weight mass отрицательных outcomes | Fit |
|---|---:|---:|---:|---|
| uniform | 7 | 7.0 | 0.2857142857142857 | denied |
| half-life 90d + anchor 20% | 7 | 6.9966680626302065 | 0.2887279779004711 | denied |

В обоих случаях n < 64, ESS < 40. Сходство ESS не говорит о качестве моделей;
они не обучались. Fits=0, market strategy evaluations=0.

## Публикация и следующий шаг

`Reports/Exia/Genetic_Primus/pre_fit_admission_v1_20261007`:
`admission.json`, `summary.json`, `completion_receipt.json`.
59 explicit source/metadata inputs, два outputs, exclusive write без перезаписи.
Markdown вне JSON receipt. Старые contracts/source bank unchanged.

Python только `E:/Crypto_exchange/.venv/Scripts/python.exe -X utf8`, одна короткая
команда за раз; без pytest/collector/network/runtime/API keys/paper/live/orders/
promotion/fitting/financial evaluation. NumPy/pandas и package `exia` не импортированы.
Market research launch и real-money readiness false; все authority flags false.

Дальше нужны явный отдельный timing/execution/data research contract и проверка
его bindings до fit. Не заменять неизвестные latency/costs нулями, не снижать
sample gates под эту fixture и не открывать sealed interval ради engineering
проверки. Этот admission plan сам по себе не является разрешением обучать модель.
