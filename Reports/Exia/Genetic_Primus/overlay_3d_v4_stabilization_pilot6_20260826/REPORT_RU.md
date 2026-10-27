# Genetic Primus overlay v4 — stabilization pilot6

Дата: 2026-08-26. Режим: offline engineering. Сеть, collector, supervisor, paper/live runtime и ордера не запускались.

## Изменение objective относительно v3

- Half-life 90 дней с 20% uniform anchor вместо uniform weighting.
- Максимум 6 активных features вместо 8.
- Momentum opportunity threshold: 24 bps вместо 16 bps.
- ALLOW coverage: 3–25% вместо 5–35%.
- Coverage penalty: 80 bps вместо 40 bps.
- Turnover penalty: 2.0 вместо 0.5 bps на fill/day.
- Action-instability penalty: 40 вместо 20 bps.
- Search budget не изменён: максимум 107 гипотез на origin, 5 islands × population 4 × 5 generations.

## Проверки

- `py_compile` для `rolling.py` и `rolling_v4.py`: успешно.
- Две новые v4 deterministic regressions: passed.
- Четыре прежние v3 regressions: passed.
- Один v4 smoke-origin: exit code 0.
- Шесть v4 origins: exit code 0.

## Smoke origin 0: v4 против v3

| Метрика | v3 | v4 |
|---|---:|---:|
| Deployment net, bps | +80.599 | +418.542 |
| Stress net, bps | +39.311 | +399.855 |
| Trades | 41 | 18 |
| Max DD | 1.669% | 0.620% |
| Validation paired LCB, bps | -35.988 | -16.272 |
| Audit paired LCB, bps | -82.986 | -18.156 |
| Active features | 8 | 1 |

Smoke дал достаточное инженерное улучшение для одного заранее ограниченного pilot6, но predeployment gate остался false.

## Результаты pilot6

| Агент | Сумма, bps | Положительных origins | Trades | Compounded, bps | Aggregate max DD |
|---|---:|---:|---:|---:|---:|
| GeneticPrimusChallenger v4 | +373.909 | 3/6 | 96 | +371.102 | 3.585% |
| Challenger stress | +277.200 | 3/6 | 96 | +271.945 | 4.190% |
| Momentum return_1 | +117.047 | 3/6 | 424 | +91.629 | 11.610% |
| NoTrade / active | 0.000 | 0/6 | 0 | 0.000 | 0.000% |
| Mean reversion return_1 | -1874.407 | 0/6 | 424 | -1754.088 | 17.490% |

Challenger v4 по origins, bps: `+418.542, ~0.000, -118.689, +8.595, -10.430, +75.892`.

## v4 против v3

- Сумма deployment: `+373.909` против `+205.153 bps`.
- Stress: `+277.200` против `+60.482 bps`.
- Trades: 96 против 144.
- Положительные origins: 3/6 против 5/6.
- Средний validation LCB: `-12.530` против `-13.346 bps`.
- Средний independent-audit LCB: `-39.497` против `-57.866 bps`.
- Средний audit turnover: 11.87 против 12.52 fills/day.
- Среднее число активных features: 3.33 против 5.50.
- Action drift: 0.235 против 0.507 — теперь проходит лимит 0.25.
- Normalized model drift: 2.964 против 2.483 — ухудшился и значительно выше лимита 0.75.

v4 лучше по PnL, stress, sparsity, turnover и action drift, но его aggregate PnL сильно зависит от origin 0. Устойчивость по доле origins и outer bootstrap ухудшилась.

## Promotion gate

- Outer paired LCB: `-7.439 bps`.
- Independent-seed outer audit LCB: `-7.439 bps`.
- Stress LCB: `-20.442 bps`.
- Positive outer fraction: 50% при минимуме 65%.
- Fills/day: 10.667 — turnover gate пройден.
- Action total variation: 0.235 — action-drift gate пройден.
- Predeployment temporal gates: false на всех 6 origins.
- Также не пройдены minimum origins, drawdown-vs-incumbent, direction/state concentration и model drift.

Итог: `promotion_gate_passed=false`, `promotion_approved=false`, `promotion_authority=false`. Новый sealed outer interval не открывается.

## Контроль adaptive multiple-testing

v4 является одной дополнительной заранее зафиксированной stabilization-variant на уже открытом engineering-периоде. После просмотра pilot6 параметры больше не корректировались и новые варианты не запускались. Эти результаты не могут считаться sealed prospective evidence.

## Воспроизводимость и safety

- Dataset SHA-256: `f66529b4135d04ddc35b7a6535667d1c6d53cd39c34adb86f23e54d4082b2de7`.
- Rolling protocol SHA-256: `5477b2a03eb60089799a91dc26fb8d2e1800ced3ad5ba132958b4d05e6e857c1`.
- Последний challenger model SHA-256: `cf3dc866bc491e665c98190c09865302d84becf7ebc42ff9f7130baaaf4f09c6`.
- `credentials_allowed=false`, `paper_allowed=false`, `live_allowed=false`, `orders_enabled=false`, `promotion_authority=false`.

## Следующее изменение до нового обучения

Не подстраивать ещё один objective на январском pilot. Следующий кодовый этап должен быть измерительным:

1. Ввести явный статус `NO_ELIGIBLE_CHALLENGER`, если ни один validation-finalist не проходит structural/coverage/paired gates.
2. Не интерпретировать диагностический raw challenger как deployable candidate при этом статусе.
3. Заменить parameter-centroid drift функциональным logit/action drift на фиксированном hash-anchored наборе, поскольку sparse linear genomes могут иметь большой parameter drift при близких действиях.
4. Сохранять paired LCB и structural metrics для каждого top-K finalist, чтобы видеть полный Pareto frontier без повторного обучения.

После этих изменений можно сформировать один окончательный контракт следующего objective до открытия нового периода данных.
