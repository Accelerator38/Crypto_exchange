# Genetic Primus V8 Clean — Stage 1 outer6

Дата фиксации: 2026-08-26. Статус: только offline engineering; не является promotion evidence для live runtime.

## Safety и воспроизводимость

- `credentials_allowed=false`
- `paper_allowed=false`
- `live_allowed=false`
- `orders_enabled=false`
- `promotion_authority=false`
- Во всех пяти ячейках завершено по 6 последовательных origins, из них 5 incumbent-comparable.
- Для каждого решения заполнен `model_sha256`; SHA присутствует в соответствующем model ledger.
- Purge между `train_end` и label cutoff равен 28 часам, то есть 7 барам по 4 часа.
- Challenger model SHA и scaler SHA пересчитываются на каждом origin.
- Dataset/config/protocol и все итоговые parquet закреплены SHA256 в manifests и aggregate summary.

Первый запуск `expanding_uniform` завершился native access violation в Pandas `DataFrame.iloc` внутри горячего цикла simulator. Scalar Series indexing заменён на заранее скопированные NumPy arrays без изменения next-open, cost или ledger semantics. Прямая синтетическая проверка simulator прошла; затем все пять чистых outer6-ячеек завершились без crash. Аварийный неполный каталог сохранён отдельно и не смешан с результатами.

## Weekly outer6 ranking

| Rank | Variant | Audit paired LCB, bps | Positive outer slices | Stress-cost LCB, bps | Promotion pass |
|---:|---|---:|---:|---:|---:|
| 1 | `rolling_730_uniform` | -4.3124 | 38.33% | -4.6026 | 0% |
| 2 | `rolling_730_hl180` | -4.8255 | 31.67% | -5.2380 | 0% |
| 3 | `rolling_365_hl90` | -11.3947 | 51.67% | -9.8841 | 0% |
| 4 | `rolling_180_hl30` | -14.9189 | 61.67% | -14.6442 | 0% |
| 5 | `expanding_uniform` | -16.0960 | 23.33% | -15.2841 | 0% |

Provisional top-2: `rolling_730_uniform`, `rolling_730_hl180`. Все audit и stress lower bounds отрицательны; ни одна модель не допускается к promotion.

Короткие окна чаще дают положительные локальные slices, но проигрывают по нижней доверительной границе. Это соответствует повышенной variance и более низкому effective sample size при агрессивном recency weighting. На одинаковом 730-дневном окне uniform остаётся лучше half-life 180 по primary и stress LCB.

## Baselines на тех же шести origins

Сырые суммы не являются promotion metric, но полезны как диагностика.

| Variant / agent | Total net, bps | Worst origin, bps | Worst DD | Trades |
|---|---:|---:|---:|---:|
| 730 uniform challenger | -1074.16 | -1520.26 | 22.4% | 144 |
| 730 uniform active | -287.82 | -1085.15 | 20.9% | 135 |
| 730 uniform incumbent | -1005.72 | -1085.15 | 20.9% | 105 |
| 730 HL180 challenger | -226.54 | -760.02 | 15.3% | 140 |
| 730 HL180 active | -1170.74 | -1512.00 | 22.4% | 149 |
| 730 HL180 incumbent | -1798.86 | -1512.00 | 22.4% | 114 |
| Momentum return_1 | -1093.92 | -1300.69 | 17.2% | 993 |
| MeanReversion return_1 | -3079.30 | -1395.49 | 13.9% | 993 |
| NoTrade | 0.00 | 0.00 | 0.0% | 0 |

`rolling_730_hl180` challenger суммарно обошёл incumbent на 944.2 bps и momentum на 867.4 bps, но победил каждый из них только в 3 из 6 origins. Независимый paired audit LCB остался отрицательным из-за нестабильности и хвостовых потерь. Оба challenger проиграли NoTrade по суммарному результату.

## Promotion gate

- `rolling_730_uniform`: только origin 1 имел положительный audit LCB (`+0.856`), но stress LCB был `-0.033`, а positive-slice fraction 58.3%; promotion отклонён.
- На origins 2–5 `rolling_730_uniform` audit LCB был отрицательным.
- У `rolling_730_hl180` audit и stress LCB были отрицательными на всех пяти comparable origins.
- Основные блокирующие gates: paired selection/audit LCB, stress-cost LCB, positive-slice fraction и периодически drawdown versus incumbent.
- Action/model drift не был причиной отказа top-2, но прохождение drift gate само по себе не компенсирует отрицательную статистическую границу.

## Top-3 против top-5 ensemble

| Variant | Ensemble | Mean selection fitness | Mean audit fitness | Selected fraction |
|---|---:|---:|---:|---:|
| `rolling_730_uniform` | top-3 | -71.626 | -71.975 | 33.3% |
| `rolling_730_uniform` | top-5 | -74.367 | -74.679 | 66.7% |
| `rolling_730_hl180` | top-3 | -70.183 | -70.951 | 66.7% |
| `rolling_730_hl180` | top-5 | -75.229 | -75.829 | 33.3% |

Top-3 имеет лучшее среднее selection/audit fitness у обоих лидеров. Fitness сравнивается только внутри одного evaluator и не является доходностью.

## Почему cadence Stage 2 не запущен

Запечатанный validation interval содержит 52 weekly origins. Outer6 является ограниченным pilot, поэтому aggregate manifest фиксирует:

- `outer6_pilot_completed=true`;
- `full_stage1_completed=false`;
- `eligible_for_stage2=false`.

Запуск одинаковых шести origins для cadence 1/3/7/30 был бы методологически неверным: он охватывал бы соответственно 6, 18, 42 и 180 календарных дней. Это сравнило бы разные рыночные интервалы и создало ложный cadence winner. Stage 2 допускается только после полного Stage 1 либо после отдельной заранее зафиксированной схемы одинакового календарного cutoff.

## Итог

Outer6 подтверждает пользу строгого promotion gate, но не торговое преимущество genetic challenger. `rolling_730_uniform` остаётся первым по preregistered ranking, `rolling_730_hl180` вторым. Оба отклонены; NoTrade остаётся непревзойдённым по суммарному net result. Никакого live/paper promotion не выполнено.
