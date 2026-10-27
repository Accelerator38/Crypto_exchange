# Genetic Primus — V7 proposal matrix

Дата: 2026-08-28.

## Решение

Ни одна из четырёх заранее зарегистрированных fixed opportunity hypotheses не выбрана как основа V7 genetic overlay. Genetic training не запускался. Результаты являются только development evidence на уже раскрытом интервале `[2024-08-01, 2026-07-01)` и не имеют promotion authority.

Матрица была зафиксирована до вычисления returns: четыре proposals, по одному варианту параметров, 233 трёхдневных origins, base cost 16 bps, stress cost 24 bps, четыре календарных блока и независимый audit bootstrap seed.

## Результаты

| Policy | Base compound | Stress compound | Max DD | Trades/day | Audit LCB vs NoTrade | Positive blocks |
|---|---:|---:|---:|---:|---:|---:|
| NoTrade | 0.00% | 0.00% | 0.00% | 0.00 | 0.00 bps | 0/4 |
| Momentum_return_1 | −92.57% | −98.67% | 92.64% | 24.42 | −145.08 bps | 0/4 |
| RelativeMomentumRank_return_12 | −39.85% | −58.13% | 55.65% | 5.18 | −37.17 bps | 1/4 |
| V7_EMA_TrendConsensus | +21.60% | −7.32% | 35.42% | 3.88 | −17.45 bps | 2/4 |
| V7_ResidualRankConsensus | −29.22% | −46.16% | 45.53% | 3.91 | −27.34 bps | 1/4 |
| V7_DonchianBreakout | −12.06% | −29.90% | 20.68% | 3.24 | −23.36 bps | 1/4 |
| V7_MarketAlignedTrend | +21.22% | −16.90% | 34.65% | 5.39 | −17.03 bps | 2/4 |

## Temporal stability

Доходность EMA consensus по блокам: `+61.96%, −14.62%, −14.54%, +2.89%`.

Доходность market-aligned trend: `+53.58%, −7.71%, −19.20%, +5.85%`.

Обе положительные по полному периоду модели зависят главным образом от первого блока 2024 года и отрицательны в двух следующих режимах. Поэтому выбирать их по aggregate compound было бы temporal overfitting.

## Почему никто не выбран

EMA consensus провалил четыре gate: менее трёх положительных блоков, отрицательный paired audit LCB, отрицательный stress return и drawdown выше 10%.

Market-aligned trend провалил те же четыре gate и дополнительно turnover gate. Residual-rank и Donchian также имеют отрицательную full-period доходность.

Основная общая проблема — семантика исполнения. Policies пересматривают target каждые четыре часа. Даже отфильтрованные EMA/market proposals делают 3.9–5.4 сделки в день. Дополнительные 8 bps round-trip cost превращают base-прибыль в stress-убыток. Это не исправляется безопасным подбором ещё одного deadband после просмотра outcomes.

## Следующий исследовательский шаг

Не запускать genetic overlay поверх любой из этих четырёх policies. Следующая материально новая гипотеза должна быть origin-locked:

- одно решение на начало трёхдневного deployment origin;
- затем удержание позиции до конца origin либо заранее зарегистрированный risk exit;
- genetic model может только разрешить или отклонить origin-level opportunity;
- direction задаётся фиксированным trend consensus;
- turnover budget измеряется на origin, а не на каждом 4h баре;
- разработка только на уже раскрытых данных;
- следующий sealed evaluation — только на будущем интервале после 2026-07-01.

Это изменение проверяет гипотезу о соответствии execution cadence целевому deployment horizon, а не подгоняет пороги неудачных 4h policies.
