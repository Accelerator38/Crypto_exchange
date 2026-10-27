# Genetic Primus: revealed August diagnostic

Расчёт зафиксирован 4 сентября 2026 как retrospective engineering diagnostic. Августовские данные были уже раскрыты до регистрации objective, поэтому результат не является validation, audit или основанием для promotion. После просмотра результатов параметры и candidate set не изменялись.

## Контракт

- Десять неперекрывающихся origins по три дня: `[2026-08-01, 2026-08-31)` UTC.
- Expanding train, purge 7 четырёхчасовых баров, scaler refit на каждом origin.
- Один frozen challenger: genome `[0, 0, 1, 1, -1]` поверх EMA TrendConsensus.
- Сравнения: `NoTrade`, прямой EMA proposal, GA overlay.
- Normal/stress round-trip costs: 16/24 bps.
- Search evaluations: 0; promotion authority: false.

## Результат за 30 дней

| Candidate | Normal return | Stress return | Normal DD | Stress DD | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoTrade | 0.000% | 0.000% | 0.000% | 0.000% | 0 |
| EMA TrendConsensus | +11.001% | +9.656% | 6.575% | 6.872% | 122 |
| GA overlay | **+15.645%** | **+14.931%** | **4.405%** | **4.615%** | **62** |

Overlay превзошёл EMA в 9 из 10 origins и примерно вдвое сократил сделки. Однако он был прибыльным относительно NoTrade только в 4 из 10 origins. Большая часть месячного результата пришлась на origin 7, где EMA дал +22.143%, а overlay +21.217%. Поэтому высокий compounded return не является устойчивой оценкой среднего эффекта.

## По окнам, normal costs

| Origin | EMA | GA overlay | EMA trades | GA trades |
| ---: | ---: | ---: | ---: | ---: |
| 1 | +0.486% | +1.106% | 11 | 4 |
| 2 | -1.019% | -0.943% | 16 | 9 |
| 3 | -0.728% | -0.214% | 15 | 4 |
| 4 | -0.948% | -0.619% | 13 | 6 |
| 5 | +0.542% | +0.733% | 13 | 2 |
| 6 | -1.077% | -0.248% | 12 | 5 |
| 7 | +22.143% | +21.217% | 10 | 10 |
| 8 | -2.565% | -2.347% | 12 | 11 |
| 9 | -0.279% | +0.096% | 6 | 5 |
| 10 | -3.847% | -2.201% | 14 | 6 |

## Paired HAC

| Contrast | Mean, bps/day | SE, bps/day | One-sided LCB, bps/day |
| --- | ---: | ---: | ---: |
| Overlay − EMA, normal | +13.41 | 5.45 | **-10.04** |
| Overlay − EMA, stress | +15.40 | 5.59 | **-8.64** |
| Overlay − NoTrade, normal | +51.01 | 52.20 | **-173.60** |
| Overlay − NoTrade, stress | +48.93 | 51.95 | **-174.62** |

Все LCB отрицательны. Наблюдаемый средний incremental effect положителен, но статистически надёжное преимущество не подтверждено. Особенно слаб контраст с NoTrade из-за концентрации результата в одном импульсном окне.

## Сопоставление с первым prospective origin

В действительно зафиксированном до данных окне `[2026-09-01, 2026-09-04)` EMA получил -1.991%, GA overlay -0.525%, NoTrade 0%. Картина согласуется с августом в одном отношении: abstention обычно уменьшает ущерб EMA. Но GA всё ещё способен оставаться убыточным, и prospective evidence пока состоит лишь из одного origin.

## Вывод

Overlay выглядит полезным **risk filter для EMA**, а не доказанной самостоятельной alpha-моделью. Его наиболее убедительный эффект — меньше turnover и drawdown. Целевой тест остаётся неизменным: завершить все десять сентябрьских prospective origins без настройки. До положительных LCB против EMA и NoTrade incumbent остаётся `NoTrade`.

Summary SHA: `6b17438beea6addc5d80a759f430436c389849dd52a1449feed53e52af94e3ff`.

Completion SHA: `e07f6877c9712c3bc9c9bbcc9406c0d2655a389277cfa196868ff781fe7487f0` (23 pinned machine artifacts).
