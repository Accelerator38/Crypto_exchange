# Exia 4h market-quorum screen v1

## Цель

Проверить восемь заранее зафиксированных 4h-сигналов без изменения основного
таймфрейма и без доступа к прежнему prospective holdout. В семейство вошли
EMA trend, Donchian breakout и relative momentum с одним общим трехрежимным
market quorum: `bullish`, `bearish`, `neutral`. Neutral является NoTrade.

Переход между окнами был последовательным: development -> validation -> OOS
-> sanity. Для перехода требовались минимум 20 сделок и положительные stress
mean, bootstrap LCB и family-wise LCB. Лимиты открытия следующих окон: 4, 2 и
1 кандидат соответственно.

## Результат development

| Кандидат | Сделки | Stress mean, bps | LCB, bps | Family-wise LCB, bps |
|---|---:|---:|---:|---:|
| MQ_EMA12_48_LONG | 875 | 34.98 | -6.73 | -28.63 |
| MQ_EMA12_48_BOTH | 1875 | 19.71 | -16.16 | -34.73 |
| MQ_REL_MOM24 | 1484 | 5.22 | -20.05 | -32.03 |
| MQ_REL_MOM12 | 1885 | -4.01 | -22.05 | -31.23 |
| MQ_DONCHIAN20_EMA48 | 1300 | 15.30 | -26.81 | -46.97 |
| MQ_DONCHIAN55_EMA96 | 794 | 32.18 | -33.73 | -65.08 |
| MQ_EMA24_96_BOTH | 1245 | 19.26 | -36.22 | -64.19 |
| MQ_EMA12_48_SHORT | 1000 | 6.35 | -51.01 | -76.63 |

Ни один кандидат не прошел development gate. Поэтому validation, OOS и
sanity не рассчитывались. Старый prospective holdout также не использовался.

## Корень проблемы

У всех кандидатов отрицательная медианная сделка: от `-39.63` до `-97.86`
bps. Положительный mean у части вариантов создают редкие трендовые выигрыши до
`+11998 bps`, тогда как P01 находится примерно между `-709` и `-1014` bps.
Это асимметричная стратегия с редкими большими выигрышами, но слишком частыми
убыточными входами и кластеризованным downside.

Market quorum уменьшил бессмысленную торговлю в neutral, но сам по себе не
исправил распределение сделок. Продолжать подбор EMA, Donchian или breadth
threshold по этому же development означает увеличивать overfit.

## Следующий эксперимент

Следующий семейный тест должен изолировать только position policy:

- вход фиксируется как `MQ_EMA12_48_LONG`, у которого лучший development LCB;
- параметры входа и market regime больше не изменяются;
- максимум четыре заранее зафиксированных exit-policy;
- stop/trailing/time-exit считаются по OHLC 4h и ATR, а не по close-only target;
- при одновременном касании stop и target используется консервативный stop-first;
- development используется для выбора максимум одной policy;
- validation остается закрытым до выбора;
- OOS/sanity открываются только после положительного validation LCB;
- прежний prospective holdout запрещен для настройки.

Цель exit-policy не максимизировать mean. Она должна сделать медиану сделки
положительной, уменьшить P01 и получить положительные обычный и family-wise
LCB при stress cost 16 bps.

Авторитетные артефакты находятся в
`Reports/Exia/four_hour_candidate_screen_v1_final`. Повторный прогон
`four_hour_candidate_screen_v1_repeat` совпал по всем четырем parquet-таблицам.
Все paper/live/orders/promotion флаги равны false.
