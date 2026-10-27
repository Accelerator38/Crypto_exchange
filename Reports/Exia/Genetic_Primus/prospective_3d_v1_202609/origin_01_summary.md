# Prospective 3-day origin 01

Период: `[2026-09-01 00:00, 2026-09-04 00:00) UTC`. Расчёт выполнен offline после закрытия периода. Genetic search не запускался (`search_evaluations = 0`).

## Данные

- Публичный источник: Bitget USDT-FUTURES historical candles, без credentials.
- Новый участок: `[2026-08-18 00:00, 2026-09-04 00:00) UTC`.
- 1m archive: 195 840 строк, 16/16 партиций, dataset SHA `404819685854211bdee30f228800347c2f0acb433f7521bee9a03bf0df0beae1`.
- 1m full8 panel SHA: `e5fc664d6917c8cc9eada16781486352436d8e5576f2c514988dd962c79dc463`.
- Новый 4h parquet: 816 строк, 102 свечи на символ, SHA `7e11fdc0a30f172986b68325719bbdf2b7709845fad874f0d22afd885e814fa0`.
- Immutable manifest SHA: `6fe51ca8fd54e16f6baf2e21c04f2b129ea2abd5739df8ed3f08f0b8f9f68f7b`.
- Составной snapshot SHA: `033354efce9aa6a6ceab85e9249fa3b016e7e8dd70765f5d957d43843076e654`.

## Результат

| Candidate | Normal return | Stress return | Normal max DD | Stress max DD | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| NoTrade | 0.000% | 0.000% | 0.000% | 0.000% | 0 |
| EMA TrendConsensus | -1.991% | -2.197% | 2.149% | 2.311% | 21 |
| GA overlay `f7a4ae11…` | -0.525% | -0.604% | 0.900% | 0.964% | 8 |

Paired daily sum overlay минус EMA: `+1.474%` normal и `+1.602%` stress. Paired daily sum overlay минус NoTrade: `-0.525%` normal и `-0.604%` stress. Сумма paired daily returns может немного отличаться от разности compounded returns.

## Решение

GA overlay снизил turnover, потери и drawdown относительно EMA, то есть abstention сработал полезно. Но абсолютная доходность отрицательна в обоих cost-сценариях, поэтому risk contrast против NoTrade не пройден. Promotion отсутствует; incumbent остаётся `NoTrade`.

Это только первый из десяти зарегистрированных origins. Параметры, candidate set и gates после просмотра результата не меняются. Итоговый HAC и остальные promotion-метрики допустимо считать только после завершения всего 30-дневного outer-интервала.

Result SHA: `4e2b05988cdc8224eee370658eaf5e7a64db60ef18f260a0936da5e29d2ff6f8`.

Safety: credentials/orders/paper/live/promotion/runtime/network — `false`.
