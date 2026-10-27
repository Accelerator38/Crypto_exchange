# Genetic_Primus V1-V3 vs standard 4h models

Статус: retrospective research comparison; paper/live authority отсутствует.

## Единый контракт

- Bitget full8, 4h, решения по закрытой свече, исполнение на следующем open.
- Stress cost: 16 bps round trip.
- Development: 2022-01-01..2023-12-31; validation: 2024-01-01..2024-12-31.
- Common LCB: moving-block bootstrap, 5000 samples, alpha=0.05.
- Dataset SHA: `f66529b4135d04ddc35b7a6535667d1c6d53cd39c34adb86f23e54d4082b2de7`.
- V1/V2 validation остаётся закрытой, потому что версии не прошли собственный development gate.
- OOS и sanity не открывались для этого сравнения.

## Development: три версии Primus

| Model | Trades | Mean | Median | Common LCB | Drawdown | Protocol status |
|---|---:|---:|---:|---:|---:|---|
| Genetic_Primus_V3 | 3480 | 62.86 | 39.07 | 46.05 | 16.41% | DEVELOPMENT_V3_PASSED_STRICT_GATES |
| Genetic_Primus_V2 | 3319 | 55.43 | 7.70 | 37.47 | 16.56% | DEVELOPMENT_V2_REJECTED_TEMPORAL_ROBUSTNESS |
| Genetic_Primus_V1 | 4272 | 23.36 | 7.65 | 8.73 | 19.84% | DEVELOPMENT_REJECTED_SLICE_GATES |

## Development: общий рейтинг

| Rank | Model | Family | Trades | Mean | Median | Common LCB | DD |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | Genetic_Primus_V3 | neurogenetic | 3480 | 62.86 | 39.07 | 46.05 | 16.41% |
| 2 | Genetic_Primus_V2 | neurogenetic | 3319 | 55.43 | 7.70 | 37.47 | 16.56% |
| 3 | Genetic_Primus_V1 | neurogenetic | 4272 | 23.36 | 7.65 | 8.73 | 19.84% |
| 4 | MQ_EMA12_48_LONG | standard_market_quorum | 875 | 34.98 | -52.01 | -9.96 | 17.06% |
| 5 | MQ_EMA12_48_BOTH | standard_market_quorum | 1875 | 19.71 | -54.02 | -16.41 | 36.15% |
| 6 | MQ_REL_MOM24 | standard_market_quorum | 1484 | 5.22 | -46.13 | -19.17 | 26.35% |
| 7 | MQ_REL_MOM12 | standard_market_quorum | 1885 | -4.01 | -39.63 | -21.59 | 27.48% |
| 8 | MQ_DONCHIAN20_EMA48 | standard_market_quorum | 1300 | 15.30 | -76.31 | -23.80 | 40.21% |
| 9 | MQ_DONCHIAN55_EMA96 | standard_market_quorum | 794 | 32.18 | -97.86 | -34.39 | 46.46% |
| 10 | MQ_EMA24_96_BOTH | standard_market_quorum | 1245 | 19.26 | -77.53 | -35.75 | 56.81% |
| 11 | MQ_EMA12_48_SHORT | standard_market_quorum | 1000 | 6.35 | -56.77 | -49.46 | 40.28% |
| 12 | CONTROL_ALWAYS_LONG | control | 8 | -3557.34 | -3946.31 | n/a | 71.54% |
| 12 | CONTROL_ALWAYS_SHORT | control | 8 | 3525.34 | 3914.31 | n/a | 78.55% |
| 12 | CONTROL_NO_TRADE | control | 0 | n/a | n/a | n/a | -0.00% |

## Validation diagnostic

| Model | Trades | Mean | Median | Common LCB | DD | Status |
|---|---:|---:|---:|---:|---:|---|
| MQ_REL_MOM24 | 759 | 30.79 | -30.21 | -2.25 | 10.88% | POST_REJECTION_VALIDATION_DIAGNOSTIC |
| MQ_EMA24_96_BOTH | 626 | 48.98 | -70.11 | -14.05 | 32.72% | POST_REJECTION_VALIDATION_DIAGNOSTIC |
| MQ_REL_MOM12 | 1026 | 13.56 | -37.01 | -15.03 | 21.45% | POST_REJECTION_VALIDATION_DIAGNOSTIC |
| MQ_EMA12_48_LONG | 516 | 70.03 | -58.45 | -21.06 | 21.44% | POST_REJECTION_VALIDATION_DIAGNOSTIC |
| MQ_DONCHIAN55_EMA96 | 414 | 45.14 | -62.68 | -35.59 | 31.36% | POST_REJECTION_VALIDATION_DIAGNOSTIC |
| Genetic_Primus_V3 | 1887 | -13.46 | 3.39 | -36.94 | 49.99% | VALIDATION_V3_REJECTED |
| MQ_EMA12_48_BOTH | 1005 | 12.23 | -68.94 | -45.26 | 39.32% | POST_REJECTION_VALIDATION_DIAGNOSTIC |
| MQ_DONCHIAN20_EMA48 | 705 | 11.20 | -100.89 | -52.95 | 31.50% | POST_REJECTION_VALIDATION_DIAGNOSTIC |
| MQ_EMA12_48_SHORT | 489 | -48.76 | -82.21 | -104.22 | 38.21% | POST_REJECTION_VALIDATION_DIAGNOSTIC |
| CONTROL_ALWAYS_LONG | 8 | 11827.24 | 10198.89 | n/a | 43.04% | POST_REJECTION_VALIDATION_DIAGNOSTIC |
| CONTROL_ALWAYS_SHORT | 8 | -11859.24 | -10230.89 | n/a | 78.65% | POST_REJECTION_VALIDATION_DIAGNOSTIC |
| CONTROL_NO_TRADE | 0 | n/a | n/a | n/a | -0.00% | POST_REJECTION_VALIDATION_DIAGNOSTIC |

## Вывод

- На development V3 сильнее лучшего стандартного правила `MQ_EMA12_48_LONG`: LCB 46.05 bps у лучшего Primus против -9.96 bps у standard baseline.
- Но V3 не переносится на validation: mean -13.46, median 3.39, LCB -36.94 bps, drawdown 49.99%.
- Следовательно, рост V1 -> V3 показывает улучшение оптимизации development, а не подтверждённое улучшение обобщения.
- Ни одна строка этого отчёта не открывает OOS, paper или live.

`player_one` и `player_two` находятся в XLSX как справочные результаты: их routing contract отличается, поэтому они не входят в основной rank.
