# Exia 4h market-quorum candidate screen v1

Verdict: `NO_4H_CANDIDATE_PASSED_STAGED_GATE`.

This is staged retrospective research. It grants no paper/live authority.

## Stage progression

- Development -> validation: none
- Validation -> OOS: none
- OOS -> sanity: none
- Final passed: none

## Evaluated windows

| Candidate | Window | Trades | Stress mean | LCB | Family-wise LCB | Gate |
|---|---|---:|---:|---:|---:|---|
| MQ_DONCHIAN20_EMA48 | development | 1300 | 15.30 | -26.81 | -46.97 | FAIL |
| MQ_DONCHIAN55_EMA96 | development | 794 | 32.18 | -33.73 | -65.08 | FAIL |
| MQ_EMA12_48_BOTH | development | 1875 | 19.71 | -16.16 | -34.73 | FAIL |
| MQ_EMA12_48_LONG | development | 875 | 34.98 | -6.73 | -28.63 | FAIL |
| MQ_EMA12_48_SHORT | development | 1000 | 6.35 | -51.01 | -76.63 | FAIL |
| MQ_EMA24_96_BOTH | development | 1245 | 19.26 | -36.22 | -64.19 | FAIL |
| MQ_REL_MOM12 | development | 1885 | -4.01 | -22.05 | -31.23 | FAIL |
| MQ_REL_MOM24 | development | 1484 | 5.22 | -20.05 | -32.03 | FAIL |

## Direction and market regime

| Candidate | Window | Regime | Direction | Trades | Stress mean | LCB |
|---|---|---|---|---:|---:|---:|
| MQ_DONCHIAN20_EMA48 | development | bearish | SHORT | 648 | 11.45 | -56.17 |
| MQ_DONCHIAN20_EMA48 | development | bullish | LONG | 652 | 19.12 | -28.11 |
| MQ_DONCHIAN55_EMA96 | development | bearish | SHORT | 416 | 36.66 | -47.95 |
| MQ_DONCHIAN55_EMA96 | development | bullish | LONG | 378 | 27.24 | -74.95 |
| MQ_EMA12_48_BOTH | development | bearish | SHORT | 1000 | 6.35 | -51.32 |
| MQ_EMA12_48_BOTH | development | bullish | LONG | 875 | 34.98 | -8.66 |
| MQ_EMA12_48_LONG | development | bullish | LONG | 875 | 34.98 | -9.36 |
| MQ_EMA12_48_SHORT | development | bearish | SHORT | 1000 | 6.35 | -50.64 |
| MQ_EMA24_96_BOTH | development | bearish | SHORT | 693 | 28.98 | -42.29 |
| MQ_EMA24_96_BOTH | development | bullish | LONG | 552 | 7.07 | -74.99 |
| MQ_REL_MOM12 | development | bearish | SHORT | 876 | -8.45 | -35.06 |
| MQ_REL_MOM12 | development | bullish | LONG | 1009 | -0.15 | -24.86 |
| MQ_REL_MOM24 | development | bearish | SHORT | 671 | 5.04 | -24.61 |
| MQ_REL_MOM24 | development | bullish | LONG | 813 | 5.37 | -31.33 |

Neutral is an explicit NoTrade state in this family.

## Distribution diagnostics

| Candidate | Trades | Median | Std | P01 | Worst | Best | Mean hold bars |
|---|---:|---:|---:|---:|---:|---:|---:|
| MQ_DONCHIAN20_EMA48 | 1300 | -76.31 | 531.58 | -996.87 | -1497.91 | 4265.96 | 8.33 |
| MQ_DONCHIAN55_EMA96 | 794 | -97.86 | 780.07 | -944.87 | -1497.91 | 11720.65 | 13.43 |
| MQ_EMA12_48_BOTH | 1875 | -54.02 | 497.16 | -822.87 | -2308.57 | 4265.96 | 7.61 |
| MQ_EMA12_48_LONG | 875 | -52.01 | 475.28 | -708.97 | -1221.68 | 4265.96 | 7.21 |
| MQ_EMA12_48_SHORT | 1000 | -56.77 | 515.42 | -1013.98 | -2308.57 | 3879.89 | 7.96 |
| MQ_EMA24_96_BOTH | 1245 | -77.53 | 682.73 | -991.49 | -1612.42 | 11998.16 | 12.00 |
| MQ_REL_MOM12 | 1885 | -39.63 | 387.28 | -787.81 | -1564.45 | 4417.03 | 3.41 |
| MQ_REL_MOM24 | 1484 | -46.13 | 499.89 | -797.17 | -1330.09 | 10506.26 | 4.70 |

All candidates have a negative median trade. Positive means are driven by rare large winners, while clustered downside keeps every bootstrap LCB below zero.

Detailed symbol slices are stored in `symbol_metrics.parquet`.
All paper/live/orders/promotion flags remain false.
