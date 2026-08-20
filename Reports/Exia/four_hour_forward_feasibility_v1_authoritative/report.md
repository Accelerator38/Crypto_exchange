# Exia 4h sparse-event forward feasibility v1

Verdict: `NO_EVENT_HORIZON_PASSED_STAGED_GATE`.

This audit labels event information directly; it does not use a stop, trailing rule or position policy.
It grants no paper/live authority.

## Stage progression

- Development -> validation: none
- Validation -> OOS: none
- OOS -> sanity: none
- Final passed: none

## Results

| Event | H | Window | Events | Labels | Coverage | Mean | Median | LCB | Family LCB | MFE med | MAE med | Gate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BULLISH_REGIME_TRANSITION | 1 | development | 605 | 605 | 1.00 | 28.05 | -7.17 | 7.59 | -4.32 | 94.76 | 69.68 | FAIL |
| BULLISH_REGIME_TRANSITION | 3 | development | 605 | 605 | 1.00 | 36.30 | -5.76 | 1.78 | -15.85 | 181.47 | 123.42 | FAIL |
| BULLISH_REGIME_TRANSITION | 6 | development | 605 | 605 | 1.00 | -3.23 | -47.34 | -52.01 | -89.06 | 229.64 | 217.66 | FAIL |
| BULLISH_REGIME_TRANSITION | 12 | development | 605 | 605 | 1.00 | 6.26 | -60.28 | -65.15 | -110.13 | 300.22 | 326.25 | FAIL |
| BULL_TREND_FAST_RECLAIM | 1 | development | 259 | 259 | 1.00 | -14.32 | -31.29 | -38.91 | -53.39 | 83.78 | 88.39 | FAIL |
| BULL_TREND_FAST_RECLAIM | 3 | development | 259 | 259 | 1.00 | 7.59 | -17.96 | -32.06 | -54.40 | 139.66 | 151.37 | FAIL |
| BULL_TREND_FAST_RECLAIM | 6 | development | 259 | 259 | 1.00 | -11.61 | -0.45 | -67.03 | -105.67 | 225.45 | 212.37 | FAIL |
| BULL_TREND_FAST_RECLAIM | 12 | development | 259 | 259 | 1.00 | 35.55 | -22.36 | -48.07 | -106.67 | 318.27 | 261.77 | FAIL |
| CONTROL_MQ_FRESH_ACTIVATION | 1 | development | 875 | 875 | 1.00 | 19.16 | -9.10 | 3.28 | -7.58 | 94.76 | 71.66 | CONTROL |
| CONTROL_MQ_FRESH_ACTIVATION | 3 | development | 875 | 875 | 1.00 | 20.33 | -8.80 | -7.35 | -24.94 | 177.91 | 128.79 | CONTROL |
| CONTROL_MQ_FRESH_ACTIVATION | 6 | development | 875 | 875 | 1.00 | -8.24 | -42.70 | -48.51 | -74.10 | 237.41 | 218.08 | CONTROL |
| CONTROL_MQ_FRESH_ACTIVATION | 12 | development | 875 | 875 | 1.00 | 11.32 | -38.70 | -46.09 | -87.62 | 317.58 | 321.54 | CONTROL |
| TOP_QUARTILE_BREAKOUT20 | 1 | development | 472 | 472 | 1.00 | -20.84 | -36.63 | -41.71 | -56.02 | 125.35 | 112.49 | FAIL |
| TOP_QUARTILE_BREAKOUT20 | 3 | development | 472 | 472 | 1.00 | -10.49 | -52.88 | -49.40 | -76.36 | 194.98 | 190.41 | FAIL |
| TOP_QUARTILE_BREAKOUT20 | 6 | development | 472 | 472 | 1.00 | -0.20 | -67.78 | -55.83 | -94.40 | 268.02 | 293.48 | FAIL |
| TOP_QUARTILE_BREAKOUT20 | 12 | development | 472 | 472 | 1.00 | 45.10 | -35.40 | -18.80 | -56.47 | 366.67 | 370.24 | FAIL |

Returns are next-open to open after H bars and include the sealed 16 bps stress round-trip cost.
Overlapping labels are allowed, but the LCB resamples complete UTC-day clusters.
All paper/live/orders/promotion flags remain false.
