# Exia 4h fixed-entry exit-policy screen v1

Verdict: `NO_EXIT_POLICY_PASSED_STAGED_GATE`.

Entry and regime are sealed to `MQ_EMA12_48_LONG`; only position exits differ.
This is staged retrospective research and grants no paper/live authority.

## Stage progression

- Development -> validation: none
- Validation -> OOS: none
- OOS -> sanity: none
- Final passed: none

## Results

| Policy | Window | Trades | Mean | Median | LCB | Family LCB | P01 | P01 vs control | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| ATR_BREAKEVEN_PROGRESS24 | development | 1264 | 13.09 | -18.67 | -14.33 | -23.17 | -704.43 | 4.54 | FAIL |
| ATR_HARD_2_TIME24 | development | 955 | 27.11 | -47.34 | -10.89 | -25.12 | -731.28 | -22.31 | FAIL |
| ATR_TRAIL_2_5_TIME24 | development | 1104 | 17.73 | -38.87 | -14.81 | -25.68 | -744.07 | -35.10 | FAIL |
| CONTROL_SIGNAL_EXIT | development | 875 | 34.98 | -52.01 | -8.64 | -22.60 | -708.97 | 0.00 | CONTROL |

Gate requires >=20 trades, positive stress mean, median, ordinary and family-wise LCB, plus >=100 bps P01 improvement versus the same-window control.
All fills use completed-bar ATR, next-open signal execution, gap-aware stops, prior-extrema trailing and stop-first ambiguity handling.
Detailed trades, symbol slices and exit-reason slices are separate Parquet tables.
All paper/live/orders/promotion flags remain false.
