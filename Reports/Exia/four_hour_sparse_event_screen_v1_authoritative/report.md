# Exia 4h sparse-event entry screen v1

Verdict: `NO_SPARSE_EVENT_PASSED_STAGED_GATE`.

All events use one sealed 4h ATR/progress position lifecycle. A zero event does not close a position.
This is staged retrospective research and grants no paper/live authority.

## Stage progression

- Development -> validation: none
- Validation -> OOS: none
- OOS -> sanity: none
- Final passed: none

## Results

| Event | Window | Events | Trades | Mean | Median | LCB | Family LCB | P01 | P01 vs control | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| BULLISH_REGIME_TRANSITION | development | 605 | 502 | -37.20 | -16.00 | -87.47 | -104.20 | -819.53 | -44.17 | FAIL |
| BULL_TREND_FAST_RECLAIM | development | 259 | 216 | -27.74 | -16.00 | -95.45 | -120.64 | -756.03 | 19.33 | FAIL |
| CONTROL_MQ_FRESH_ACTIVATION | development | 875 | 692 | -32.19 | -16.00 | -74.64 | -91.04 | -775.35 | 0.00 | CONTROL |
| TOP_QUARTILE_BREAKOUT20 | development | 472 | 350 | -7.93 | -16.00 | -65.42 | -86.27 | -913.76 | -138.41 | FAIL |

Gate: >=20 trades, positive stress mean/median/ordinary LCB/family-wise LCB and >=100 bps P01 improvement versus the event control.
Detailed trades, symbols, exit reasons and market regimes are stored in separate Parquet tables.
All paper/live/orders/promotion flags remain false.
