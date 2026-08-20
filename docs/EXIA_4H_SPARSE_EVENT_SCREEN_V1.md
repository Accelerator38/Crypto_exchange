# Exia 4h sparse-event entry screen v1

## Decision

The experiment is closed with `NO_SPARSE_EVENT_PASSED_STAGED_GATE`.

No event opened validation, OOS or sanity. The current 4h long-trend event branch must not be extended with another threshold grid. This result grants no paper or live authority.

## Isolation contract

- Dataset, 4h profile, costs and staged windows are inherited by SHA from the sealed market-quorum family.
- Position lifecycle is inherited by SHA from `ATR_BREAKEVEN_PROGRESS24`.
- Every candidate is long-only and executes a completed-bar event at the next 4h open.
- A zero event holds the current position; it is not a signal-state exit.
- Hard stop, breakeven, trailing, no-progress and 24-bar maximum holding settings are identical for every event.
- Only the event definition differs.

This removes duplicated dataset, cost, window and exit settings from the new family config.

## Preregistered events

| Event | Role | Definition |
|---|---|---|
| CONTROL_MQ_FRESH_ACTIVATION | Control | Fresh activation of the previous long EMA/market-quorum state |
| BULLISH_REGIME_TRANSITION | Candidate | First confirmed broad bullish bar with aligned local EMA trend |
| TOP_QUARTILE_BREAKOUT20 | Candidate | Fresh prior-20-bar breakout in the strongest full8 momentum quartile |
| BULL_TREND_FAST_RECLAIM | Candidate | Fast-EMA reclaim after a pullback in a confirmed broad bullish trend |

## Development result

| Event | Events | Trades | Mean bps | Median bps | LCB bps | Family LCB bps | P01 bps | P01 delta bps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CONTROL_MQ_FRESH_ACTIVATION | 875 | 692 | -32.19 | -16.00 | -74.64 | -91.04 | -775.35 | 0.00 |
| BULLISH_REGIME_TRANSITION | 605 | 502 | -37.20 | -16.00 | -87.47 | -104.20 | -819.53 | -44.17 |
| TOP_QUARTILE_BREAKOUT20 | 472 | 350 | -7.93 | -16.00 | -65.42 | -86.27 | -913.76 | -138.41 |
| BULL_TREND_FAST_RECLAIM | 259 | 216 | -27.74 | -16.00 | -95.45 | -120.64 | -756.03 | 19.33 |

All candidates failed positive mean, median, ordinary LCB and family-wise LCB requirements. None improved P01 by the required 100 bps.

## Slice audit

Some symbol means were positive, but none had a positive ordinary or family-wise moving-block LCB. The strongest apparent slice was `BULL_TREND_FAST_RECLAIM + ADA/USDT`: 30 trades, mean `+123.58 bps`, median `+15.93 bps`, LCB `-8.50 bps`, family-wise LCB `-110.85 bps`. It remains a diagnostic observation and cannot be promoted or opened on later windows.

Across the complete family there were zero positive-LCB symbol slices. Full results are stored in `symbol_metrics.parquet` with separate columns for ordinary and family-wise LCB.

## Root-cause interpretation

Sparse timing reduced activation but did not produce positive costed expectancy. Between 80% and 86% of candidate trades ended at a stop. The fixed lifecycle exposes a large negative tail, while occasional long trend winners are too rare to make the lower confidence bound positive.

The evidence now rejects two nearby explanations:

1. A generic ATR exit does not rescue the continuous MQ entry.
2. A sparse transition, breakout or pullback event does not rescue the same long-trend premise under a common lifecycle.

The next experiment should therefore not optimize another strategy or stop. It should measure entry information directly with a preregistered forward-return feasibility audit at 1, 3, 6 and 12 bars, including stress-costed return, MFE and MAE. If an event has no positive LCB at any fixed horizon, it should be retired before any further position-policy work. This separates absence of predictive information from execution-policy failure.

## Artifacts

- Family: `configs/exia_4h_sparse_event_family_v1.json`.
- Runner: `tools/run_exia_4h_sparse_event_screen_v1.py`.
- Authoritative run: `Reports/Exia/four_hour_sparse_event_screen_v1_authoritative`.
- Deterministic repeat: `Reports/Exia/four_hour_sparse_event_screen_v1_authoritative_repeat`.
- Exact repeat equality: metrics, trades, exit-reason metrics, symbol metrics and regime metrics.

All paper, live, order and promotion flags remain `false`.
