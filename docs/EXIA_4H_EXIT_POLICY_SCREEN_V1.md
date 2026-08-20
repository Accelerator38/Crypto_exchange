# Exia 4h fixed-entry exit-policy screen v1

## Decision

The experiment is closed with `NO_EXIT_POLICY_PASSED_STAGED_GATE`.

No policy opened validation, OOS or sanity. This result grants no paper or live authority.

## Fixed inputs

- Timeframe: `4h` (`exia_4h_v1`).
- Entry and market regime: `MQ_EMA12_48_LONG` from the sealed market-quorum family.
- Entry-family SHA-256: `3042e2834ff0d70cd0d3b7e29a418d7fd47f1ca49286e0ab0b825e6435c54f70`.
- Dataset SHA-256: `f66529b4135d04ddc35b7a6535667d1c6d53cd39c34adb86f23e54d4082b2de7`.
- Stress round-trip cost: `16 bps`.
- Development window: `[2022-01-01, 2024-01-01)`.
- Validation, OOS and sanity remain unopened unless the preceding gate passes.

Only position exits differed. There was no entry, regime, symbol or timeframe retuning.

## Execution contract

- A close-time signal is entered or exited at the next 4h open.
- ATR at entry contains completed bars only.
- A gap through a stop fills at the bar open.
- A bar touching stop and target uses the stop price first.
- A trailing stop uses extrema completed before the current bar.
- The control has the original signal-driven exit and is comparison-only.

## Gate

An alternative needs all of the following in each opened window:

- at least 20 closed trades;
- positive stress-costed mean;
- positive stress-costed median;
- positive ordinary moving-block LCB;
- positive family-wise moving-block LCB;
- P01 at least 100 bps better than the same-window control.

At most one alternative may advance to the next window.

## Development result

| Policy | Trades | Mean bps | Median bps | LCB bps | Family LCB bps | P01 bps | P01 delta bps | Mean hold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CONTROL_SIGNAL_EXIT | 875 | 34.98 | -52.01 | -8.64 | -22.60 | -708.97 | 0.00 | 7.21 |
| ATR_HARD_2_TIME24 | 955 | 27.11 | -47.34 | -10.89 | -25.12 | -731.28 | -22.31 | 6.49 |
| ATR_TRAIL_2_5_TIME24 | 1104 | 17.73 | -38.87 | -14.81 | -25.68 | -744.07 | -35.10 | 5.43 |
| ATR_BREAKEVEN_PROGRESS24 | 1264 | 13.09 | -18.67 | -14.33 | -23.17 | -704.43 | 4.54 | 4.62 |

## Interpretation

`ATR_BREAKEVEN_PROGRESS24` improved the median substantially, but it remained negative and its P01 improvement was only 4.54 bps. Its mean also fell because repeated stop/re-entry cycles increased turnover. The two simpler ATR policies worsened P01.

The negative median therefore is not primarily a missing universal stop-loss problem. Most original signal exits are still poor after costs, while static ATR stops cannot distinguish a valid trend from a false activation. The result does not justify another grid of ATR multipliers.

The next bounded research step should replace the continuous EMA-state entry with a sparse event entry while retaining the same 4h panel, execution model and staged windows. The event must demonstrate better activation quality before any additional exit optimization. Candidate examples are a fresh bullish regime transition, a confirmed cross-sectional breakout, or pullback continuation after a confirmed broad trend. These should be preregistered as a small mutually exclusive family, not combined into an ensemble.

## Artifacts

- Authoritative run: `Reports/Exia/four_hour_exit_policy_screen_v1_final`.
- Deterministic repeat: `Reports/Exia/four_hour_exit_policy_screen_v1_repeat`.
- Exact repeat equality: metrics, trades, exit-reason metrics and symbol metrics.
- Control equality: all 875 control trade timestamps, prices, directions, holding periods and gross returns match the prior `MQ_EMA12_48_LONG` development ledger.

All paper, live, order and promotion flags remain `false`.
