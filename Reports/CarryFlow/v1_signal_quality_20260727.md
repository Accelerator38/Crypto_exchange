# CarryFlow root 003: signal quality analysis

Generated: `2026-07-27T08:26:53.262551+00:00`

This report is research-only. It does not authorize paper or live orders.

## Verdict

The root exposed two independent defects: activation is too rare, and `diagnostic.edge` is not calibrated as a forward return. The fixed six-bar exit then gives back favorable excursion.

## Activation funnel

| Stage | Count | Share |
|---|---:|---:|
| Flat observations | 445 | 100.0% |
| OI passed | 75 | 16.9% |
| Raw actor candidates | 6 | 1.3% |
| Actor selected | 4 | 0.9% |
| Policy allowed | 4 | 0.9% |

## OI gate

The threshold is 150 bps over three bars. 370 observations failed it.
Median failed OI change was -18.0 bps; p95 was 119.4 bps. Only 15 failures were within 25 bps of the threshold.

Lowering the threshold is not supported: most rejected observations are materially weaker, not borderline cases.

| Symbol | Flat | OI pass | Pass rate | Median, bps | p90, bps |
|---|---:|---:|---:|---:|---:|
| BTC | 58 | 4 | 6.9% | 21.7 | 117.3 |
| ETH | 58 | 10 | 17.2% | 21.4 | 186.3 |
| SOL | 52 | 6 | 11.5% | -42.2 | 190.8 |
| BNB | 58 | 9 | 15.5% | 5.0 | 230.5 |
| XRP | 58 | 10 | 17.2% | 44.7 | 201.3 |
| DOGE | 58 | 10 | 17.2% | -5.1 | 265.2 |
| ADA | 57 | 18 | 31.6% | -32.9 | 641.7 |
| LINK | 46 | 8 | 17.4% | 24.1 | 373.0 |

## Signal calibration

Mean predicted move: 235.14 bps.
Mean realized gross move: 36.51 bps; mean net: 24.53 bps.
Realized/predicted ratio: 15.53% on only 4 trades.

`diagnostic.edge` may rank candidates, but its current linear conversion to bps is not a valid expectancy estimate.

| Edge component | Mean contribution | Share |
|---|---:|---:|
| basis | 1.015 | 36.3% |
| crowding | 0.936 | 33.5% |
| funding | 0.112 | 4.0% |
| oi_excess | 0.234 | 8.4% |
| price_divergence | 0.499 | 17.9% |

Basis and crowding dominate the score; OI is mostly a binary gate and contributes little to ranking after the threshold is passed.

| Bar | Symbol | Regime | Policy | Predicted | 6h gross | 6h net |
|---:|---|---|---|---:|---:|---:|
| 4 | LINK | range_low_vol | ALLOW_OPEN | 208.6 | 122.7 | 110.7 |
| 26 | SOL | range_low_vol | ALLOW_OPEN | 250.0 | 180.8 | 168.9 |
| 46 | ADA | range_low_vol | ALLOW_OPEN | 232.0 | -403.5 | -415.5 |
| 52 | LINK | choppy_up | ALLOW_OPEN | 250.0 | -37.5 | -49.5 |

All five allowed trades were in `range_low_vol`; the two bearish SOL candidates were blocked by low regime confidence.

## Exit behavior

All 4 closed trades used `max_holding`. Target: 240 bps; stop: 120 bps.
Target hits by six bars: 0; stop hits: 1. All 4 trades had MFE above modeled cost.
MFE/MAE use sealed hourly OHLC and are diagnostic excursions, not proof that a trailing exit could have filled at the extreme.

| Fixed exit | Observations | Mean net, bps | Positive |
|---:|---:|---:|---:|
| 1h | 4 | 6.70 | 50.0% |
| 2h | 4 | -32.61 | 50.0% |
| 3h | 4 | -54.02 | 25.0% |
| 4h | 4 | -64.59 | 25.0% |
| 6h | 4 | -46.35 | 50.0% |
| 8h | 3 | -33.64 | 33.3% |
| 12h | 3 | -19.45 | 66.7% |

The four-hour row is an in-sample hypothesis, not a selected policy.

## Next design constraints

- Do not lower the OI threshold based on this root.
- Separate candidate ranking score from calibrated expected move.
- Calibrate expected move on independent train/validation/OOS data.
- Evaluate one profile-level cost-aware exit against the fixed six-bar exit; do not expose independent live exit knobs.
- Treat the four-bar result as a hypothesis only because it was selected after observing five trades.
