# CarryFlow root 003: signal quality analysis

Generated: `2026-07-27T08:01:57.411692+00:00`

This report is research-only. It does not authorize paper or live orders.

## Verdict

The root exposed two independent defects: activation is too rare, and `diagnostic.edge` is not calibrated as a forward return. The fixed six-bar exit then gives back favorable excursion.

## Activation funnel

| Stage | Count | Share |
|---|---:|---:|
| Flat observations | 562 | 100.0% |
| OI passed | 72 | 12.8% |
| Raw actor candidates | 12 | 2.1% |
| Actor selected | 7 | 1.2% |
| Policy allowed | 5 | 0.9% |

## OI gate

The threshold is 150 bps over three bars. 490 observations failed it.
Median failed OI change was -13.8 bps; p95 was 118.8 bps. Only 19 failures were within 25 bps of the threshold.

Lowering the threshold is not supported: most rejected observations are materially weaker, not borderline cases.

| Symbol | Flat | OI pass | Pass rate | Median, bps | p90, bps |
|---|---:|---:|---:|---:|---:|
| BTC | 74 | 0 | 0.0% | 0.0 | 54.2 |
| ETH | 74 | 4 | 5.4% | 0.6 | 111.0 |
| SOL | 74 | 14 | 18.9% | 0.0 | 211.9 |
| BNB | 56 | 9 | 16.1% | 0.0 | 202.5 |
| XRP | 74 | 13 | 17.6% | 0.0 | 182.7 |
| DOGE | 68 | 16 | 23.5% | 0.8 | 286.2 |
| ADA | 74 | 12 | 16.2% | -20.9 | 174.8 |
| LINK | 68 | 4 | 5.9% | -12.1 | 126.2 |

## Signal calibration

Mean predicted move: 212.17 bps.
Mean realized gross move: 7.66 bps; mean net: -4.33 bps.
Realized/predicted ratio: 3.61% on only 5 trades.

`diagnostic.edge` may rank candidates, but its current linear conversion to bps is not a valid expectancy estimate.

| Edge component | Mean contribution | Share |
|---|---:|---:|
| basis | 0.993 | 44.8% |
| crowding | 0.721 | 32.5% |
| funding | 0.200 | 9.0% |
| oi_excess | 0.089 | 4.0% |
| price_divergence | 0.213 | 9.6% |

Basis and crowding dominate the score; OI is mostly a binary gate and contributes little to ranking after the threshold is passed.

| Bar | Symbol | Regime | Policy | Predicted | 6h gross | 6h net |
|---:|---|---|---|---:|---:|---:|
| 4 | BNB | range_low_vol | ALLOW_OPEN | 173.3 | -65.4 | -77.4 |
| 10 | LINK | range_low_vol | ALLOW_OPEN | 250.0 | 77.0 | 65.0 |
| 28 | BNB | range_low_vol | ALLOW_OPEN | 225.3 | 59.5 | 47.5 |
| 45 | SOL | bearish | NO_TRADE | 250.0 | 23.8 | 11.8 |
| 47 | SOL | bearish | NO_TRADE | 223.0 | 83.1 | 71.1 |
| 59 | DOGE | range_low_vol | ALLOW_OPEN | 217.9 | -59.3 | -71.3 |
| 68 | BNB | range_low_vol | ALLOW_OPEN | 194.3 | 26.5 | 14.5 |

All five allowed trades were in `range_low_vol`; the two bearish SOL candidates were blocked by low regime confidence.

## Exit behavior

All 5 closed trades used `max_holding`. Target: 240 bps; stop: 120 bps.
Target hits by six bars: 0; stop hits: 0. All 5 trades had MFE above modeled cost.
MFE/MAE use sealed hourly OHLC and are diagnostic excursions, not proof that a trailing exit could have filled at the extreme.

| Fixed exit | Observations | Mean net, bps | Positive |
|---:|---:|---:|---:|
| 1h | 5 | 8.12 | 40.0% |
| 2h | 5 | 13.24 | 60.0% |
| 3h | 5 | 16.35 | 60.0% |
| 4h | 5 | 26.95 | 80.0% |
| 6h | 5 | -4.33 | 60.0% |
| 8h | 4 | 7.18 | 75.0% |
| 12h | 4 | 26.99 | 50.0% |

The four-hour row is an in-sample hypothesis, not a selected policy.

## Next design constraints

- Do not lower the OI threshold based on this root.
- Separate candidate ranking score from calibrated expected move.
- Calibrate expected move on independent train/validation/OOS data.
- Evaluate one profile-level cost-aware exit against the fixed six-bar exit; do not expose independent live exit knobs.
- Treat the four-bar result as a hypothesis only because it was selected after observing five trades.
