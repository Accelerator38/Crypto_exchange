# CarryFlow root 003: signal quality analysis

Generated: `2026-07-27T08:26:53.275121+00:00`

This report is research-only. It does not authorize paper or live orders.

## Verdict

The root exposed two independent defects: activation is too rare, and `diagnostic.edge` is not calibrated as a forward return. The fixed six-bar exit then gives back favorable excursion.

## Activation funnel

| Stage | Count | Share |
|---|---:|---:|
| Flat observations | 501 | 100.0% |
| OI passed | 59 | 11.8% |
| Raw actor candidates | 11 | 2.2% |
| Actor selected | 6 | 1.2% |
| Policy allowed | 5 | 1.0% |

## OI gate

The threshold is 150 bps over three bars. 442 observations failed it.
Median failed OI change was 0.0 bps; p95 was 110.5 bps. Only 13 failures were within 25 bps of the threshold.

Lowering the threshold is not supported: most rejected observations are materially weaker, not borderline cases.

| Symbol | Flat | OI pass | Pass rate | Median, bps | p90, bps |
|---|---:|---:|---:|---:|---:|
| BTC | 66 | 0 | 0.0% | 5.3 | 50.5 |
| ETH | 60 | 2 | 3.3% | 0.0 | 99.8 |
| SOL | 60 | 1 | 1.7% | -13.7 | 65.5 |
| BNB | 66 | 8 | 12.1% | 4.1 | 153.2 |
| XRP | 60 | 4 | 6.7% | 3.3 | 111.3 |
| DOGE | 66 | 8 | 12.1% | 29.7 | 195.8 |
| ADA | 60 | 19 | 31.7% | 0.0 | 344.5 |
| LINK | 63 | 17 | 27.0% | 22.4 | 201.9 |

## Signal calibration

Mean predicted move: 205.43 bps.
Mean realized gross move: 17.53 bps; mean net: 5.54 bps.
Realized/predicted ratio: 8.53% on only 4 trades.

`diagnostic.edge` may rank candidates, but its current linear conversion to bps is not a valid expectancy estimate.

| Edge component | Mean contribution | Share |
|---|---:|---:|
| basis | 0.949 | 41.7% |
| crowding | 0.868 | 38.2% |
| funding | 0.200 | 8.8% |
| oi_excess | 0.058 | 2.5% |
| price_divergence | 0.199 | 8.7% |

Basis and crowding dominate the score; OI is mostly a binary gate and contributes little to ranking after the threshold is passed.

| Bar | Symbol | Regime | Policy | Predicted | 6h gross | 6h net |
|---:|---|---|---|---:|---:|---:|
| 4 | ETH | bearish | ALLOW_OPEN | 203.0 | 64.9 | 52.9 |
| 16 | ADA | bullish | NO_TRADE | 169.1 | -169.0 | -181.0 |
| 22 | XRP | range_low_vol | ALLOW_OPEN | 250.0 | 5.5 | -6.5 |
| 36 | SOL | range_low_vol | ALLOW_OPEN | 204.2 | -54.3 | -66.3 |
| 43 | ADA | range_low_vol | ALLOW_OPEN | 164.5 | 54.0 | 42.0 |
| 63 | LINK | range_low_vol | ALLOW_OPEN | 250.0 | n/a | n/a |

All five allowed trades were in `range_low_vol`; the two bearish SOL candidates were blocked by low regime confidence.

## Exit behavior

All 4 closed trades used `max_holding`. Target: 240 bps; stop: 120 bps.
Target hits by six bars: 0; stop hits: 0. All 4 trades had MFE above modeled cost.
MFE/MAE use sealed hourly OHLC and are diagnostic excursions, not proof that a trailing exit could have filled at the extreme.

| Fixed exit | Observations | Mean net, bps | Positive |
|---:|---:|---:|---:|
| 1h | 5 | -29.46 | 0.0% |
| 2h | 5 | -21.52 | 20.0% |
| 3h | 5 | -40.26 | 20.0% |
| 4h | 4 | -19.23 | 0.0% |
| 6h | 4 | 5.54 | 50.0% |
| 8h | 4 | -12.80 | 50.0% |
| 12h | 4 | -29.48 | 25.0% |

The four-hour row is an in-sample hypothesis, not a selected policy.

## Next design constraints

- Do not lower the OI threshold based on this root.
- Separate candidate ranking score from calibrated expected move.
- Calibrate expected move on independent train/validation/OOS data.
- Evaluate one profile-level cost-aware exit against the fixed six-bar exit; do not expose independent live exit knobs.
- Treat the four-bar result as a hypothesis only because it was selected after observing five trades.
