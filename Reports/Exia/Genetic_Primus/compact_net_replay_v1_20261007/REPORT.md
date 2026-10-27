# Compact net replay — 2026-10-07

Offline research only. No live/paper/promotion authority.

Calibration: June 20–July 26; 12 causal 3-day retrains. Benchmark: August 1–September 30 inclusive; 20 frozen origins.
Parameters chosen BEFORE benchmark replay; frozen forecast fits reused. Costs: fee 6bps/fill; adverse 2bps normal / 8bps stress.
Signed funding rates observed; OHLCV open used as registered mark proxy. Not observed exchange marks or executable fills.
10,000 USDT shared capital; 10% per selected asset, no redistribution, no borrowing; unchanged targets carry units.
Actual exposure may drift above the 80% target budget. DD 10/20/30% are research constraints, not live risk agreements.

| Policy | Net normal % | Net stress % | DD normal % | Audit log LCB normal/stress bps/day |
|---|---:|---:|---:|---:|
| linear6_uniform | 13.0705 | 12.5058 | 9.3083 | -9.205/-9.849 |
| linear6_half_life_90_anchor20 | 7.5929 | 7.3680 | 6.4178 | -8.619/-8.928 |
| NoTrade | 0.0000 | 0.0000 | 0.0000 | 0.000/0.000 |
| mlp8_half_life_90_anchor20 | -1.2382 | -1.5446 | 4.4105 | -11.477/-12.075 |
| mlp8_uniform | -3.0925 | -3.4182 | 9.5340 | -23.850/-24.315 |
| EMA48_long_only | 8.5750 | 8.1633 | 11.5937 | -17.932/-18.435 |

The Aug19–22 exceptional return window remains included; independent source verification is pending.
Bootstrap banks: shared across policies, block=3 calendar days, 2048 samples, seeds 7301/9703; daily cash bins included.
LCBs are robustness diagnostics, not multiplicity-adjusted confirmation; global immutable objective registry is incomplete.
12 additional linear calibration fits; 0 new benchmark fits; 560 known previous fits including superseded attempt; 572 total.
GA/random search skipped: no complete global registry or authorized production evidence. No new prospective interval sealed.
Next: independently validate the common price jump and actual execution/mark assumptions; do not tune Aug-Sep thresholds.
