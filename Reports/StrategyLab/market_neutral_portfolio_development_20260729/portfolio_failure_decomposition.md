# P5 portfolio failure decomposition

## Safety

- Source verdict: `terminal_rejected_development`
- Immutable replay parity: `passed`
- Validation/OOS/sanity opened: `false`
- Continuation/paper/live allowed: `false`

## Root-cause indicators

- Market beta: `0.0500`
- Market correlation: `0.1174`
- Alpha intercept: `49.0490 bps`
- Worst five loss share: `24.82%`
- Best five profit share: `36.99%`
- Maximum drawdown: `$7.6533`
- Candidate vs raw baseline mean: `50.8281` vs `71.8294 bps`

Flags: `cost_dominant=false`, `market_beta_material=false`, `ranking_transform_underperformed_baseline=true`, `symbol_concentration_material=true`, `tail_loss_concentration_material=false`

## Year stability

| Year | Portfolios | Mean net bps | LCB bps | Net PnL USD |
|---|---:|---:|---:|---:|
| 2022 | 46 | 31.7616 | -73.2284 | 5.8441 |
| 2023 | 50 | 68.3693 | -36.1376 | 13.6739 |

## Direction contribution

| Direction | Legs | Mean net bps | Net PnL USD |
|---|---:|---:|---:|
| LONG | 192 | 60.7476 | 11.6635 |
| SHORT | 192 | 40.9086 | 7.8545 |

## Symbol contribution

| Symbol | Legs | Mean net bps | Net PnL USD |
|---|---:|---:|---:|
| ADA | 44 | -250.8826 | -11.0388 |
| BNB | 41 | -84.8335 | -3.4782 |
| BTC | 54 | 94.3326 | 5.0940 |
| DOGE | 61 | -192.1048 | -11.7184 |
| ETH | 38 | 29.7421 | 1.1302 |
| LINK | 39 | 139.3114 | 5.4331 |
| SOL | 52 | 559.6391 | 29.1012 |
| XRP | 55 | 90.8156 | 4.9949 |
