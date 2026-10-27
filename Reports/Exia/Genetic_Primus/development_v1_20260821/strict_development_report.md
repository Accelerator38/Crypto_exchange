# Genetic_Primus V1 strict development freeze

Status: `DEVELOPMENT_REJECTED_SLICE_GATES`.

Validation allowed: `false`. Validation, OOS,
sanity, paper and live were not opened. Evaluation uses the stress round-trip cost.

## Overall development result

- Closed trades: 4272
- Mean: 23.355147 bps
- Median: 7.654739 bps
- LCB 95%: 8.270523 bps
- Max drawdown: 19.842115%
- Supported months: 24/24

## Failed strict gates

- `nonpositive_market_state_lcb:trend_up`
- `nonpositive_market_state_lcb:volatile_range`
- `nonpositive_market_state_lcb:transition`
- `nonpositive_direction_lcb:SHORT`
- `nonpositive_symbol_lcb:ADA/USDT`
- `nonpositive_symbol_lcb:BNB/USDT`
- `nonpositive_symbol_lcb:BTC/USDT`
- `nonpositive_symbol_lcb:DOGE/USDT`
- `nonpositive_symbol_lcb:ETH/USDT`
- `nonpositive_symbol_lcb:XRP/USDT`

## Weakest supported slices

- `symbol:XRP/USDT`: trades=545, mean=-4.588, median=-3.436, LCB=-34.302 bps
- `market_state:trend_up`: trades=511, mean=4.415, median=5.302, LCB=-26.196 bps
- `market_state:volatile_range`: trades=741, mean=11.988, median=4.391, LCB=-21.117 bps
- `symbol:DOGE/USDT`: trades=577, mean=21.585, median=3.920, LCB=-8.386 bps
- `direction:SHORT`: trades=2146, mean=11.854, median=14.883, LCB=-7.309 bps
- `symbol:BNB/USDT`: trades=389, mean=19.714, median=-1.007, LCB=-6.950 bps
- `symbol:ETH/USDT`: trades=461, mean=25.084, median=3.600, LCB=-3.936 bps
- `market_state:transition`: trades=1800, mean=15.287, median=7.710, LCB=-3.538 bps

## Immutable identity

- Experiment manifest SHA: `d23472539dad3eadd5760a709de59ea18aea0be282353a556a292654448e5d5a`
- Champion file SHA: `b90d5e6bc0f6bf7e54d8f757deee55b4a8c7bffd12d106bf848d3cdca0125e98`
- Raw genome SHA: `fe30dd4307c65de112282ee5cbc715c561964dcfd63f1b7a30d59d52050a09a3`
- Scaler SHA: `5be04193487f9a83e355f387066850cff83f1a535e18f75f405f841defcb0793`
- Dataset SHA: `f66529b4135d04ddc35b7a6535667d1c6d53cd39c34adb86f23e54d4082b2de7`

This freeze artifact has no promotion authority.
