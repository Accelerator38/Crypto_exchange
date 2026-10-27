# Genetic_Primus experiment

Status: `ENGINEERING_SMOKE_V2`.

Research only. This artifact cannot authorize paper or live orders.

## Selected development champion

- Generation: 1
- Fitness: 1.7576
- Closed trades: 1393
- Mean net: 25.400909722467908 bps
- Median net: 11.720027720026241 bps
- LCB 95% net: -5.99503784110892 bps
- Max drawdown: 0.1954
- Failed development gates: nonpositive_lcb, nonpositive_market_state_lcb:trend_up, nonpositive_market_state_lcb:quiet_range, nonpositive_market_state_lcb:volatile_range, nonpositive_market_state_lcb:transition, nonpositive_direction_lcb:LONG, nonpositive_symbol_lcb:ADA/USDT, nonpositive_symbol_lcb:BNB/USDT, nonpositive_symbol_lcb:BTC/USDT, nonpositive_symbol_lcb:DOGE/USDT, nonpositive_symbol_lcb:ETH/USDT, nonpositive_symbol_lcb:LINK/USDT, nonpositive_symbol_lcb:XRP/USDT

## Evidence

- `evolution_overview.png`: fitness, expectancy, drawdown and activity by generation.
- `market_state_progress.png`: mean and LCB by causal market state.
- `champion_equity_drawdown.png`: champion equity and drawdown.
- `experiment_tables.xlsx`: separate Generation, Period, Slice and Trade worksheets.
- `generation_champions/`: immutable best genome and metrics for every generation.

Validation/OOS/sanity were not opened by this development runner.
