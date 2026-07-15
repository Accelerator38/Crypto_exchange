# Legacy Panteon Trade Runtime

Updated: 2026-07-15

`src/panteon_runtime/Panteon_Trade.py` is legacy infrastructure. It still
provides the Bitget/MEXC market bridge and the R&D Flash/ensemble runtime, but it
is not an approved Bitget order entrypoint.

Use `Start_panteon.py` as the only root launcher. For Bitget:

- `multi` and `singlton(<actor>)` are virtual/R&D routes;
- non-virtual startup requires `PANTEON_TRADE_REGIME=policy`;
- `policy` registers one sealed manifest actor and bypasses all legacy actor
  selection before any signal is evaluated;
- v1 `_run_cycle()` and `run_once()` are never called by the v2 bridge runner.

The policy route still reuses narrowly scoped infrastructure:

- public market/funding reads from the bridge and Bitget clients;
- exact closed-bar and order-book reads from
  `src/panteon_v2/app/bitget_adapter.py`;
- confirmed order execution and reconciliation through the v2
  `TradeExecutor`, `OrderLedger` and `PositionTracker`.

It does not reuse the v1 real player, Flash ranking, actor rotation, Genetics
promotion or shadow tournament for Bitget live decisions.

See `docs/PANTEON_3_OPERATOR_GUIDE.md` for launch behavior and
`docs/PANTEON_VNEXT_STRATEGY.md` for the evidence contract.
