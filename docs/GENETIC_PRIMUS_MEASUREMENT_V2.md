# Measurement repair v2

## Freeze

Legacy V5/V6 rejections and V7 no-selection are retained. Their numerical PnL/drawdown and PBO interpretations are under review. No old result, sealed config, registry or simulator is overwritten. The new measurement lab is not a new tradable strategy and cannot authorize paper/live operation.

## Economic contract

Equal initial capital is assigned to each fixed-universe sleeve. Each entry opens a fixed quantity sized from that sleeve's current equity, bounded by equity including the entry fee. The quantity changes only on an explicit fill. Cash, realized PnL, entry/exit fees, unrealized PnL and equity are recorded consistently. Synthetic shorts are for arithmetic verification only: funding, borrow, margin and execution latency are not calibrated, so this contract has no market readiness.

OHLC timestamps are candle-open times. Features from a candle become available at its close; orders may execute at the next open at that timestamp under an explicit zero-latency assumption. The final close is timestamped at the interval end, never at the final candle open. Events are sequential, including events sharing a timestamp; returns are never added to merge different holding periods. Initial capital is included in the drawdown high-water mark.

Missing bars in the fixed market universe fail closed. Missing/nonfinite features imply abstention, not removal of a symbol from portfolio weights. No exchange or credential component is imported.

## Distinct horizons

Three-day retrain cadence is not a three-day minimum holding period. Decisions remain every four hours. Positions can carry across retrain boundaries if the target is unchanged. Forced liquidation at those boundaries is evaluated separately with a frozen policy. No old 1/3/6-bar vote labels are used in this lab: search directly scores executed cash PnL.

## Optimizer diagnostic contract

The preregistered JSON limits the model to two active features from four fixed inputs and coefficients from {-1,0,1}. Sparse deterministic grid, random search and genetic search receive exactly 24 unique candidate evaluations each, including the same NoTrade/allow-all anchors. All use the same train-only scaler, chronological intervals, costs and bootstrap bank. Validation never selects a second winner. Fitness is one paired block-bootstrap lower bound, with risk/evidence gates kept separate. Recency weighting and stacked selection penalties are disabled here deliberately, rather than silently hidden in ranking tie-breakers.

The diagnostic sequence is: hand-calculated accounting oracles; causal/prefix and gap tests; positive-control/null optimizer checks; one bounded revealed-data comparison. Three diagnostic origins are not evidence sufficient for trading or a conclusion that GA is universally useful/useless.

## Blind gate

No blind run is authorized by the measurement-lab contract. It needs a separately frozen scientifically justified contract, a data-exposure registry, sufficient unseen data, calibrated market costs, and verified positive development evidence. Missing prerequisites produce an explicit no-go result, not relabeling of historical data as unopened.
