# Synthetic long/NoTrade net accounting — not a calibrated market objective

Separate immutable engineering contract. No market data, fit, pytest, collector,
network or runtime. SYN_LINEAR_USDT only, no short/leverage/margin/liquidation
simulation. Old label matrices, contracts and gross results are not requalified.
Market_calibration=null is missing calibration, not zero costs. Safety stays false.

Input prices are explicitly synthetic mid references, never observed fills or
OHLCV opens. Buy reference is increased by entry half-spread+slippage; sell
reference decreased by exit half-spread+slippage. Half-spread is already a
one-side input; it is not halved again. Fees are per-side bps times that side's
adjusted fill notional, not both times entry notional. No second subtraction of
spread/slippage after adjusted-fill PnL. Components are recorded separately.

Allocated quote=common capital × explicit allocation fraction in [0,1]. Fixed
quantity=allocated/(entry_fill × (1+entry_fee_rate)), reserving entry fee inside
the allocation. This is a synthetic sizing/accounting convention, not exchange
margin execution or lot rounding. Net quote=reference gross−execution drag−fees
−signed funding debit−explicit borrow debit; return divides by common capital,
not trade notional. Stress can change filled quantity under the same quote budget.

Funding requires an explicit complete fixture event list. Empty list is an
explicit synthetic no-settlement scenario, not evidence of zero real funding.
For long, positive rate pays and negative receives: quantity × settlement mark
× signed bps/10000. Events must be ordered, unique and strictly inside the
period. Equal entry/exit settlement timestamps are rejected because sequence
evidence is not implemented. Outside events are not silently included/dropped.
Unknown funding, missing borrow input or missing side rates are rejected.

NoTrade requires zero allocation/borrow, null prices and no events; all PnL and
costs are zero, no funding evidence is needed for a non-position. Invalid cost
scenario schemas are still rejected. Paired agents share scenario hash, period,
comparison key, instrument and capital; two longs also share reference prices
and funding inputs. Allocation may differ. Paired delta is challenger minus
incumbent net return; it is not bootstrap evidence or promotion authority.

Arithmetic uses local Decimal precision 50, bounded explicit decimal inputs,
decimal-string outputs, no exchange rounding claim. Precision/scenario contents
do not depend on ambient context in account(). Synthetic zero/base/stress
fixtures are deliberately asymmetric test constants, not recommended tariffs.
Legacy 16/24/40 aggregate roundtrip budgets are not accepted as side components
or automatically mapped. OHLCV last-trade opens are not proved mid prices.

Compile one pure file, check bps conversion directly, then checker --run and
--verify. A separate cash-proceeds oracle checks decomposition, fees and directions;
NoTrade, paired identity, funding signs/boundaries, unknown costs and unsupported
price semantics have regressions. Artifacts persist hashes/counters only.
This module does not implement rolling portfolio marks, terminal liquidation,
fee tiers, inverse contracts, actual funding coverage, causal availability of
funding/cost predictions or fitted net targets. Report is outside receipt.
