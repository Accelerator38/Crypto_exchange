# Reference-open accounting: explicit bridge, unknown costs block targets

This layer does not repurpose synthetic-mid accounting as market execution.
The reference is the observed historical OHLCV open, NOT mid or an observed
fill. Price drag is an explicitly supplied relative adverse adjustment against
that reference; it is not an inferred half-spread. No market-cost defaults.

Pure arithmetic accepts complete explicit inputs for Long/NoTrade. Quantity is
allocated quote / (entry adjusted price * (1 + entry fee)); entry fee is reserved
inside the allocation. Fixed quantity is liquidated at the registered 18-bar
exit. Entry/exit fees each use that side's adjusted notional. Funding requires
an ordered complete list of settlement marks and signed rates, strictly inside
the period; ambiguous boundary settlements are rejected. Positive funding rate
debits Long, negative credits it. Borrow is an explicit quote debit, never an
implicit zero. NoTrade must have zero exposure/costs and no reference prices.

Cash recovery and gross-minus-attributed-cost paths reconcile within Decimal
precision tolerance. Paired actions require the same quote capital, interval,
symbol and reference context, with identical observed inputs for Long/Long.
This is per-interval accounting, not a continuous full8 portfolio/equity ledger,
intraperiod drawdown, observed exchange rounding, latency or fill attestation.

The immutable v1 contract is a DRAFT: nine parameter registrations are None.
It cannot enable market net calculations or fitting. Fixture rates are fabricated
unit-test inputs, not cost estimates, recommendations or a normal/stress market
scenario. Old aggregate round-trip bps are not split into per-side costs.

`--check-code` and `--verify-code` hash prior derived artifacts without parsing
their numeric values and use fabricated arithmetic inputs. `--build` parses the
previously completed gross-label samples, verifies causal open/purge alignment
and attaches source SHA/line/canonical sample identity to blocked net requests.
Every net/stress/paired target stays None. All samples/outcomes/proposals remain;
unknown fees/funding/capital/borrow do not turn into zero costs.
`--verify` rebuilds the same requests and checks output bytes/receipt. No raw
parquet/native decoder, numpy/pandas, network, models, runtime, keys or orders.
Local machine files contain reference opens; stdout contains only counters.

Next choose measured sources or an explicitly authorized conditional engineering
scenario. Any such registration must be separate and immutable, with its own
provenance/assumption labels. It must not alter this completed draft/checks or
grant training, promotion or trading authority. Net training objective,
threshold and portfolio allocation still require independent registration.
All nine safety flags remain false.
