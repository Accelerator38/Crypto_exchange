# Historical gross diagnostic labels, not net training targets

The human confirmed ingestion lag 0 and the separate execution scenario:
instant decision computation, entry at the first 4h open STRICTLY after decision
readiness, exit exactly 18 bars after entry. These are engineering assumptions,
not measured inference latency or exchange fills. Open is a reference price.

Preparation reconstructs the frozen causal schedule and verifies its receipt.
Only 571 eligible candidate events are used; the unpurged event and two context
events are excluded. All eight symbols and all outcomes are retained, including
zeros, losses and proposal abstention. No proposal/outcome-based row filtering.
The complete exit-row availability and seven-bar purge remain mandatory.

`--check-code` and `--verify-code` use metadata and fabricated prices, no parquet.
`--prepare` separately records human authorization for this diagnostic raw-data
step. Nine consecutive `--file N` commands each read exactly one previously
approved source, verify byte SHA before/after, and retain only scheduled opens
with full-row canonical SHA and immutable source identity. Native reads reuse
the pinned Arrow descriptor, bounded batch size, no threads/prebuffer/pandas.
The backend can decompress later rows inside a batch; only prefix rows enter
Python records. Existing or partially written outputs are never overwritten.

`--finalize` joins existing features and entry/exit opens. Sample lineage includes
feature file SHA/line, canonical feature-row SHA/prefix SHA, and both observed
row identities. Gross return uses Decimal precision 50 on decimal representations
of source numbers. It is not a costs-adjusted, paired strategy return or PnL.
Local machine artifacts contain historical opens and gross labels; stdout only
contains bounded counters. JSONL artifacts and metadata have fixed size caps.

`--verify` reconstructs the same samples from derived opens/features and checks
all byte identities and receipt, without rereading parquet. No label values for
deployment are read or joined. No costs, net objective, scaler/model fits, search,
network, collector, trading runtime, keys or orders. All nine safety flags remain
false. Historical snapshots are not prospective or original-arrival evidence.

Before fitting, register a reference-price bridge and net costs/accounting.
These gross diagnostic labels must not be silently promoted to a net objective.
