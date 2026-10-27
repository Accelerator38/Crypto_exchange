# Causal label schedule, separate from target values and fitting

The preceding materialization registered data availability lag 0, as explicitly
selected by the human. This does not register an execution/inference delay.
This component defaults that delay to None and denies label-value reads.

The checker constructs a price-free hypothetical schedule with delay 0 and
checks synthetic clock boundaries. It does not choose this scenario for use.
Entry is the first 4h open STRICTLY after decision readiness; even delay 0
therefore does not use the open simultaneous with the decision. Exit is exactly
18 bars after entry. The complete exit row becomes available at exit + one
bar + ingestion lag. Label maturity must be no later than training_asof minus
seven bars. Equality passes; one millisecond later fails.

Existing 18-bar candidate events are preserved. Training-prefix and deployment
context rows are not training samples. Excluded opportunities are retained with
timestamps/reasons, rather than dropped silently. No prices, outcomes, costs,
net targets, feature selection, scaler or model parameters are computed.
The schedule never grants read, fit, financial or trading authority.

`--check` creates immutable metadata outputs; `--verify` checks reproducibility.
Both verify the prior materialization receipt and derived file bytes without
parsing their numerical values or opening parquet. All nine safety flags remain
false. Old snapshots/availability assumptions are not original exchange-arrival
evidence and historical periods are not prospective.

A later explicitly chosen execution scenario may use the schedule for target
materialization with before/after source hashes, symbol/row dependency lineage,
retention of zero and negative outcomes, and no deployment-label leakage.
An OHLCV open is a reference price, not a demonstrated executable fill. Gross
open/open diagnostics must not become a net training objective by implication.
