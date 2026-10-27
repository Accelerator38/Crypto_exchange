# Synthetic completed labels → paired training-matrix manifest

Scope: bounded SYN_* fixtures, complete-row availability, no parquet, fit,
scaler statistics, costs, portfolio evaluation, network, pytest or runtime.
This separately versioned engineering contract cannot register a net objective
or requalify historical results. All nine safety flags remain false.

The label is a synthetic diagnostic `exit_open / entry_open - 1`. Both opens are
shifted by the old first-grid-open-strictly-after-ready rule and separated by
18 bars. Neither is a realized fill. Gross labels are not net profitability or
the final training objective. Cost_model is explicitly null, not zero.

The predecessor origin/prefix gate excludes unavailable/unpurged/incomplete
label candidates before target price validation or arithmetic. A complete panel
requires both rows for all symbols; publication is compared to train_asof−7bars.
The manifest distinguishes each symbol's label publication from the conservative
panel maximum; unequal publication times within cutoff are not rejected.
One millisecond beyond cutoff excludes the candidate. Earlier published revisions
are not silently inferred from a final snapshot. Eligible label OHLC must be
finite/positive and internally consistent. Unavailable values may be poisoned
without affecting paired X/y or deployment.

Each X row contains exactly return_12 and efficiency_12, recomputed from the
independently pinned feature prefix and checked against its predecessor hash.
Matrix row IDs/order are paired with y and checked against the scaler-input IDs.
The manifest binds X, y, paired IDs/X/y, entry/exit source-row hashes, dependency
ranges, publication and cutoff, contract and source snapshot. No prices, X/y or
label-value excerpts are persisted. Sign counts are synthetic diagnostics only.
Negative, zero, flat and abstention examples are retained; no forced trade filter.

Deployment uses an explicit five-field whitelist with only the two feature
values, symbol, asof, proposal and feature availability. It cannot inherit a
training row or target field. Values exist transiently for assertions, while
only the deployment hash is persisted. No model SHA is claimed: there is no fit.

Per-origin shape, distinct decision count and scaler plan are not minimum
opportunities/ESS or promotion evidence. Full8 market execution, net costs,
recency weights, actual compute/model availability and a registered market
objective remain separate requirements before any historical fit.

Workflow: compile the one new pure file, directly check gross_open_target on
synthetic constants, then checker --run and --verify. Outputs exclusive-create;
predecessor receipts and source/metadata banks remain immutable. Report lies
outside receipt. Checker persists only manifests/hashes/counters.
