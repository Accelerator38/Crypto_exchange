# Synthetic origin → feature dependencies → scaler-input bank

No parquet, model/scaler fit, financial labels/returns, network, pytest or runtime.
Only SYN_* symbols and bounded synthetic candles are accepted. The existing
draft availability contract remains unregistered; this adapter changes none
of the frozen predecessors. All nine authority flags remain false.

Origin uses the pinned engineering rule train_asof=origin−1bar, decision in the
first origin bar, explicit ingestion/execution lags, three-day shifted exit and
seven-bar additional purge. Candidate training decisions are supplied explicitly;
the fixture calendar is not a new trading/retrain schedule registration.
The deployment entry/exit must also lie inside the half-open snapshot coverage.

Full-snapshot and independently pinned visible-prefix hashes detect substitution.
Every visible bar from the fixed history start must exist with the fixed revision,
all symbols, and publication by asof. Later publications do not backfill an earlier
decision. Complete-row scenario availability cannot precede close+explicit lag.
Physical future envelopes are validated but unavailable price values are not
validated or used in calculations (they are included only in snapshot hashing).

Return_12 and efficiency_12 require exactly 13 closes. EMA proposal uses the entire
fixed prefix (minimum 48 bars), with old pure ema kernel, first-close seeding,
spans 12/48 and slope lag 3. Sparse linear feature selection does not reduce
proposal dependencies to 13 bars. Flat paths abstain but are not removed from
the scaler-input bank merely for abstention. Values are hashed, not output.

Before constructing training feature rows, entry/exit whole-row availability
must meet train_cutoff and both label rows must exist for every symbol. A late
label row excludes the candidate for the whole fixture universe (conservative
fixed-complete-panel policy). No label price/return is calculated. Every origin
gets a newly derived scaler-input manifest: row IDs, feature-row hash and asof.
No statistics, scaler payload or coefficients are computed, reused or fitted.

Prefix pins must correspond exactly to the used training decisions plus the
deployment decision. The helper records physical row publication bounds but
does not measure feature compute completion or model availability. Actual
arrival evidence, immutable historical revision lineage, minimum opportunities/
ESS, scaler/ridge formulas, costs and statistical promotion gates remain outside
this adapter. Source flags and hashes do not independently prove real arrivals.

Compile one new pure file, check one pure context function, then use checker
--run and --verify. The checker verifies immutable predecessor receipt banks,
runs bounded deterministic synthetic cases, and exclusive-creates new results.
Mutation tests retain visible prefix pins; only the whole synthetic snapshot
identity is re-pinned when future input values change. Thus visible dependencies
cannot silently change under the same prefix identity. Report is outside receipt.
