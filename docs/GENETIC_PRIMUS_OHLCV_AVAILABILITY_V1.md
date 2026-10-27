# Complete-row OHLCV timing contract (draft)

This is a separately versioned engineering contract. It does not edit or
requalify the frozen research_execution_contract, preregistration or past runs.
No training, financial evaluation, collector, pytest, parquet read or network.

Scope: a 4h row timestamp names the start of its complete OHLCV bucket. There
is no independently observed open stream. Every field, including open, uses
the scenario boundary `start + 4h + explicit ingestion lag`. Lag has no default;
the draft JSON leaves lag and evidence/revision identities null. Synthetic
zero/positive lag cases are not calibrated or registered market assumptions.
The contract remains draft and fit stays false even if evidence digests are
supplied: syntax or a digest claim does not independently verify arrival.

Closed-prefix planning handles exact cutoff equality, sub-bar lag, multi-bar
lag and the half-open snapshot boundary. It is a calendar plan, not proof that
rows exist. Each feature supplies all contiguous dependency bars; gaps,
duplicates and reordering are rejected. All must be covered and available by
the decision. Sparse feature selection means separate features/gates, not
omitting dependencies of a rolling indicator. This helper does not calculate
feature values, scaler statistics or fits; any future builder must propagate
the actual dependencies, not only the last bar.

Entry remains the old contract's first grid open strictly after decision plus
explicit execution delay; exit is shifted equally, preserving 18 bars / 3 days.
These are target prices for a future label, not executable fills. Label
availability is the maximum of the complete entry/exit row availability.
Training eligibility additionally requires availability <= train asof - 7 bars
and both rows strictly inside the half-open input coverage. Equality is allowed;
one millisecond earlier is rejected. The seven-bar purge is additional to
label maturity, not a replacement for the 18-bar horizon.

Pure counterfactual tests check compatibility with the old execution_window
and completed_label_gate while passing complete-row availability explicitly.
No claim that the previous evaluator actually applied this policy. A known
later publication or late revision invalidates an earlier assumed lag unless
the registered policy handles it; final historical data are not original
arrival evidence. Source_rows=240 is not minute uniqueness evidence.

Run order: compile only ohlcv_availability_v1.py; direct synthetic pure function
check; then the short stdlib-only checker --run, followed by --verify. The
checker verifies predecessor receipts and source/metadata identity banks,
never raw market data. Output creation is exclusive, no overwrite. Report is
outside the completion receipt. All nine safety flags remain false.
