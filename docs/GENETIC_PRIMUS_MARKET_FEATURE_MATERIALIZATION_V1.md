# Historical feature materialization, separate from labels and fits

Human scenario choice is required before --prepare 0|1 or any --file command.
--check-code and --verify-code read metadata and fabricated fixture rows only;
they do not authorize a scenario or import the native decoder. Default lag is
unset, not zero. Approval of this data preparation is not model/financial/live
authority: all nine safety flags remain false.

The scenario explicitly declares final pinned snapshots and does not claim
original arrivals/revisions. Only the existing 11-file approved manifest is
referenced. Sources are ordered by their frozen key-audit time ranges, gaps/
overlap rejected. For the single Sep22 decision only nine files are needed;
the Sep22..27 tails are not decoded. The last included timestamp is decision
minus one bar minus explicit ingestion lag. No new sealed data or sources.

Every --file N handles exactly one source: size/SHA before/after, capped local
BufferReader, no threads/prebuffer, strict footer/schema and canonical ordering.
Arrow may decompress later values within a batch; only eligible slices are
converted to Python and fed to state. No future values enter features/artifacts.
There are no price examples or feature values in stdout; exceptions print type
only. Optional NumPy/pandas/exia/network clients are blocked by the old decoder.

Events retain the anchored 18bar training-candidate calendar, with separate
training-prefix and deployment contexts. Candidate features are NOT completed/
purged training opportunities. No labels, costs, returns/PnL of strategies,
scaler statistics, models, orders or search are computed. Label materialization
needs separate execution-delay registration.

EMA state continues across files. A machine-only JSON checkpoint contains
EMA values and the last 13 OHLCV rows per symbol, including historical prices;
this is necessary for continuity and is explicitly disclosed, not a price-free
artifact claim. Local JSONL files contain derived feature values and dependency
metadata. No user source files are modified. Artifacts exclusive-create, no
overwrite/delete; failed incomplete parts remain and prevent blind retries.

Parent/checkpoint identities, input bank and derived files are verified. Saved
state is trusted only through this identity chain, not by EMA recomputation
from its short tail. Per-file publication checks do not prove a simultaneous
global source snapshot or OS/native-memory attestation. No real funding/cost
or arrival calibration follows from this materialization.

Finalize/verify read only local generated artifacts and source/metadata pins,
no parquet reread. Compile the pure planner/checkpointer first, then one direct
unset-scenario check, then metadata/fabricated --check-code and --verify-code.
Do not execute --prepare or --file while the scenario question is unanswered.
