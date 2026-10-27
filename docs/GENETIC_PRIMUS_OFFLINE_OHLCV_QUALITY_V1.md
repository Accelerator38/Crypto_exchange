# Counter-only historical OHLCV quality audit

Scope: exactly the eleven historical parquet files in the completed
offline_input_verification_v2_20261007 receipt, approved for local reads.
No new sealed data, fitting, returns, price examples, collector, pytest or network.
All nine safety flags remain false. The completed previous sources are immutable.

The pure helper checks nulls, numeric type (excluding bool), finite values,
positive OHLC, nonnegative volume, exact high/low inequalities and integer
source_rows=240. Zero volume is a diagnostic, not an automatic rejection.
No post-data tolerance selection. Counters distinguish invalid rows from
potentially overlapping field violations. They are not profitability metrics.

First compile only the new pure helper; then directly check a synthetic pure
function. Prepare validates predecessor receipts and local metadata, and runs
deterministic pure regressions. Each --file N command decodes only one approved
file, at most 8 MiB / 200000 rows, batches of 4096, local BufferReader, no threads
or pre-buffer. Optional NumPy/pandas/exia/network-client imports are blocked.
An audit hook blocks socket and process launches; this is not an OS sandbox
or complete native-library attestation. No decoded price values are retained.

Prepare pins sidecars, linked panel/archive JSON metadata and relevant producer
source. It checks output file claims against the approved identities. Linked
raw 1m parquet/chunks/archive contents are NOT read or independently verified.
Matching dataset digest references only establish metadata consistency.
For September derived datasets the producer byte SHA is explicitly bound;
older timeframe sidecars lack such a producer binding. Current old resampling
code is static explanatory evidence, not proof of its historical execution.

Local resampling code floors timestamps to a 4h bucket start and counts 240
source rows. Such counts do not prove unique 1m bars or source authenticity.
Fully aggregated high/low/close/volume must not be assumed available at bucket
start. Bucket-close availability plus a separately registered ingestion/latency
assumption is needed unless actual field arrival evidence exists. Neither
exchange arrival nor late revisions are established by this audit. Do not
silently alter the frozen label/preregistration contracts.

Finalize and verify read only metadata/counter JSON and pinned source files,
never parquet. Receipt binds preparation, eleven file summaries and result.
File hashes are checked before/after separate commands, not a simultaneous
global snapshot. Outputs are exclusive-create; no overwrite or retry mutation.
