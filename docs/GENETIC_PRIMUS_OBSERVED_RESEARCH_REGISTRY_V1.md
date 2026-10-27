# Genetic Primus: metric dictionary and observed identity registry

## Scope and non-authority

This is an additive, metadata-only consolidation over the pinned catalog and
context links. It does not train, predict, replay trades, recompute PnL, decode
market parquet, open networks or use trading runtime. It does not rewrite old
JSONs, registries, contracts or ledger snapshots. All execution/credentials/orders/
paper/live/promotion/comparison-authority flags remain false.

The output is an **observed research registry**, not proof of complete history,
an immutable global launch registry, a count of independent financial trials,
or a calibrated DSR/PBO calculation. Its content-hash receipt is not a signature.

## Pinned inputs

The original inventory contains 1784 JSON artifacts with recorded byte identities.
Catalog SHA256:
`cafb0375dab4e0e608004c3fe9c216302ebc30331a745a0885d2c285ddbdb08b`.
Context-links SHA256:
`d2bec13b402512aa7d5ef08cef10eccd9789d7fb58909e52df1d855307f81c03`.
Both inventories are used as metadata, not evidence that the underlying market
data or execution were independently correct.

Nine objective-registry config snapshots are read only through byte-hash-bound
references recorded in the ledger. Their objective entries retain original
config-path/hash claims. Only a contract already byte-verified in bound context
inputs receives that narrower verification status; the other config contents
are not re-audited or silently equated.

## Metric dictionary

The dictionary covers all 14 canonical metric names and 233 normalized fields
in the frozen catalog. It has 26 observed adapter/field-tail/unit aliases. Each
definition has its own semantic digest independent of how often it was observed.

Definitions distinguish:

- Total reported net/gross, marked NAV, and hypothetical terminal-adjusted NAV.
- Mean/LCB per trade vs per calendar day. Bootstrap/alpha/bank provenance is still
  required; normalization alone does not establish confidence semantics.
- Sampling-unknown drawdown vs bar-close drawdown, neither asserted intrabar.
- Reported trades, fills and position transitions; counts do not imply real orders.
- Fees relative to initial capital, not every possible execution/funding cost.

Source percent divides by 100, bps by 10000, fractions remain fractions, counts
remain nonnegative integers. Missing/invalid values are not zero. Definitions
describe reported quantities and their unresolved caveats; they do not repair
accounting or change original values. A field-tail alias is not a full JSON pointer;
the catalog retains complete original source fields.

## Identity rules

Objective IDs are explicit original strings. Names/paths/families are never
converted into IDs, nor are strings case-folded or trimmed. Same ID with multiple
contract-hash claims is exposed as a conflict/variant set, not treated as equivalent
objective semantics. Root declarations and ledger/registry declarations retain
source path, byte SHA and field provenance.

Ledger collections have different roles: declared attempts, historically reported
attempts, diagnostics, engineering checks/updates, reviews, drafts, registrations,
objective declarations and deferred work. A registration is not execution; an
engineering revision is not an extra financial trial.

Only a result reference with a matching **file-byte hash** can join a verified
result-artifact group. Copies with the same verified result bytes can share that
group. Same objective/name or an unresolved reference cannot cause deduplication.
An explicit conflicting hash blocks a byte match. A separately recomputed semantic
result identity is distinguished from bytes, including the reversal result's
semantic SHA plus `result_file_sha256`. Semantic-only references are not used as
byte-identity groups.

Grouping is about observed artifacts, **not** statistical trials: one result may
contain multiple arms, fits, seeds or post-selection decisions; multiple results
may reuse a hypothesis/data. Declared candidate/search budgets remain reported
claims, never summed into a global testing denominator. Unknown or non-scalar
counts remain null. Unlinked financial artifacts are listed, not guessed into
trial records.

Ledger previous-file references are checked by bytes; cycles are detected with
a finite bounded traversal. Even a valid chain does not prove complete adaptive
history. Historical statistical trial count remains null, unknown historical
adaptive decisions true, global DSR/PBO and global immutability verification false.

## Reproduction and receipts

Use only `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8`, one short local
command at a time. No pytest/collector is necessary.

- Pure code: `src/exia/genetic_primus/research_identity_registry_v1.py`.
- Builder: `tools/build_genetic_primus_observed_registry_v1.py`.
  `--verify-only` checks metadata without writing.
- Regressions: `tools/check_genetic_primus_observed_registry_v1.py`.
- Artifact verifier: `tools/check_genetic_primus_observed_registry_artifacts_v1.py`.

Outputs use exclusive creation in a new `observed_research_registry_v1_20261005`
directory. The builder refuses overwrite. Completion binds inputs, output bytes,
module and builder. All 1784 originals, predecessor artifacts/context sidecars and
registry snapshots are checked before/after. The frozen future EMA config is
byte-checked through predecessor inputs, not evaluated or used to tune anything.
No large model/return arrays are retained during consolidation.

The artifact verifier independently reconstructs expected metadata products,
checks exact output bytes, receipts, provenance and one-group-per-observation
coverage. Partial outputs without a verified completion receipt are incomplete.

Next scope: resolve accounting/valuation and provenance gaps for a narrowly
specified comparable subset. If later authorized, future financial attempts need
explicit preregistered IDs/contracts/budgets and append-only launch/completion
receipts. This snapshot does not retroactively invent or register missing trials.
