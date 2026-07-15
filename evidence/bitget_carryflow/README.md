# Bitget CarryFlow evidence snapshots

This directory contains immutable, credential-free snapshots required to
reproduce the current CarryFlow diagnosis. Runtime collection continues under
`Retrodate/`, which remains intentionally ignored because its files grow while
the collector is running.

## Included segments

- `hourly_root_v1_20260712`: 58 contiguous complete hourly samples, 100%
  full8 context coverage. Collection ended after a power interruption. Its
  final replay passed execution parity but produced no candidate signals,
  fills, or closed trades. The replay summary, sealed manifest, and activation
  trace are included.
- `hourly_root_v2_recovery_20260715_precommit`: two contiguous complete hourly
  samples collected after recovery. It was finalized before committing so one
  tape never mixes collector source revisions.
- `hourly_root_v3_a1281fa_20260715_pre_policy_runtime`: four contiguous complete
  hourly samples with 100% full8 context coverage, collected on revision
  `a1281fab6f027bcf9ae78ed3635755bb582cb021`. It was finalized immediately
  before the production policy-runtime wiring changed.

Segments with separate hash-chain roots must not be concatenated or treated as
one promotion window. They may be replayed independently. A replay report
records both the evidence collector revision and the policy revision, so later
policy commits can evaluate an older immutable market tape without rewriting
its provenance.

No segment contains API credentials, account data, orders, or live-trade
authorization.
