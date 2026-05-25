# Panteon Flash - audit V8

Date: 2026-05-22. Updated after the 2026-05-24 cleanup pass.

## Original V8 result

The audit reported 7 open tasks and 0 blockers:

- P1: decompose `main_loop.py`.
- P1: add a unit test that disabled PnL LCB gates do not require LCB fields.
- P1: document that `_promotion_rejection_reason` treats LCB gates as opt-in.
- P1: document anchor, portfolio, terminal deny keys, denied regimes, and symbol degradation.
- P2: add voting-policy invariant tests.
- P2: add formal anchor vs switch-margin tests.
- P2: unify shadow-confirmation key formats.

V8 also recommended moving from pure selection toward an economic layer:
cost-aware sizing, funding-aware scoring/sizing, NoTrade as fee saving, and
volatility-adjusted exposure.

## Status after this pass

Closed:

- LCB gates are opt-in and covered by tests in `test_promotion_manifest.py`.
- `_promotion_rejection_reason` documents the opt-in contract.
- Voting-policy invariants are covered in `test_voting.py`.
- Anchor vs switch-margin priority is covered in `test_flash_allocator.py`.
- Shadow confirmation now accepts only `label` for actor fallback and
  `(label, symbol, action_name)` for symbol/action confirmation.
- New mechanisms are documented in `docs/PANTEON_FLASH_MECHANISMS.md`.
- Flash degradation state tracking moved from `main_loop.py` into
  `app/degradation_tracking.py`.
- Flash audit emission moved from `main_loop.py` into
  `app/audit_emission.py`, with compatibility wrappers left in place.
- Flash per-symbol decision orchestration moved into
  `app/decision_paths/flash.py`; `main_loop.py` now delegates through an
  explicit callback contract to avoid circular imports.
- Economic-layer knobs were added as opt-in, neutral-by-default settings:
  NoTrade fee-saving score, funding-aware score/risk tilt, actor edge sizing,
  and volatility-adjusted `risk_mult`.
- Latest dashboard publishing now disables itself after a Windows
  `PermissionError`, so a locked root-level PNG cannot spam logs or delay a
  long retrodate run after the run-specific artifacts are already written.

Remaining:

- The high-risk Flash pieces from V8 are closed. The remaining decomposition
  opportunity is broader non-Flash `main_loop.py` cleanup, which should be a
  separate maintenance branch because it does not change the Panteon Flash
  pre-live trading profile.

Blockers: 0.
