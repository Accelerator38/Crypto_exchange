# Genetic_Primus V4 post-freeze audit

Status: `ADDITIONAL_REJECTION_ZERO_ENTRY_FOLDS`.

The strict run was already rejected by return, LCB and drawdown gates.
This audit adds an independent incidence failure: 2024_H1 and 2024_H2
contain mark-to-market returns from positions entered earlier, but no new entries.
Therefore those folds are not independent evidence of entry generalization.

Validation 2025, OOS 2026, paper and live remain closed.
