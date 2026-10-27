# Overlay v4 measurement artifact smoke r2

Дата: 2026-08-26. Один offline origin. PnL этого повторного origin не анализировался и не использовался для настройки objective.

## Проверенные инварианты

- Exit code: 0.
- Challenger status: `NO_ELIGIBLE_CHALLENGER`.
- `challenger_deployable=false`.
- Validation top-K: 5 finalists, eligible: 0.
- Raw challenger и stress challenger имеют `evidence_role=diagnostic_outer_deployment`.
- Active, active stress и фиксированные baselines сохраняют `evidence_role=outer_deployment`.
- Completed eligible outer origins: 0.
- `no_eligible_challenger_evidence=true`.
- `promotion_gate_passed=false`.
- `promotion_authority=false`.

## Top-K ledger

Сохранены все новые поля каждого finalist: turnover, drawdown, direction/state/symbol concentration, ALLOW coverage, coverage gate, paired minimum/objective/absolute/baseline LCB, eligibility и blockers.

## Functional drift

- Hash-anchored rows: 512.
- Anchor SHA-256: `fddf3ec890b1b3631be7a18e025ddab5465729dfa75f3aa2b6a3fcd3e7f173fa`.
- Manifest и model ledger используют один anchor SHA.
- Семантика: bounded logit distance и action disagreement на одинаковых raw rows после frozen scaler каждой модели.
- Deployment action-mix drift хранится отдельно.

## Safety

`credentials_allowed=false`, `paper_allowed=false`, `live_allowed=false`, `orders_enabled=false`, `promotion_authority=false`.

Таким образом, положительный диагностический PnL больше не может попасть в outer promotion evidence при отсутствии eligible challenger.
