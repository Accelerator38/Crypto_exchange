# Overlay v4 measurement artifact smoke

Дата: 2026-08-26. Один offline origin. Прогон предназначен только для проверки измерительных артефактов; PnL не используется для выбора или настройки objective.

## Результат

- Процесс завершён с exit code 0.
- Root status: `OFFLINE_ROLLING_3D_ENGINEERING`.
- Completed origins: 1.
- Challenger status: `NO_ELIGIBLE_CHALLENGER`.
- `challenger_deployable=false`.
- `promotion_approved=false`.
- `promotion_gate_passed=false`.
- `promotion_authority=false`.

## Validation top-K

- Finalists: 5.
- Eligible finalists: 0.
- Все 13 новых полей присутствуют: turnover, drawdown, direction/state/symbol concentration, ALLOW coverage, coverage gate, paired minimum/objective/absolute/baseline LCB, eligibility flag и blockers.
- Raw challenger сохранён только как diagnostic artifact.

## Functional drift

- Anchor rows: 512.
- Anchor SHA-256: `fddf3ec890b1b3631be7a18e025ddab5465729dfa75f3aa2b6a3fcd3e7f173fa`.
- Origin 0 functional action drift: 0.0, так как предыдущего challenger ещё нет.
- Origin 0 bounded-logit drift: 0.0 по той же причине.
- Manifest и model ledger содержат одинаковый anchor SHA.
- Outer gate декларирует семантику `fixed_hash_anchor_action_disagreement` и `fixed_hash_anchor_bounded_logit_distance`.
- Deployment action-mix drift сохраняется отдельно и больше не подменяет functional drift.

## Safety

`credentials_allowed=false`, `paper_allowed=false`, `live_allowed=false`, `orders_enabled=false`, `promotion_authority=false`.

Новое обучение или повторный pilot6 по результатам этого smoke не разрешались и не выполнялись.
