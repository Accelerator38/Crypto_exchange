# Genetic Primus V6 — engineering smoke 2025-02-01

Дата выполнения: 2026-08-27.

## Назначение и статус

Выполнен один offline origin на ранее раскрытом интервале `[2025-02-01, 2025-02-04)`. Прогон предназначен только для проверки вычислительного pipeline, ledger, hashes и fail-closed gates. Он не является outer evidence, не входит в objective registry и не может использоваться для настройки V6 или promotion.

Основной V6 sealed interval `[2026-01-01, 2026-07-01)` не открывался и остаётся `UNOPENED`.

## Результат выполнения

- статус: `OFFLINE_ROLLING_3D_ENGINEERING`;
- завершено origins: 1;
- решений в ledger: 136;
- active trades: 0;
- active return: 0 bps (NoTrade);
- challenger status: `NO_ELIGIBLE_CHALLENGER`;
- challenger deployable: false;
- promotion approved: false.

Диагностический, не допущенный к deployment challenger:

- net return: +92.05 bps;
- stress-cost return: +88.01 bps;
- trades: 4;
- max drawdown: 1.00%;
- actions: 2 short, 124 flat, 10 long.

Fixed baselines на том же трёхдневном origin:

- NoTrade: 0 bps;
- Momentum_return_1: +277.29 bps, 67 trades, max drawdown 12.30%;
- MeanReversion_return_1: −754.00 bps, 67 trades, max drawdown 12.83%;
- RelativeMomentumRank_return_12: −219.57 bps, 13 trades, max drawdown 2.17%.

Эти числа описывают только один уже раскрытый трёхдневный отрезок. Они не оценивают generalization и не доказывают превосходство challenger либо momentum.

## Gate diagnostics

- paired selection LCB: +0.10 bps;
- independent audit LCB: −16.69 bps;
- validation-eligible finalists: 0;
- predeployment training gate: failed;
- temporal audit: failed.

Причины отказа:

- `temporal_audit_lcb`;
- `action_diversity`;
- `feature_stability`;
- `no_eligible_validation_finalist`;
- `insufficient_completed_outer_origins`.

Положительный diagnostic deployment return не отменяет отрицательный independent audit LCB. Fail-closed evaluator правильно оставил active model в NoTrade.

## Проверки целостности

- 34 984 supervised rows;
- 13 119 overlay opportunity rows;
- scaler rows: 35 040;
- purge gap: 28 часов, то есть 7 баров по 4 часа;
- deployment horizon: 72 часа;
- все 136 решений содержат один и тот же active model SHA, совпадающий с origin ledger;
- все 10 artifact hashes совпали;
- rolling snapshot явно содержит `non_evidentiary=true`, `sealed_interval_touched=false`;
- objective registry SHA в manifest равен null, поэтому smoke не маскируется под зарегистрированный sealed objective;
- credentials, paper, live, orders и promotion authority: false.

## Идентификаторы

- base V6 config SHA: `aea9dbb1b3acd926afc1b0509742f870f4fbdf844036e63f25d3375a91b07d93`;
- engineering override SHA: `98a3e5ac4ce196102e623d60548b94f6953156eb34dd6b7c326fc6532ab2f927`;
- run manifest SHA: `0a2340682a3800783d52a65ff1cd655a0cd75d62b77a771691d90d2022c7eeda`;
- rolling snapshot SHA: `8af8f3c5c1e09194e6796f40fb8aaad2c60e05b5fe5269e6c3b67a2bca04ab52`;
- model manifest SHA: `316dbafc8b6ecec1732cfd99649a11eb31e3a8c38d8d31fcf6d5e9956306b02d`.

## Вывод и следующий этап

Engineering pipeline V6 технически готов к длинному прогону: provenance, purge, model SHA ledger, abstention и fail-closed gates работают. При этом smoke выявил ожидаемую статистическую проблему — кандидат не прошёл независимый audit и stability gates.

Следующее решение — открывать ли заранее зарегистрированный 60-origin sealed interval V6. Если его открыть, objective, features, thresholds, seeds и gates менять по результатам уже нельзя; прогон должен быть единственным полным evaluation данного objective.
