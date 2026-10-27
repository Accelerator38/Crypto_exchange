# Статический инвентарь тестов Genetic Primus

Дата: 2026-09-08. Область: `tests/*genetic_primus*.py`. Все 51 файла разобраны через Python AST как текст; найдено 205 определений test-функций. **Ни один тест в ходе этого аудита не исполнялся.** Импорт collector/выполнение fixtures не производились. AST-синтаксическая читаемость не доказывает успешность тестов или их полноту.

Часть функций вызывает другие тесты или параметризована; 205 — число определений, не число независимых проверок и не число PASS. Сохранённые результаты прежних запусков отделены от этого инвентаря.

В таблице имя дополняется префиксом `tests/test_genetic_primus`.

| Суффикс файла | Test-функций | Назначение/граница доказательства |
|---|---:|---|
| `_accounting_v2.py` | 13 | Fixed quantity, PnL/fees, terminal close, drawdown, causality, missing bars, insolvency; не биржевое исполнение |
| `_attribution_august_v1.py` | 4 | Разложение gap/holding/fees и trade buckets |
| `_attribution_august_v2.py` | 2 | Повтор base-регрессий и допустимые горизонты HAC |
| `_entry_only_ablation_v1.py` | 6 | Stateful entry permission, resets, prefix causality, synthetic cash ledger |
| `_experiment_matrix_v4.py` | 3 | Контракт ограниченной матрицы экспериментов |
| `_hac_adapter_v1.py` | 7 | Квалифицированный статистический адаптер |
| `_measurement_3d.py` | 2 | Трёхдневные измерительные конструкции |
| `_measurement_lab_v2.py` | 9 | Исследовательская measurement v2 линия |
| `_model_comparison_v1.py` | 2 | Старый сравнительный протокол агентов |
| `_offline_safety_regressions.py` | 4 | Baseline units/SHA, strict median, resume guards |
| `_overlay_v2.py` | 4 | Legacy overlay v2 |
| `_overlay_v3.py` | 4 | Legacy overlay v3 |
| `_overlay_v4.py` | 5 | Stabilization/anchored recency контракты |
| `_overlay_v5_contract.py` | 7 | Absolute DD для NoTrade, stability/signs, concentration, DSR/PBO, hysteresis/cooldown |
| `_overlay_v6_contract.py` | 8 | Sealed origins, registry ancestry, units, relative momentum, ESS/engineering override |
| `_promotion_v4.py` | 3 | Legacy promotion-правила; не доказательство текущего допуска |
| `_prospective_contract_v1.py` | 4 | Замороженный prospective contract |
| `_prospective_manifest_v1.py` | 3 | Data manifest/provenance |
| `_recency_fitness.py` | 5 | Recency weighting/fitness |
| `_retrained_august_v1.py` | 4 | Temporal bounds, future invariance, search budget, strict screen |
| `_rolling_3d.py` | 6 | Rolling evaluator/scaler/model identity |
| `_simplification_v4.py` | 7 | Ограничение модели и baseline-comparisons |
| `_v1.py` | 8 | Раннее поколение модели/движка |
| `_v2.py` | 7 | Раннее поколение модели/движка |
| `_v3.py` | 7 | Раннее поколение модели/движка |
| `_v3_generalization_audit.py` | 1 | Протокол generalization audit |
| `_v3_terminal.py` | 1 | Legacy terminal protocol |
| `_v4_foundation.py` | 4 | V4 measurement foundation |
| `_v4_stage_gate.py` | 2 | V4 stage transitions |
| `_v5_contract.py` | 2 | V5 event/data contract |
| `_v5_data_gate.py` | 2 | V5 data readiness |
| `_v5_data_source.py` | 3 | V5 source handling |
| `_v6_contract.py` | 1 | V6 event contract |
| `_v6_data_gate.py` | 1 | V6 data gate |
| `_v6_data_source.py` | 2 | V6 source handling |
| `_v6_feasibility.py` | 3 | Event feasibility |
| `_v6_stage_gate.py` | 2 | V6 stage transitions |
| `_v7_contract.py` | 1 | V7 event contract |
| `_v7_data_gate.py` | 2 | V7 data gate |
| `_v7_data_source.py` | 2 | V7 source handling |
| `_v7_feasibility.py` | 2 | Event feasibility |
| `_v7_proposals.py` | 5 | V7 proposals |
| `_v7_stage_gate.py` | 2 | V7 stage transitions |
| `_v8_clean.py` | 5 | Clean-room contract/safety |
| `_v8_offline_replay.py` | 1 | Collector replay; **не запускался**, ранее проблемный execution path |
| `_v8_prospective.py` | 3 | Event prospective logic, не допуск OHLCV overlay |
| `_v8_supervisor.py` | 2 | Ограниченный supervisor, не квалификация длительного live runtime |
| `_v8_tape_checker.py` | 1 | **Не запускался**, прямой пользовательский запрет сохранён |
| `_validation_ci_v1.py` | 8 | Statistical qualification machinery |
| `_validation_power_v1.py` | 9 | Synthetic null/power experiment machinery |
| `_warmstart_august_v1.py` | 4 | Search parity, anchors, budget, frozen isolation |

## Что тесты уже защищают

Существуют полезные проверки бухгалтерского тождества и невозможных состояний, причинности признаков/решений, SHA/provenance, фиксированных временных границ, бюджетов поиска, отдельного audit seed, некоторых promotion-контрактов, abstention, drift, concentration и restart/checkpoint guards. Последние маленькие ablation-наборы были прямо исполнены в предыдущих задачах и прошли; весь исторический набор заново не запускался.

Наличие теста legacy v4/v5/v6 не означает, что последняя lab/retrain/ablation линия вызывает ту же реализацию. Это проверяется отдельно через release-contract и call graph.

## Недоказанное покрытие перед продом

В обследованных Genetic Primus тестах и accounting_v2 не найдены сценарии реальных partial fills, lot/tick/min-notional, funding, borrow/margin/mark-price, неясного исхода submit после timeout, восстановления биржевых ордеров по client ID и повторных ack/fill после reconnect. Ключевые слова не являются формальной полнотой поиска, поэтому это формулируется как **отсутствие доказанной квалификации**, а не утверждение об отсутствии любой такой функции в репозитории.

Необходимы три отдельных уровня:

1. Pure/unit: schemas, units, cutoffs, stale/invalid inputs, deterministic state machine, model SHA.
2. Offline execution/fault: partial fills, rounding, out-of-order events, duplicates, reject/timeout, restart между submit/ack/fill, повреждённый checkpoint, disk-full, risk limits и reconciliation. Только детерминированные часы и ограниченные сценарии.
3. Отдельно разрешённый shadow/paper: parity между реальным потоком доступных сигналов и replay, measured latency/costs, recovery и operational soak. Это не заменяет независимый статистический тест edge.

Запрет на проблемные collector/tape/replay-запуски остаётся действующим. Их нельзя добавлять в общий unbounded pytest-suite под предлогом полного покрытия.
