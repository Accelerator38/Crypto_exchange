# Genetic Primus overlay v3 — engineering pilot6

Дата прогона: 2026-08-26. Режим: строго offline research. Сеть, collector, supervisor, paper/live runtime и ордера не запускались.

## Итог

Последний `binary sparse overlay` завершил один smoke-origin и шесть последовательных 3-дневных origins. На открытом pilot-интервале challenger получил `+205.153 bps` по сумме срезов (`+206.626 bps` при последовательном compounding), был положительным в 5 из 6 origins и превзошёл momentum (`+117.047 bps` по сумме; `+91.629 bps` compounded). Stress-вариант challenger остался положительным: `+60.482 bps`.

Модель не promoted. `promotion_gate_passed=false`, `promotion_approved=false`, `promotion_authority=false`.

## Проверка плана

1. Старый objective не переобучался; использован новый v3 temporal/overlay objective.
2. Исправлена семантика folds: inner training folds больше не считаются outer slices; rolling 730d применяется согласованно.
3. Использован linear binary ALLOW/ABSTAIN overlay с максимум 8 активными features, feature/parameter sparsity и ranking по paired incremental return.
4. Четыре детерминированные overlay-v3 регрессии выполнены напрямую без pytest, сети и collector: `4 passed`. `py_compile` изменённых Python-файлов также прошёл.
5. Один origin smoke завершён: challenger `+80.599 bps`, stress `+39.311 bps`, momentum `+471.699 bps`.
6. Шесть origins завершены с exit code 0 и полным root manifest.
7. Новый long outer interval не запечатан: обязательные stability/promotion gates не пройдены.

## Исправленные дефекты, найденные реальным прогоном

- Evolution на inner-интервале ошибочно получал bootstrap bank следующего temporal-validation интервала. Теперь island selection использует inner folds и общий inner selection bank.
- Incumbent validation scorer на origins 2+ получал inner bank при validation timestamps. Теперь challenger и incumbent сравниваются на одном validation calendar/bank.
- Overlay v3 попадал в несовместимый legacy promotion gate. Legacy gate оставлен только для v1/v2; v3 использует temporal diagnostic gate и outer gate.
- Runner передавал сокращённый origin ledger в outer gate вместо model ledger с drift/predeployment полями. Теперь используется `models.parquet`.

Все несоответствия были остановлены fail-closed проверкой timestamp/hash; повреждённые/неполные каталоги не переиспользовались и не удалялись.

## Протокол

- Deployment: 6 origins по 3 дня, 2025-01-01 — 2025-01-19 UTC.
- Retrain cadence: 3 дня.
- Training window: rolling 730 дней.
- Label purge: 7 четырёхчасовых баров = 28 часов.
- Для origin 0: inner 2023-01-02 00:00 — 2024-08-31 16:00; validation 2024-09-01 20:00 — 2024-10-31 20:00; audit 2024-11-02 00:00 — 2025-01-01 00:00 UTC.
- Scaler fit: только inner interval; в origin 0 — 35,040 строк.
- Search: 5 islands × population 4 × 5 generations; 71–76 фактически уникальных selection genomes на origin при верхней границе 100 и полном adaptive budget 107.
- Выбранный ensemble: размер 1 во всех 6 origins.
- Архитектура: binary linear, 30 inputs, 1 ALLOW output, 31 raw parameter; после sparsity у выбранных моделей 1–9 активных parameters и 0–8 активных features.

## Результаты открытого pilot6

| Агент | Сумма, bps | Положительных origins | Trades | Compounded, bps | Aggregate max DD |
|---|---:|---:|---:|---:|---:|
| GeneticPrimusChallenger v3 | +205.153 | 5/6 | 144 | +206.626 | 2.969% |
| Challenger stress | +60.482 | 4/6 | 144 | +60.515 | 3.560% |
| Momentum return_1 | +117.047 | 3/6 | 424 | +91.629 | 11.610% |
| NoTrade / active incumbent | 0.000 | 0/6 | 0 | 0.000 | 0.000% |
| Mean reversion return_1 | -1874.407 | 0/6 | 424 | -1754.088 | 17.490% |

Challenger по origins, bps: `+80.599, ~0.000, +57.638, +4.099, +10.301, +52.516`.

## Сравнение с прежними попытками

На точно том же dataset и тех же шести 3-дневных origins прежний overlay v2 получил `-81.657 bps`, 0/6 положительных origins и 12 trades. Новый v3 улучшил сумму на `+286.810 bps`, достиг 5/6 положительных origins и обошёл momentum на `+88.106 bps` по сумме срезов.

Старый weekly rolling-730 uniform outer6 дал challenger `-1074.164 bps`, active `-287.822 bps`, momentum `-1093.918 bps`. Это только контекст: cadence и deployment interval отличались, поэтому прямой ranking с текущим pilot некорректен.

## Почему promotion запрещён

Положительные outer pilot-метрики недостаточны:

- completed outer origins: 6 при минимуме 60;
- outer paired LCB: `+20.759 bps`, independent-seed audit LCB: `+20.759 bps`, stress LCB: `+3.528 bps`;
- positive outer fraction: 83.33%;
- внутренний temporal predeployment gate: false на всех 6 origins;
- per-origin validation LCB: `-35.988, 0.000, -21.049, -5.796, -10.620, -6.622 bps`;
- per-origin independent audit LCB: `-82.986, 0.000, -63.665, -22.339, -128.808, -49.397 bps`;
- fills/day: 16.0;
- max action total variation между retrains: 0.5074;
- max normalized model drift: 2.4825;
- не пройдены drawdown-vs-incumbent, turnover, direction/state/symbol concentration, action drift, model drift и temporal predeployment gates.

Это сочетание означает: результат на открытом январском pilot улучшился, но внутреннее temporal evidence указывает на нестабильность/selection bias. Открывать новый sealed interval сейчас методологически рано.

## Воспроизводимость и safety

- Dataset SHA-256: `f66529b4135d04ddc35b7a6535667d1c6d53cd39c34adb86f23e54d4082b2de7`.
- Dataset manifest SHA-256: `5d79526614a41f1b22ca963b9fb6c07584b8b60eb0d49d76c574d6fdcc666dd5`.
- Rolling protocol SHA-256: `a5203c99e7aa4b6b5ee7634c8bf4821d1235849704d97651f1deeaf47a55e06d`.
- Promotion protocol SHA-256: `461b27a7bda03aafadf44cb5aaa3a7106aa744081fbf7f35c258fe33a66c2bad`.
- Selection, temporal-validation и audit bootstrap banks сохранены отдельно для каждого origin вместе с cutoff, folds SHA и content SHA.
- Model ledger содержит 6 различных challenger model SHA; 816 active decisions не имеют пустого model SHA. Active SHA один, потому что ни один challenger не был promoted.
- Последний challenger model SHA: `3d4f18fe99417e2ac9c1e0ec8157df552009e0c0682c52b9d76c65658f5fea6e`.
- `credentials_allowed=false`, `paper_allowed=false`, `live_allowed=false`, `orders_enabled=false`, `promotion_authority=false` для всех origins.

## Следующий обоснованный шаг

Не открывать sealed outer interval и не увеличивать search budget. Сначала уменьшить turnover/concentration и retrain drift, затем добиться положительного validation и independent-audit LCB хотя бы на большинстве origins текущего открытого engineering-периода. Только после этого зафиксировать неизменяемые config/data/protocol hashes и открыть длинный outer interval минимум на 60 трёхдневных origins.
