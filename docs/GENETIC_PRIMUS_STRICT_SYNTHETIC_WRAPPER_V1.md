# Genetic Primus: strict synthetic wrapper v1

Дата: 2026-10-06. Продолжение synthetic accounting probe; старые контракты, simulator и результаты не изменяются.

## Завершённая задача

Создан отдельный offline wrapper с проверкой artificial inputs **до callee** и entry-model identity sidecar для каждого решения. Wrapper ограничен тремя ранее зарегистрированными payloads: carry, retarget и abstain. Это не универсальный market evaluator, не online ledger и не разрешение обучения.

Реализация: `src/exia/genetic_primus/strict_synthetic_wrapper_v1.py`. Pure validation импортирует только stdlib и использует явно переданный API инженерной спецификации, без research package initializer.

## Строгие входы

Проверяются plain JSON types/caps, отсутствие nonfinite/неподдержанных объектов, envelope и column schemas, полный упорядоченный fixed-universe grid, цены, integer timestamps, signed discrete targets, causal signal/model availability и reference cost scenario. Boolean direction или boolean price не подменяются значениями 1/0.

Для всех трёх origins проверяются lowercase SHA256, hash каждого **synthetic** model payload, соответствие модели строке решения и model availability до signal/fill. Input/model-bank/boundary-plan semantic hashes сохраняются в proof. Это не проверка реально обученных weights/scalers, labels или elapsed training latency.

Весь payload дополнительно должен совпасть с уже закреплённым input SHA предыдущего probe. Даже корректные по schema новые цены/targets отвергаются как незарегистрированные. Symbols только `SYNTH_A/SYNTH_B`, scenarios только normal 16/stress 24 bps; нельзя подставить рынок или другую liquidation policy.

Ошибки validation проверены с запретным callee: ошибка совпадает с pure validator и callback не вызывается. Model banks prevalidated ещё до загрузки numeric callee в builder.

## Callee identity

Разрешён только namespace созданный предыдущим verified-source loader. Wrapper сверяет SHA accounting-файла, compiled code identities simulator и source-defined CashSleeve methods, а также их globals namespace. Неизвестный callable, замена simulator или метода отвергаются до расчёта.

Параметры задаются явно: bars/targets/symbols, start/end/timeframe/cost, пустой force-flat tuple и details=true. Старый synthetic probe loader сохраняет no-network/no-process/write-scope guard и ограничения native threads. При dependency import возможен заблокированный hostname metadata request, не разрешённый socket/connect.

Эти проверки не являются OS sandbox или полным proof всех loaded globals/generated methods/compiled dependencies. Полный runtime graph остаётся unverified; версия и init-file SHA NumPy/pandas не закрывают binary/transitive identity.

## Sidecar и происхождение полей

Original decisions и trades не переписываются. Sidecar содержит для каждой decision:

- Active `decision_model_sha256` из actual decision, сверенный с bound input model.
- `position_entry_model_sha256_before/after`; null означает flat.
- Entry reference и indices actual closed trades до/после решения.
- Actual quantity и позицию, с проверкой carry без resize/reset.
- `field_origin=wrapper_derived_bound_to_actual_closed_trade`.

При target change и terminal close проверяются entry SHA/timestamp/signal/price/quantity/direction и actual exit price/reason. Все trades должны быть атрибутированы ровно один раз. При carry сохраняется первоначальная entry identity несмотря на смену модели решения. NoTrade не получает искусственной entry identity.

Данные сверены также с независимым entry-identity рядом предыдущего rational oracle. Флаг `entry_sha_is_original_decision_field=false` сохранён: поле создано wrapper, не выдано за поле simulator.

Sidecar рассчитан после завершения полного offline интервала. Ссылки на future closed trades служат **только атрибуцией**, не признаками модели и не источником решений/fitness. Использование этих ссылок в causal features недопустимо. Online open-position attribution и незавершённые lifecycles не подтверждены этим этапом.

## Результаты

Шесть прежних фиксированных combinations: три synthetic случая × два reference costs. Опубликованы **648 sidecar decisions**, **12 actual closed trade bindings**, **12 terminal records**. Все original accounting outputs и metrics **побайтно по canonical JSON идентичны** соответствующим outputs предыдущего probe. Новых стратегий/market samples не создавалось.

**5426 deterministic assertions** прошли: построчное сравнение с предыдущим oracle/trades, malformed/bool/future/missing/duplicate input rejection, payload SHA binding, повреждённые trades/decisions, изменённые source functions и preservation исходных outputs. Это число assertions на шести fixtures, не число независимых экспериментов или статистический testing budget.

По одному новому файлу выполнен py_compile; сначала проверен только pure validator, затем preflight, регрессии, exclusive publication и independent artifact checker. **1784** historical JSON и прежние контракты не изменены. Frozen/prospective interval не открыт.

## Provenance и воспроизведение

Receipt связывает predecessor SHA, inputs, wrapper/builder source hashes и outputs. Package init-file/Python executable hashes перепроверены, но full loaded dependency fingerprint не заявляется. Markdown не входит в hashed output bank.

Только `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8`, по одной команде:

- `tools/check_genetic_primus_strict_synthetic_wrapper_v1.py`.
- `tools/build_genetic_primus_strict_synthetic_wrapper_v1.py --verify-only` (шесть bounded artificial calls без публикации).
- Builder без флага — новый exclusive snapshot, не перезаписывает existing output.
- `tools/check_genetic_primus_strict_synthetic_wrapper_artifacts_v1.py` (read-only artifact verification и повтор тех же fixtures).

Output: `Reports/Exia/Genetic_Primus/strict_synthetic_wrapper_v1_20261006/{wrapper.json,environment.json,summary.json,completion_receipt.json}`.

## Что остаётся

Подтверждены synthetic input/payload binding и entry sidecar для **закрытых искусственных интервалов**. Не подтверждены full contract integration, online ledger, real model/scaler/data/fold/resample/cutoff payloads, training latency, realistic exchange execution и полный runtime graph. Старые dependency/terminal ambiguities не исправлены задним числом.

Следующий ограниченный этап: собрать fingerprint фактически загруженных Python modules/extensions и явно перечислить незакрытые native/transitive зависимости; свести инженерные gates в readiness record. Не объявлять эти gates открытыми по числу tests. Новое обучение/market evaluation остаётся отдельным решением; прежний отклонённый objective и sealed interval не переиспользовать автоматически.

Credentials/orders/paper/live/execution/promotion/training/financial evaluation/network flags остаются false. Model fits и market strategy evaluations — 0; прибыльность моделей не изменилась.
