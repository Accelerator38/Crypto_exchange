# Genetic Primus: synthetic accounting probe v1

Дата: 2026-10-06. Только ограниченная проверка бухгалтерской функции на artificial inputs, не обучение и не market backtest.

## Что изменилось относительно stub этапа

На предыдущем этапе callee лишь записывал параметры. Здесь исполнена текущая закреплённая функция `simulate_cash_targets` на маленьких synthetic pandas DataFrames и сопоставлена с независимым oracle. Подтверждение относится к этим fixtures и текущему source SHA; старые результаты не становятся исторически подтверждёнными задним числом.

Оригинальные accounting/model/runtime файлы и frozen контракты не изменены. Новый harness: `src/exia/genetic_primus/synthetic_accounting_probe_v1.py`.

## Изоляция и последовательность

1. Статическое чтение accounting-файла, SHA и AST: разрешены прямые imports NumPy/pandas/dataclasses/typing; нет source-level file/dynamic operations или неожиданного top-level execution.
2. py_compile одного нового файла; затем чистая rational NoTrade проверка без исполнения accounting source.
3. Metadata versions: workspace Python 3.12.4, NumPy 2.4.2, pandas 3.0.0. Ничего не устанавливалось и не обновлялось.
4. Минимальная `_price(100)` проверка закреплённого source; затем 18-bar NoTrade sanity с equity 1 и 0 fills.
5. Маленькая фиксированная fixture matrix, регрессии, exclusive publication и отдельный artifact checker.

Accounting source SHA: `f9294a2e4b2220aa7552ebed7d2480e7e7647744fe5a62330888acba569af5bb`. Harness компилирует именно проверенные bytes в отдельный module namespace: без research package `__init__`, без использования accounting `.pyc`. Для импортов отключена запись bytecode. Запрошен single-thread limit для BLAS/OMP/MKL/NumExpr.

Audit hook блокирует socket events, внешние процессы, destructive operations и Python-level writes за пределами нового artifact directory. При dependency import был заблокирован запрос локального hostname (`socket.gethostname`): имя не получено/не выводилось. Первоначальный дополнительный assert требовал вообще пустого списка blocked events и потому не прошёл; сама price function прошла. После уточнения допускается только этот заблокированный metadata request, не соединения или процессы. Неожиданный другой blocked event останавливает probe.

Этот hook — защитный слой на Python audit events, не OS-level sandbox и не доказательство отсутствия любой native operation. Изолированность callee опирается также на pinned source и ограниченные artificial inputs. Никакие API keys, orders, collector, supervisor или runtime modules не использовались.

## Матрица

- Три прежних synthetic случая: carry, retarget, abstain; normal 16 и stress 24 bps. Шесть комбинаций, каждая 54 свечи/две artificial symbols/108 decisions.
- Один 18-bar single-symbol NoTrade sanity.
- Один negative control с forced-open reset на первой retrain boundary. Он проверяет существующее поведение аргумента, **не является зарегистрированной deployment policy**.

Итого восемь фиксированных artificial accounting cases. Это не восемь новых стратегий, не адаптивный поиск objective и не evidence доходности. Preflight, regression и artifact verification повторяют эти же детерминированные fixtures; это не независимые samples.

## Oracle и результат

Oracle использует stdlib `Fraction`: conservation of cash, signed quantity, actual-notional entry/exit fees, realized/unrealized PnL и NAV snapshots рассчитываются отдельно от accounting implementation. Он не вызывает accounting, NumPy/pandas или boundary transition helpers. Начальные sleeves равны; капитал непрерывен, без rebase/rebalance; unchanged target сохраняет quantity.

Сопоставлены trades, решения, open/after-fill/close/terminal events, portfolio returns и summary metrics. Drawdown проверен на тех же discrete snapshot phases, **не intrabar**. Oracle и tested function разделяют заявленную модель учёта, поэтому совпадение не доказывает экономическую правильность synthetic collateralized shorts или realistic exchange execution.

**17 356 сравнения**, максимальная absolute error **4.44e-16**, tolerance **1e-10**. Отдельно прошли **129 regression assertions**: стабильная quantity, entry-model attribution закрытых trades, compounding, zero-fill NoTrade, terminal cash=equity, corrupted result rejection и safety flags. Counts являются assertions на фиксированных fixtures, не statistical testing budget.

Negative control подтвердил exit по origin open, отличающемуся от previous close при том же boundary timestamp. Default carry не делает retrain-induced сделки. В конце интервала происходят реальные simulated terminal exits, их fees и realized PnL согласованы с oracle.

## Ledger gap

Actual decisions содержат `model_sha256` активной модели, но не entry-model SHA позиции. Actual closed trades содержат `entry_model_sha256`; на carry fixtures он совпадает с моделью первоначального входа, а не последнего origin.

Oracle сохраняет отдельный derived entry-model ряд, явно названный `oracle_decision_entry_sha_derived_not_original`. Он не сертифицируется как исходное поле simulator. Gate `per_decision_entry_sha_directly_available` остаётся false. Поэтому полная интеграция forward ledger contract ещё не подтверждена.

## Provenance и environment limits

Receipt связывает predecessor, synthetic output hashes, новый harness/builder и inputs. Версии NumPy/pandas, paths и on-disk package init SHA наблюдены; Python executable hash также проверяется до/после. Package init hashes не доказывают реально загруженный `.pyc`, compiled extensions, DLLs или полный transitive graph. `runtime_dependency_graph_verified=false` сохраняется.

Все 1784 frozen historical JSON перепроверены; старые контракты/результаты не переписаны. Новых model fits и market strategy evaluations — 0. Credentials/orders/paper/live/execution/promotion/training/financial evaluation/network flags остаются false. Synthetic function execution не является разрешением financial run.

## Воспроизведение

Только `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8`, команды по одной, short stdout:

- `tools/check_genetic_primus_synthetic_accounting_probe_v1.py`.
- `tools/build_genetic_primus_synthetic_accounting_probe_v1.py --verify-only` (выполняет только эти synthetic cases, не пишет артефакты).
- Builder без флага — новый exclusive snapshot; existing output не перезаписывает.
- `tools/check_genetic_primus_synthetic_accounting_artifacts_v1.py` (сверяет hashes и воспроизводит synthetic fixtures).

Артефакты: `Reports/Exia/Genetic_Primus/synthetic_accounting_probe_v1_20261006/{fixtures.json,environment.json,summary.json,completion_receipt.json}`. Markdown не входит в hashed output bank.

## Следующая задача

Закрыть forward integration gap в отдельном offline wrapper: строгая входная schema/model-payload binding и entry-model identity sidecar для каждого решения с явным происхождением. Проверять его по actual decisions/trades на этих же фиксированных fixtures, не на новом market interval. Отдельно расширить loaded-dependency fingerprint; не называть direct package hashes полным runtime bank. Market learning, новые objectives и sealed interval остаются за отдельными gates.
