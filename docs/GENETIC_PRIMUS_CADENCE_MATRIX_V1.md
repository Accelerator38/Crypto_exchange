# Cadence matrix v1

Реализованный контракт семи offline-конфигураций от9октября2026. Четыре независимых
clock fields: label_horizon_days, label_stride_days, decision_cadence_days,
retrain_cadence_days. В текущей матрице horizon3дня, linear6 с7параметрами,
uniform, L2=.01, threshold20bps, costs и capital account совпадают у всех A–G.

| ID | Horizon | Labels stride | Decision cadence | Retrain cadence |
|---|---:|---:|---:|---:|
| A | 3 | 3 | 3 | 3 |
| B | 3 | 1 | 3 | 3 |
| C | 3 | 1 | 1 | 3 |
| D | 3 | 1 | 1 | 1 |
| E | 3 | 1 | 1 | 2 |
| F | 3 | 1 | 1 | 4 |
| G | 3 | 1 | 1 | 5 |

Модель fit-time <= decision-time. На решениях между fits используется сохранённый
scaler и текущий causal feature vector; SHA модели, scaler, training/folds/cutoff,
feature-prefix и model age записаны в ledger. Общие fits переиспользуются между
кандидатами. Полный dataset содержит future outcomes, но fitter выбирает только
available<=fit−8bars: train_end=fit−1bar и purge7bars. Inner folds также purged;
для фиксированного linear L2 они не используются для tuning.

Label anchor сохраняет исходную сетку Aug1UTC; dense bank использует тот же anchor.
Daily3-day labels перекрываются на2дня. Kish ESS weights не объявляется числом
независимых outcomes. Calendar block-bootstrap4096draws: primary7дней,
sensitivity3/14, common indices для всех policies; seeds3039/7079. Report slices
фиксированы по датам и не зависят от retrain cadence; terminal partial отдельно.

Development1мая–25июля2026 (86дней) и раскрытый report1августа–30сентября (61день).
197unique linear fits /scalers:115development,82report. Предусмотрены44policy/scenario
replays с NoTrade, EMA daily, EMA3d и passive full8 controls. Исторические inputs,
code, folds, cutoffs и resample banks фиксируются до fits в registration.json.
Глобальный registry старых adaptive попыток неполон; DSR/PBO не заявляются.

Ledger переносит бухгалтерскую семантику frozen postclose v1. Капитал10000USDT,
8long/cash sleeves10%, неизменный target сохраняет units. Fees6bps/fill,
adverse drag2normal/8stress. Signed funding reported rates ×4h-close proxy,
включая terminal reports до terminal sale. Reference last completed4h-close,
assumed signal/fit ready100ms, fill350ms с неизменной за350ms ценой. Это допущения;
historical L2/consumer availability/rules/account fees/marks не квалифицированы.
DD —4h-close/operation marks. 10/20/30% — research post-ranking budgets, не stop-loss.
Нет margin/liquidation модельного фьючерсного счёта. Все9authority flagsfalse.

Для A обязательно точное совпадение21original models/scalers/targets и обоих
денежных replays с frozen run8октября. Другие clocks оцениваются на одном капиталe,
ledger и costs. Новые результаты на раскрытом report не объявляются независимой
валидацией, не выбираются новые thresholds или architecture по ним.

Команды выполняются последовательно через
`E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8 tools/run_genetic_primus_cadence_matrix_v1.py`:
prepare; checks; train --period report --start-day0 --days1; smoke;
train с неперекрывающимися chunks<=7дней; replay для одной period/policy/scenario;
audit; summarize; plot для каждой period; verify. Chunks сохраняют отдельный
sealed file каждого fit; complete cached fit не обучается повторно. JSON полностью
кодируется перед exclusive-create. Никакого pytest, collector, сети, background,
wall-clock ожиданий, API-ключей, runtime, ордеров, paper/live или promotion.
