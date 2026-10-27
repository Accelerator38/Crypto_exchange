# Релизный контракт и offline-исполнение — 2026-09-08

Реализован отдельный fail-closed контракт continuous sparse EMA overlay и
детерминированный **intent-level** replay исполнения. Это инженерный этап,
не обучение, не новое подтверждение прибыльности и не допуск к торгам.

## Результат проверки

| Группа прямых регрессий | Passed |
|---|---:|
| Контракт, hashes, scaler/cutoff, rounding, abstention | 5 |
| Latency, depth, partial IOC, stale/future data, idempotency | 6 |
| Long/short, reversal, reduce-only, fees, funding, reconciliation | 6 |
| Expiry, risk, stale marks, checkpoint, Decimal, bounded replay, общий риск | 7 |
| File bundle, model binding, fee/rule validity, projected loss, demo | 7 |
| **Всего** | **31** |

Проверки выполнены последовательно, отдельными короткими командами,
только `E:\Crypto_exchange\.venv\Scripts\python.exe`. Сначала выполнена
компиляция одного изменённого файла, затем прямая проверка одной чистой
функции хеширования; только после этого — группы синтетических регрессий.
Pytest, collector, training, торговые API и фоновые процессы не запускались.
Справочная проверка формулы funding выполнялась по публичной документации,
не по торговому API.

Сохранены пять receipts: `contract_tests.json`, `execution_tests.json`,
`accounting_tests.json`, `safety_tests.json`, `bundle_tests.json`. После всех
прогонов отдельно проверены их checksums и **30 совпадений SHA исходников**
(6 файлов × 5 receipts). Исправления до финальной фиксации: сравнение Decimal
в тесте приведено к числовому, а внутренняя точность деления отделена от
ограничения внешних decimal-строк. Финальные receipts относятся к одной версии.

## Файловый smoke

`synthetic_input.json` → `synthetic_result.json`: 8 событий, 3 фактических fills.
Сценарий: заявка на 2 units, исполнено 0.7 по двум уровням, остаток IOC отменён;
затем funding и reduce-only закрытие. Конечная позиция нулевая, pending orders
нет, `flat_and_reconciled=true`, `risk_latched=false`.

Все числа — **вымышленные fixtures**, не тарифы Bitget и не рыночная доходность:

- Начальный collateral: 1000 условных USDT.
- Gross realized PnL: 1.22; комиссии: 0.14144; funding: −0.0714.
- Конечные wallet/equity: **1001.00716**.
- Баланс: `1000 + 1.22 − 0.14144 − 0.0714 = 1001.00716`.

State SHA:
`d0e1e509eb94b43bea30cc45750daedc1e3049a56cf1aacd73fcae354b1f7486`.

Canonical contract SHA:
`ccb6c768a7379e228c7863ccd6e300489dc150ea64c181e2737a00c6ad40be83`.
Это canonical JSON hash, не raw-byte file hash; последний сохранён в receipts.

## Что изменилось относительно OHLCV-исследований

Нет гарантированного fill по next-open или принудительного terminal-close.
Появились: наблюдаемый стакан, latency, protective limit, lot/tick/minimums,
частичные IOC fills, фактические fees/funding, collateral/mark ledger,
срок действия rules/fees, идентичность каждого решения и restart checkpoint.
Неисполненные приказы, открытые позиции и пыль явно остаются в результате.

Старые результаты не пересчитывались. Проверены и сохранены прежние SHA
`measurement_lab_v2.py`, `accounting_v2.py`, сентябрьского prospective config.
Также отдельно совпали SHA результатов `origin_01.json` и `origin_02.json`:
`4e2b05988cdc8224eee370658eaf5e7a64db60ef18f260a0936da5e29d2ff6f8`
и `8303592a39609f6d9712017cb3dbf658ef45d75251f3d2463105653ba8577738`.
Ничего не коммитилось и не пушилось.

## Почему это пока не production-ready

Статус: **OFFLINE_CANDIDATE_UNQUALIFIED**, 14 qualification blockers:
4 незаданных денежных risk budget поля, 8 отсутствующих категорий evidence,
qualification биржевого adapter/margin/liquidation и отдельное разрешение
на deployment. Все safety-флаги остаются `false`.

Performance gates зафиксированы по смыслу, но **не вычислялись**.
Подставленные hashes/статусы evidence не дают автоматического разрешения.
Конкретные CI-протокол, размеры выборки и пороги execution drift ещё требуют
независимой калибровки и регистрации.

Replay проверяет заданные приказы, но пока не реализует общий portfolio
controller с доказанной эквивалентностью allocation старым cash sleeves.
Поэтому `strategy_backtest_comparable=false`: синтетический результат нельзя
добавлять в таблицу доходности GA против EMA/NoTrade.

Нет recorded book/mark/funding qualification, проверки полноты feed,
калиброванного impact/latency, margin/liquidation и live acknowledgement
reconciliation. Restart проверен только для локального event log.

Следующий шаг: общий offline portfolio adapter для GA/EMA/NoTrade и короткий
replay на предоставленных записях стакана/mark/funding с проверенными rules
и fee snapshots. После этого — зарегистрированный независимый audit.
Ни paper/shadow, ни реальные торги данным этапом не разрешены.

Подробный формат, ограничения и команды:
[GENETIC_PRIMUS_RELEASE_CANDIDATE_V1.md](../../../../docs/GENETIC_PRIMUS_RELEASE_CANDIDATE_V1.md).
