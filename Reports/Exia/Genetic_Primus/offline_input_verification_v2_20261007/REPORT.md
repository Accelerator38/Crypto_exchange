# Genetic Primus: фактическая offline-проверка inputs origin_09

Дата: 7 октября 2026. Пользователь явно разрешил bounded verification старых
parquet, но не обучение, financial evaluation, сеть или новый sealed interval.

## Результат

**Структурная проверка прошла.** Последовательно прочитаны 11 файлов из
[старого manifest](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/prospective_3d_v1_202609/data_manifest_origin_09.json),
по одному на команду. Суммарный объём 3 180 806 bytes (~3,18 MB).
Каждый файл совпал с declared size/SHA до/после своего отдельного чтения.

| Измерение | Результат |
| --- | --- |
| Reference universe | 8: ADA, BNB, BTC, DOGE, ETH, LINK, SOL, XRP / USDT |
| Coverage интерпретированных timestamp keys | 2022-01-01 → 2026-09-28 UTC exclusive |
| Grid | 4 часа, int64 epoch-ms interpretation соответствует manifest |
| Timestamp count | 10 386, как в manifest |
| Total/unique timestamp-symbol rows | 83 088 / 83 088, как в manifest |
| Intra-file duplicates | 0 |
| Cross-file duplicates | 0 |
| Global missing keys | 0 по всем 8 символам |
| Per-file internal gaps | 0 |
| Projected columns | Только timestamp, symbol |
| Price values decoded/exported | Нет |
| Model fit / financial evaluation | 0 / 0 |

Нет утверждения о current fee, source authenticity, trading performance или
превосходстве над NoTrade.

## Что обнаружено в коде проверки

Первоначальный v1 schema guard требовал точного совпадения physical schema с
семью normalized columns manifest. File1 совпал по SHA/size, но содержал также
integer `source_rows`. Проверка остановилась **до key/price projection**.

Это ошибка строгости auditor, не повреждение файла или доказанный label leak.
V1 оставлен incomplete/diagnostic: preparation и оригинальные тексты обоих новых
скриптов сохранены до исправления в diagnostic_source_snapshot.json.
Законченные predecessor contracts/artifacts не изменялись.

В исправленной v2 physical и normalized schema разделены. Допускаются семь core
columns по именам и только additional integer **source_rows**; duplicate/missing
columns и другие extras, включая future-return labels, отклоняются.
Все 11 файлов имеют source_rows. Его values/semantics **не исследовались**,
не декодировались и не подавались в model features.
Это корректировка parser contract, не изменение trading objective после outcomes.

## Реальный объём чтения

Byte SHA вычислялся по целому compressed parquet: encoded price bytes входят
в file identity, но их значения не декодировались. Native decoder читал
BufferReader проверенных bytes, только timestamp/symbol batches.
Не использовались Arrow filename/remote filesystem, pandas metadata, price stats,
DataFrame или training runtime.

PyArrow 25.0.0 загружен с запретом NumPy/pandas/exia/pytest/WebSocket/HTTP clients.
`use_threads=false`, `pre_buffer=false`; socket/process audit events запрещены.
Во всех file audits совпала decoder identity (4 source/native file hashes).
Полная native dependency closure, loaded-memory identity и OS sandbox не
подтверждаются. Это ограниченный read-only engineering check, не sandbox proof.

Все команды завершались локально, без фонового процесса. Не было package install,
pytest, collector, supervisor, WebSocket, ордеров, paper/live или git push.

## Проверки и reproducibility

- Compile changed helper → прямой pure check → compile tool.
- V2 preparation: **32 pure regressions**, без market reads.
- File1..11: каждый отдельно, с оценкой предыдущего результата.
- Largest file: 79 488 rows; остальные суммарно 3 600 rows.
- Final aggregate computed из присутствия keys, не из цен/returns.
- **96 source/metadata/doc inputs** совпали с bank до/после этапов.
- `--verify` подтвердил receipt и **13 outputs**: preparation, 11 file audits,
  result. Это artifact verification, **не повторный reread parquet**.

Финальный результат:
[result.json](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/offline_input_verification_v2_20261007/result.json),
lineage:
[completion_receipt.json](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/offline_input_verification_v2_20261007/completion_receipt.json).
Per-file records содержат aggregate key counters, physical schema и компактные
presence bitmaps, не prices или model labels.

Source SHA проверялся до/после **каждого отдельного чтения**. Finalize читает
только audit JSONs и не объявляет источник заново проверенным на едином глобальном
моменте. Изменение source files после их audit потребует новой проверки.
Report создан после receipt и не входит в input bank.

## Что закрыто — и что нет

Закрыта фактическая проверка identity/schema/key-grid **этого старого historical
source**. Она подтверждает наличие rows, необходимых для proposed engineering
окна 22–25 сентября, включая поздние exit-open timestamp keys внутри coverage.
Это не подтверждение arrival этих observations в момент decisions/label availability.

Не закрыты:

- Finite/positive OHLCV, high/low consistency, volume и качество source_rows.
- Open/close timestamp meaning, arrival/late revisions и source authenticity.
- Venue/product соответствие research prices и Bitget perpetual execution.
- Actual causal labels/opportunities/scaler inputs/ESS, реальные fit/model times.
- Fees/slippage/funding/borrow, executable threshold и реалистичные fills.
- Power/multiplicity/audit banks, global objective registry и новый sealed interval.

Число 83 088 rows **не равно** opportunities или ESS и не заменяет прежние
sample gates64/40. Старый synthetic fixture7 не является размером реального
opportunity sample. Returns в этом этапе не вычислялись.

Все training/credentials/orders/execution/paper/live/promotion flags остаются
false. Разрешение на ограниченное input read не превращено в разрешение на fit.
Preregistration record и timing/data/cost contract не переписаны автоматически
под успешный grid audit.

## Следующий содержательный шаг

1. На тех же разрешённых historical inputs проверить OHLCV quality только
   агрегированными counters, по одному файлу и без цен в stdout. Отдельно проверить
   lineage исходных timestamps/source_rows, не объявляя guessed availability фактом.
2. Зафиксировать research availability/timing/cost assumptions с явной маркировкой
   допущений. Venue/funding/fee qualification для release не смешивать с sensitivity.
3. После actual input/label lineage и отдельной авторизации training — один
   ограниченный инженерный refit по frozen двум вариантам, без выбора winner
   по smoke. Только затем заранее зарегистрированное статистическое сравнение.

Дополнительный mock-stage не устранит эти неизвестные. Новый prospective/sealed
interval не открывается, реальные деньги и paper/shadow/canary не разрешены.
