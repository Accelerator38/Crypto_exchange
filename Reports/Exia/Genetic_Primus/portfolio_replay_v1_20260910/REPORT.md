# Общий портфельный replay — 2026-09-10

Следующий инженерный шаг выполнен: поверх замороженного execution v1 добавлен
единый offline portfolio controller для **GA / EMA / NoTrade**. Он сам
создаёт приказы из общей сетки признаков и решений. Раньше сравнивались
заданные извне intents; теперь одинаковы также правила формирования позиции
и распределения капитала.

Это не обучение, не рыночная валидация и не разрешение на реальные деньги.

## Зафиксированные правила

Восемь равных начальных долей капитала, без перераспределения idle cash и
без передачи прибыли между инструментами. Same-side решение сохраняет
реальную quantity; частичный вход не доливается. Нулевой target закрывает
reduce-only; противоположный — net-delta IOC с реальным частичным остатком.

Защитная цена и размер определяются по уже известному mark. Позиции и
денежный баланс сохраняются между трёхдневными origins; scaler/model identity
меняется по manifest с purge 7 баров. Завершение файла не создаёт бесплатного
закрытия. Каждое решение, включая NoTrade и abstention, имеет model SHA и
execution binding. EMA/NoTrade получили собственные явно обозначенные policy
identities, без выдачи их за GA.

## Проверки

Сначала — `py_compile` одного нового файла, затем прямая проверка чистой
функции EMA proposal. После этого последовательно выполнены короткие группы:

| Группа | Passed |
|---|---:|
| Policy, sizing, manifest, role identity, Decimal | 5 |
| Общие условия сравнения и отсутствие перераспределения | 5 |
| Latency/protection, partial fills, abstention, origins, reversal | 5 |
| Funding, cash reconciliation, собственный бюджет, risk exit | 4 |
| Full8/cutoff, hashes, stale data, restart, causality, bounds | 9 |
| **Всего** | **28** |

Среди регрессий: одинаковые targets GA/EMA дают одинаковую экономику;
перестановка порядка агентов не меняет результат; NoTrade сохраняет деньги;
будущая котировка не влияет на предыдущее sizing; прибыль одного инструмента
не финансирует другой; malformed features не превращают сравнение в молчаливый
NoTrade. Дополнительно сам контроллер явно запрещает использовать старый
kernel checkpoint вместо восстановления всего portfolio controller.

Сохранены пять `*_tests.json` receipts. Проверены их canonical checksums и
**40 совпадений SHA файлов** (8 × 5). Отдельно подтверждены **30 прежних SHA**
по receipts release/execution v1 от 8 сентября. Старые 31 тест не запускались
повторно. Все новые запуски — только явным `.venv\Scripts\python.exe`, без
pytest, collector, сети, обучения и фоновых процессов. Торговый runtime,
ключи, старые результаты и сентябрьский prospective не изменялись.

## Файловый smoke — только синтетические числа

`synthetic_input.json` → `synthetic_result.json`: **56 общих событий**, две
полные full8-границы решений, по 16 decision records у каждого из трёх агентов.
На входе: сильный одинаковый сигнал; исполнение 0.4 из запрошенных 1.2 units
на каждом инструменте; затем funding и flat-сигнал с фактическим закрытием.

| Агент | Начальный капитал | Итоговый equity | Fees | Funding | Fills |
|---|---:|---:|---:|---:|---:|
| GA | 1000 | 1001.59360 | 0.64320 | −0.3232 | 16 |
| EMA | 1000 | 1001.59360 | 0.64320 | −0.3232 | 16 |
| NoTrade | 1000 | 1000 | 0 | 0 | 0 |

У всех трёх `flat_and_reconciled=true`. Для GA/EMA:
`1000 + 2.56 gross realized PnL − 0.64320 fees − 0.3232 funding = 1001.59360`.
Равенство GA/EMA здесь специально задано одинаковыми targets — это тест
бухгалтерии и симметрии сравнения. **Это не рыночная прибыль и не доказательство
превосходства GA над EMA или NoTrade.**

Canonical portfolio contract SHA:
`89960011aaf432f2615797cfd6873fd931a36924250387df86fe90f364e42ed7`.

Canonical comparison body SHA:
`dc6b52f9de385d907d133793dba189755ee57d166be956afc8eb04f3ab5e416d`.

## Что остаётся открытым

Агенты теперь сравнимы внутри одного нового portfolio replay:
`within_replay_candidates_comparable=true`. Со старыми OHLCV-ретротестами
числа напрямую не смешиваются: `historical_research_results_comparable=false`.
CI/ESS/DSR/PBO/promotion tests не выполнялись. Инженерные fixtures не являются
новыми финансовыми наблюдениями или запуском genetic search.

Есть локальный manifest funding для восьми инструментов с общим покрытием
2026-04-28 16:00 UTC — 2026-07-27 08:00 UTC и разрешением
`recent_diagnostic_screen_only`. Сами CSV не проверялись/не импортировались.
Поиск по именам `*book*`, `*depth*`, `*mark*` в `Retrodate` не дал файлов;
это ограниченная инвентаризация, не утверждение об отсутствии данных во всём
workspace. Проверенный согласованный комплект book/mark/funding + rules/fees
для qualification пока не установлен. OHLCV не подменялся выдуманным стаканом.

Следующий шаг — read-only importer/validator такого записанного комплекта и
проверка causal features/scaler provenance, затем один короткий общий replay.
Не хватает также калибровки исполнения, margin/liquidation, live reconciliation
и независимого statistical audit. Аварийное intrabar закрытие не реализовано;
срабатывание risk budget не гарантирует ограничение фактического убытка гэпа.

`qualification_eligible=false`, `trading_authorized=false`; все семь safety
permissions остаются false. Никакие настройки paper/shadow/live не включены.

Подробности и команды:
[GENETIC_PRIMUS_PORTFOLIO_REPLAY_V1.md](../../../../docs/GENETIC_PRIMUS_PORTFOLIO_REPLAY_V1.md).
