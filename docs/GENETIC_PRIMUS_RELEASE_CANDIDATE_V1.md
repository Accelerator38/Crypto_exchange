# Release candidate v1: контракт и offline-исполнение

Дата: 2026-09-08. Это **не релиз в торговый runtime**. Контракт допускает
только локальную диагностику; `release_ready=false`, все семь торговых и
сетевых разрешений остаются `false`. Старые исследовательские артефакты,
сентябрьский prospective и V8 Clean collector не изменяются.

## Зафиксированная гипотеза

`configs/genetic_primus_release_candidate_v1.json` фиксирует continuous sparse
overlay над EMA: четыре признака, genome `(0,0,1,1,-1)`, строгий положительный
logit, abstention при отсутствующих признаках. Не используется отклонённая
entry-only абляция. Новая цель исполнения — Bitget USDT-FUTURES, линейный
perpetual с расчётами в USDT, one-way. Это отдельная ветка относительно
первоначального Binance COIN-M liquidation feed V8 Clean.

4h-признаки считаются по завершённым барам. Каждый origin длится 3 дня;
cadence также 3 дня. Expanding train начинается 2022-01-01 и заканчивается
строго перед `origin_start − 7 × 4h`; scaler median/IQR с clip 8 переобучается
на этом train, genome не меняется. Сам replay **не обучает** scaler и не
пересчитывает исторические признаки: получает подписанный хешем manifest.

`bind_model` проверяет границы и формирует новый model SHA из контракта,
origin/cutoff, scaler, хешей train/data/folds, selection/audit resample banks
и реализации. Одинаковые SHA selection и audit banks запрещены. Отличающиеся
SHA сами по себе не доказывают статистическую независимость — это задача
аудита. Нельзя присвоить новой модели старый research SHA.

Отдельный execution binding связывает настройки, правила инструментов и
все модели; он записывается в ledger. На каждом решении, включая abstention,
хранятся model SHA, execution binding, cutoff и хеш входных признаков,
proposal/target, train dataset SHA и execution dataset SHA.

## Что реализует execution replay

- Decimal-арифметика с локальной фиксированной точностью; float запрещён во
  внешних денежных значениях. Нет случайности, wall-clock, sleep или asyncio.
- Quantity округляется вниз к lot step. Защитная buy-price округляется вниз,
  sell-price вверх к tick; проверяются minimum quantity/notional. Неизвестные
  исключения для закрытия «пыли» не выдумываются: остаток остаётся видимым.
- Только marketable limit IOC, taker-only. После заданной задержки берётся
  первый свежий наблюдаемый snapshot, сформированный не раньше arrival.
  Цена не пересекает защитный limit. Учитываются доступные уровни и явный
  participation cap; остаток IOC отменяется. Maker-очередь не моделируется.
- Один pending order на инструмент. Новый приказ требует разрешения
  неопределённого состояния предыдущего. Одинаковый client ID с прежним
  payload идемпотентен, с другим payload — ошибка. Повтор snapshot не
  восстанавливает ликвидность. Время стакана не может регрессировать.
- Есть TTL, отмена, отказы, истечение срока действия fee/rule snapshot,
  ограничения свежести mark/book и модельных origins. Некорректный event
  отклоняется транзакционно, не оставляя частично изменённый ledger.
- Линейный futures ledger: collateral не списывается как покупка spot;
  реализованный PnL появляется при закрытии, unrealized PnL — по mark.
  Комиссия начисляется на фактически исполненный notional. У частичного
  разворота сохраняются реальная величина и средняя цена остатка позиции.
- Funding берётся только из явных settlement events:
  `payment = −signed_quantity × settlement_mark × rate`.
  На одинаковой временной метке funding обрабатывается **до** сделок.
  Поздняя запись требует нового хронологического replay, а не списания по
  текущей позиции задним числом. Повтор settlement не списывает деньги дважды.
- Проверяются общий gross/equity ≤ 1, order notional, абсолютный loss budget
  и drawdown; projected loss также проверяется до нового риска. После
  нарушения latch блокирует новый риск, но разрешает reduce-only закрытия.
  Рыночный гэп всё равно может превысить loss budget: это стоп-порог, не
  гарантия максимального фактического убытка.
- Состояние восстанавливается из bounded event-sourced checkpoint. Проверяются
  checksum, параметры/rules/model binding и точное совпадение state SHA после
  повторения событий. Никаких фоновых процессов или автоматических рестартов.

## Граница реализованной функциональности

Это **intent-level execution diagnostic**, не полный portfolio backtest.
`overlay_target` реализует scaler + frozen gate; `transition_order` — чистый
адаптер sizing для явно переданного sleeve equity: hold при прежнем
направлении, reduce-only при нуле, net-delta IOC при развороте. Он не является
портфельным контроллером и не создаёт приказы автоматически.

Bundle содержит заданные намерения; replay проверяет их связь с решениями,
направление overlay и риск, но не доказывает эквивалентность распределения
капитала старым cash sleeves. Поэтому `strategy_backtest_comparable=false`.
До сравнения доходностей нужен отдельный проверенный общий evaluator,
подающий GA, EMA и NoTrade одинаковый поток событий и правила капитала.

Не реализованы: live API/ack reconciliation, очередь maker, калиброванное
market impact, margin mode/maintenance tiers/liquidation, автоматические
повтор/долив/закрытие в конце периода, контроль полноты funding/book feed.
Expiry/отмена в replay — детерминированная модель, не доказательство того,
как и когда реальная биржа подтвердила отмену. Restart-тест относится к
локальному event log, не к exactly-once доставке приказов на биржу.

Все fee/lot/tick/latency/participation/risk числа в тестах **синтетические**.
Текущие значения биржи не хардкодятся. Формула стоимости futures и funding
согласуется с [пояснением Bitget](https://www.bitget.com/asia/support/articles/12560603828198);
тариф, точность расчётов и условия конкретного счёта ещё требуют проверки.
Margin/liquidation намеренно не заменены придуманными «реалистичными» fills.

## Формат и безопасный локальный запуск

Bundle schema: `exia.genetic_primus.execution_bundle/1`.
Поля: `settings`, `rules`, `models`, `events`. Денежные значения — строки,
время — целые UTC milliseconds, все events имеют уникальные `id`.

Event kinds: `funding`, `mark`, `decision`, `intent`, `book`, `cancel`, `clock`.
Это также порядок событий на одинаковой временной метке. Все входные события
уже должны быть упорядочены: runner не сортирует их с доступом к будущему.
`book/mark.exchange_ms <= ts`; quote должен быть не старше разрешённого age.
Для `decision.feature_cutoff_ms` требуется уже закрытая 4h-граница и возраст
менее одного бара. Feature provenance всё равно нужно проверять отдельно.

`rules[symbol]`: `price_tick`, `quantity_step`, `minimum_quantity`,
`minimum_notional`, `source_sha256`, `valid_from_ms`, `valid_until_exclusive_ms`.
Полный пример settings и events создаётся только по явному флагу `--demo`.
`settings.dataset_sha256` равен canonical SHA объекта `{rules, events}`.
Model provenance содержит SHA реализации из `implementation_identity()`.
Runner заново восстанавливает model manifest и проверяет его точное равенство.

По умолчанию checker только статически проверяет контракт и хеши:

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/check_genetic_primus_release_candidate_v1.py
```

Прямые синтетические тесты запускаются отдельными короткими командами с
`--group contract`, `execution`, `accounting`, `safety` или `bundle`.
Никакого pytest. Для создания **нового** demo input:

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/check_genetic_primus_release_candidate_v1.py --group bundle --demo Reports/Exia/Genetic_Primus/release_execution_v1_20260908/example_new.json
```

Затем отдельной командой, оценив предыдущий результат:

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/replay_genetic_primus_release_candidate_v1.py --input Reports/Exia/Genetic_Primus/release_execution_v1_20260908/example_new.json --output Reports/Exia/Genetic_Primus/release_execution_v1_20260908/example_result_new.json
```

Input ограничен 2 MiB, replay — максимум 10 000 events и 200 levels на сторону
(fixture: 128/8). Это безопасный короткий сегмент, не высокочастотный движок.
Результаты и receipts создаются эксклюзивно: существующие файлы не заменяются.
При изменении кода старый input сохраняют, новый binding создают отдельно.

## Допуск и следующие действия

Существующий контракт структурно проверяем, но остаётся
`OFFLINE_CANDIDATE_UNQUALIFIED`. Четыре денежных risk budget поля не заданы,
восемь категорий qualification evidence отсутствуют. Ещё два обязательных
блокера — qualification биржевого adapter/margin/liquidation и отдельное
явное разрешение на deployment. Заполнение хешей не снимает эти блокировки.

Контракт фиксирует требования: положительный paired incremental LCB против
EMA и NoTrade при normal/stress costs, ≥70% положительных outer slices,
независимый audit bank, абсолютный drawdown budget (сравнение с incumbent
только если тот торгует), раздельные minima opportunities/ESS/trades,
зарегистрированные turnover/concentration/drift, глобальный objectives
registry и DSR/PBO. **Метрики этих gates здесь не вычислялись**:
`performance_gate_evaluation=NOT_RUN`.

Конкретный CI-протокол, достаточный размер независимой выборки и новые
execution-calibrated thresholds должны быть запечатаны до следующего audit,
а не выбраны по его результату. Синтетическая PnL не является performance evidence.

Следующий технический шаг — общий offline portfolio adapter и recorded-data
qualification на предоставленных book/mark/funding записях с актуальными
instrument/fee snapshots. После этого — отдельный независимый statistical
audit. Shadow/fault tests и canary потребуют другого этапа и явного разрешения;
этот код не умеет включать paper/live или отправлять ордера.
