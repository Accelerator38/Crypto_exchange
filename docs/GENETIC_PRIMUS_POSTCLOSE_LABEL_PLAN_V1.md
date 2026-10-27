# Post-close label plan v1

Назначение — согласовать временную семантику следующей модели и описать
недостающие входы. Это price-free план clocks, не построитель финансовых
labels, не обучение, не новый objective/ranking и не разрешение торгов.

## Разница со старыми планами

`causal_market_label_plan_v1` и `compact_forecast_data_v1` использовали next
4h open, даже если сигнал был готов непосредственно после close. Новый
план соответствует `compact_ioc_execution_v1`: closed feature bar становится
доступен, compute latency, первый свежий post-signal reference snapshot,
order latency, отдельный первый свежий post-arrival snapshot. Reference
snapshot не может исполнить созданную по нему заявку. Задержанный snapshot
с exchange time до arrival не подходит. TTL исключает правую границу.

В новом плане передаются exchange, receive и observation-available clocks.
Последний — реальная доступность observation обработчику, а не download
time ретроархива; в будущем adapter передаёт его как event `ts` ядру.
Если историческая доступность не доказана, её нельзя выдумывать. Ни depth,
ни возможность полного fill clocks сами по себе не доказывают.

Deployment/retrain cadence = 3 календарных дня. Выход label-сценария
планируется на следующем scheduled origin, а не на entry fill + 3 дня:
ожидание quote не продлевает горизонт. Закрытие полного validation ledger
на 1 октября — отдельное terminal решение, не новый трёхдневный label.
Последний неполный deployment interval не удлиняется произвольно.

## Созревание и purge

Training asof = origin − 1 бар; cutoff = training asof − 7 баров.
Проверка выполняется по max доступности ВСЕХ зависимостей исхода, включая
exit/fill, costs, funding и marks. Три дня по календарю сами по себе не
доказывают, что label созрел. Поздно пришедший источник способен исключить
label даже тогда, когда само ценовое движение закончилось до cutoff.

Clock pair сохраняет source/snapshot/window/target hashes и доступность
конечного observation. `dependency_maturity` проверяет только время явно
переданных зависимостей, а не полноту economics. Поэтому даже успешная
проверка всегда оставляет `training_ready=false` и `execution_qualified=false`.
Неполный prefix имеет PENDING, не нулевой target или автоматический NoTrade.
Нет отбора только прибыльных исходов: без исходов здесь вообще нет fitness.

Old labels, model/scaler hashes и +13.07% прежнего replay не получают новый
статус. Нужны отдельные target/label/data hashes; затем новый scaler каждого
origin и зрелые inputs. Causal training history для нового clock ещё не
зарегистрирована. Старый период validation уже исследован, не prospective.

## Preflight данных

Проверяется наличие зарегистрированного bundle для full8 и 6 компонентов:
L2, marks, funding, historical rules, fees, closed features. Требуются
receive/exchange/availability clocks, начальный snapshot и непрерывные deltas,
sequence/gap/reconnect audit, actual depth, signed funding и settlement marks,
валидные исторические lot/tick/minimum/fee intervals. Self-declared quality
и hash в JSON не заменяют проверки файлов и независимого quality receipt.

Период replay: 2026-08-01 00:00 UTC — 2026-10-01 00:00 UTC exclusive.
Для terminal выхода требуется tail до `end + compute + reference wait + TTL`
(в инженерном контракте 7.1 секунды). Funding settlement ровно на правой
границе учитывается до terminal fills; это нельзя потерять при обрезке rates
по exclusive end. Дополнительный rate может уже находиться в старом raw
архиве, но normalized запрошенный период сам по себе этого не доказывает.

Для closed features минимум 49 четырёхчасовых баров, но seed EMA48 и другие
stateful признаки требуют полного causal prefix, не произвольного reseed
на локальном buffer. Для переобучения нужны данные всего зарегистрированного
training prefix, не только август–сентябрь. Training scope пока не утверждён.

Локальный inventory ограничен известными источниками и не распаковывает
большие gzip. Имена файлов и stat — подсказки, не доказанная coverage.
BTC L2/trades/derivative за декабрь 2024 подходят для будущего bounded parser
smoke, но не заменяют requested full8 execution data за август–сентябрь 2026.
Book ticker без depth также не является L2. OHLCV не превращается в книгу.

Инженерные timings/participation пока не откалиброваны. Прежний kernel —
полностью обеспеченный long/cash diagnostic, не futures margin/liquidation
simulator. Даже полный структурный bundle не выдаёт production authority.

Все проверки локальные и ограниченные; никакой сети, collector, ожиданий,
background, training или market-profit replay. Все 9 safety flags false.
