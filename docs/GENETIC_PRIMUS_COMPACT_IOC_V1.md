# Compact IOC v1 — только offline-инженерия

Это отдельное чистое ядро исполнения, не торговый runtime и не новая версия
прибыльности. Старые результаты и pinned исходники не изменяются. Генератор
проверок запускается явно через `.venv\Scripts\python.exe`, без pytest,
collector, wall-clock ожиданий, сети и фоновых процессов.

## Контракт времени и исполнения

После доступности закрытого бара и готовности сигнала берётся первый свежий
quote с exchange timestamp не раньше сигнала. Он задаёт защищённый limit;
затем отсчитывается зарегистрированная задержка доставки заявки. Исполнение
разрешено только на следующем свежем snapshot, у которого exchange и receive
timestamps не раньше arrival. Reference snapshot не может исполнить эту
заявку даже при нулевой задержке. IOC делает одну попытку; неисполненный
остаток отменяется, никогда не переносится в последующие snapshots.

Receive clock не регрессирует; exchange clock, freshness, отсутствие crossed
book и grid уровней проверяются. Повторный snapshot ID не создаёт новую
ликвидность, а изменённое содержимое при том же ID отвергается. Budget
participation общий для всех заявок на одном snapshot/side/price. Это
консервативная модель, не точная очередь venue и не доказательство fills.

При совпадении receive timestamp порядок: funding → mark → intent → book →
cancel → clock. Самые поздние deadlines исключаются: `ts >= deadline`
означает expiry. Все clocks передаются как входные события; sleep отсутствует.

Примерные инженерные параметры: compute latency 100 ms, order latency 250 ms,
TTL 2 s, quote age 500 ms, mark age 2 s, participation 10%, protection 30 bps.
Они НЕ откалиброваны по исторической задержке или фактической ликвидности.
Синтетические цены/объёмы и rules/fees прямо помечены как синтетические.

## Размер, риск, деньги

Decimal-quantity округляется вниз; protected buy price вниз, sell вверх.
Проверяются minimum quantity/notional, maximum order notional, резерв cash
с комиссией, gross exposure, свежие marks и validity interval rules/fees.
Evidence должно быть действительно и по reference/exchange clock, и по
processing clock. Sell только reduce-only; общий reserve не допускает oversell.
Издержки считаются по фактическому синтетическому fill, не по желаемому объёму.

Учёт — полностью обеспеченная long/cash диагностика: при покупке из cash
списывается весь notional, borrowing и отрицательный cash запрещены. Это НЕ
реальный futures-margin, liquidation или venue-balance simulator. Пример
drawdown gate 20% блокирует новые buys после нарушения, но разрешает выход;
он не гарантирует максимальную просадку и не согласует риск реальных денег.

Funding со знаком учитывается на позиции перед settlement (вход ровно в
settlement не платит; выход ровно в settlement платит). Требуются наблюдаемый
settlement mark, источник и `known_at_ms <= settlement_ms == receive ts`.
Contemporaneous funding mark обновляет NAV, чтобы старый mark не создавал
искусственную просадку при текущем funding. Невмещающийся в free cash reserve
покупок отменяется; borrowing для funding не разрешён. Поздно узнанный funding
отвергается: отдельный adapter/reconciliation для таких сообщений ещё нужен.

## Воспроизводимость и ограничения

Каждый intent и fill привязан к model/scaler/feature/training hashes. Модель
должна иметь causal origin, отдельный scaler, train_end до origin и cutoff
после purge 7 баров (origin минус 8 четырёхчасовых баров). Bridge читает
существующий frozen compact origin без обучения, inference или изменения.

Checkpoint содержит ограниченный accepted event prefix и hashes контрактов,
данных и реализации. Restore детерминированно повторяет prefix и проверяет
точное совпадение результата. Изменение sources/contract или повреждение
checkpoint отклоняется. Это не автоматический restart рабочего бота.

Жёсткие пределы: в конфиге 256 событий, 32 заявки, 20 уровней на сторону;
ядро не разрешает расширять пределы выше 512/64/50. До 8 symbols и 32 bindings.
Повреждённое событие атомарно отклоняется без изменения state или event prefix.
События/ledger ограничены, нет reconnect/cancellation loops или fsync.

## Почему нельзя сразу пересчитать август–сентябрь

Старый replay исполнял next-4h-open — примерно через четыре часа после close,
а не сразу после него. Его трёхдневные labels и прежние model hashes нельзя
переименовать в post-close модель. `new_clock_labels_qualified=false` обязателен.
Нужны новый target-clock контракт, causal labels и отдельное переобучение.

Для финансового replay необходимы исторические receive/exchange L2 clocks,
depth и sequence/snapshot identity, rules/fees с доказанным периодом действия,
funding availability и наблюдаемые settlement marks, по всему интервалу.
Текущие публичные rules не подставляются в прошлое, а OHLCV не превращаются
в искусственный L2. Gap/reconnect/snapshot reconstruction ещё требуют adapter
и fault-injection проверки на настоящих данных.

Проверка записывает отдельный immutable evidence root: 37 явных unittest,
8 синтетических сценариев, cash reconciliation, source/event/model hashes,
checkpoint/resume equality и график NAV/cash. График показывает издержки
круговой сделки при постоянном mark, НЕ alpha/доходность стратегии. Никакой
training, market evaluation, bootstrap/ranking, promotion или trading authority.

Все девять safety flags остаются false: credentials, orders, execution,
paper, live, promotion, training, financial evaluation и network. Прежние
прибыльности/LCB/gates не пересчитываются и не получают нового статуса допуска.
