# Genetic Primus: timing / execution / data контракт

Дата: 7 октября 2026. Этап завершён как **проект исследовательского контракта и
его проверка**, не как обучение модели или подтверждение торговой доходности.

## Результат

Созданы отдельный JSON-контракт, чистые функции проверки и bounded checker.
Замороженные release/prospective контракты, прежние модели и торговый runtime
не изменялись. Старые результаты не переаттестованы под новую семантику.

Проверки выполнялись последовательно только workspace Python с `-X utf8`:

- `py_compile` одного изменённого модуля — успешно;
- прямые проверки чистых timing и availability/purge функций — успешно;
- checker: **112 assertions**, **68 source/metadata/doc inputs неизменны**;
- `--verify`: повторены 112 assertions, совпали receipt и 2 JSON-результата.

`assessment.json`, `summary.json` и `completion_receipt.json` хранят состояния
и SHA привязки. Исходный proposal связан по semantic SHA, а файлы также имеют
byte SHA: эти два вида hashes не смешиваются.
Report создан после receipt и не входит в его input bank.

## Что исправляет новый проект

| Вопрос | Зафиксированное правило |
| --- | --- |
| Когда возможен планируемый OHLCV-вход | Первый open строго после decision + зарегистрированная задержка |
| Горизонт | 3 дня между отложенными entry и exit; обе границы сдвигаются вместе |
| Выход label | Open следующего запланированного rebalance, не legacy last-close |
| Когда label доступен | Максимум arrival времени наблюдений обоих open |
| Purge | Availability label <= training-asof − 7 четырёхчасовых баров |
| Scaler lineage | SHA opportunity feature rows, а не SHA всех label rows |
| Model lineage | SHA модели решения совпадает с заявленным model payload SHA |
| Timing | asof <= fit start <= fit complete <= model available <= decision <= submission; fit укладывается в бюджет |
| Bootstrap lineage | Закреплены SHA folds/search bank/audit bank; разные SHA не доказывают независимость seeds |
| Train identity | Только training-prefix, без полного архива с будущими deploy-ценами |

Это **OHLCV sensitivity**, не реалистичный exchange fill adapter. На точном
совпадении ready с open правило выбирает следующий open. Это намеренно
консервативная дискретизация. Даже explicit delay=0 не разрешает тот же open.
Scope ограничен decision в первом баре origin и delay < 4 часов.

Проверены точные границы, сдвиг на следующий open, поздний arrival на 1 ms,
некаузальные timestamps, неверные types, NaN/Infinity, duplicate JSON keys,
подмены hashes/границ и попытки включить authority.
Числа и hashes в положительном registration/ledger control — artificial;
они не опубликованы как зарегистрированный рынок или реальный model ledger.

## Почему старые доказательства не закрывают эту задачу

В `linear_latency_sensitivity_v1` задержка одного бара названа conservative
OHLCV sensitivity; `no_model_refits=true`. Этот эксперимент не измерял
latency биржи и не проверял полное переобучение каждые 3 дня.

Существующий release-контракт требует IOC taker, fresh snapshot после latency,
mark и явного funding, запрещая OHLCV fill fallback. Новый проект не отменяет
эти требования и не делает bar open подтверждённой исполнимой ценой.

Отложенный open-to-open label — планируемый валовой outcome для обучения,
не PnL реальной позиции. Неизменный target должен переноситься без refill,
fee/funding/borrow и единый terminal close учитываются отдельно в continuous
cash/quantity accounting. Проверка реальных цен, labels и costs здесь не выполнена.

## Оставшиеся блокеры

В новом проекте **21 обязательное поле остаётся null**:

- timing — 3: compute budget, decision-to-ready delay, SHA основания выбора;
- data — 8: manifest, timestamp spec, availability evidence, revision policy,
  universe, feature definition, training-prefix manifest, snapshot cutoff;
- costs — 10: adapter, normal/stress fee/slippage per side, calibration evidence,
  funding evidence, borrow basis, terminal policy, instrument specification.

Значения 16/24/40 bps из старой sensitivity автоматически не перенесены:
round-trip и per-side параметры нельзя отождествлять.
Неприменимость funding/borrow также требует явного основания, не default 0.

Из предыдущего pre-fit этапа сохраняются sample блокеры: **7 opportunities
вместо 64**, Kish ESS около **7 вместо 40**. Новые timestamps controls не
увеличивают размер market sample. Все outcomes, включая убытки, должны
сохраняться в весах. Независимость bank и достоверность claimed ledger
не подтверждаются одними hashes.

Все safety flags — literal false; registration complete, launch ready и
real-money ready — false. Fit=0, market evaluation=0, market-data reads=0.
Не запускались pytest, collector, WebSocket, supervisor, сеть или фоновые процессы.

## Дальнейшая последовательность

1. По существующим локальным metadata и спецификациям подготовить одну таблицу
   покрытия этих 21 полей: источник, значение, единицы, SHA, что отсутствует.
   Не создавать новые toy datasets ради видимости завершения; не угадывать costs.
2. Зарегистрировать обоснованные параметры отдельной immutable версией и
   проверить actual source/availability/revision provenance. Исторические
   gross labels не выдавать за новые execution-aligned labels.
3. После достаточных opportunities/ESS и отдельной авторизации research fit —
   один ограниченный causal origin с реальным scaler + coefficient refit,
   измеренными timing и model SHA для каждого решения. Затем два заранее
   определённых варианта uniform / half-life90d при одинаковом бюджете.
4. Оценивать calendar net after stress относительно NoTrade, с общей paired
   bootstrap bank и независимым audit; не использовать промежуточные peek
   для подбора модели. Новые sealed даты и power/multiplicity budget ещё нужны.

Цель пользователя — неубыточный конечный net после всех costs при допустимых
временных просадках. Этот этап не показывает достижения цели и не гарантирует
отсутствия убытков. Paper/shadow и реальный canary остаются закрыты.
