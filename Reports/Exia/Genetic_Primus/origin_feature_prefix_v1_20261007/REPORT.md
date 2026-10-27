# V8 Clean — origin, признаки и отдельный scaler-prefix

Дата: 2026-10-07. Этап завершён на синтетических данных. Это не обучение,
не валидация доходности и не повторный анализ historical/sealed рынка.

## Реализация

Новый отдельный адаптер `src/exia/genetic_primus/origin_feature_prefix_v1.py`
связывает ранее проверенную OHLCV availability со следующими компонентами:

- Origin context: `training_asof=origin−1bar`, решение внутри первого origin-бара,
  явные ingestion/execution delay, shifted entry/exit с сохранением 3 дней.
  Entry и exit deployment должны лежать внутри полуоткрытого покрытия snapshot.
- `return_12` и `efficiency_12`: ровно 13 цен закрытия в зависимостях. EMA-предложение:
  весь prefix от закреплённого history start, минимум 48 баров. Сужение линейной
  модели до двух признаков не сокращает зависимости EMA до 13 баров.
- Prefix с полным набором символов: никаких пропусков, дубликатов, смены revision
  или публикаций после соответствующего asof. Полный snapshot и каждый реально
  используемый prefix закреплены отдельно по SHA.
- Training feature rows отбираются только после доступности строк entry/exit
  и семибарного purge. Поздняя публикация label исключает candidate для всей
  синтетической панели. Deployment-признаки не содержат label columns.
- На каждом origin заново формируется **план входных строк** scaler: число строк,
  SHA идентификаторов и содержимого feature rows, training_asof. Статистики,
  scaler payload и модельные коэффициенты не рассчитываются и не переиспользуются.

Значения признаков в артефакты не выгружаются — фиксируются их hashes. Цен
entry/exit, labels, доходностей, costs и PnL этот адаптер не рассчитывает.
Candidate calendar задаётся явно; частота синтетических training-примеров не
регистрирует новый торговый график или cadence переобучения.

## Проверки и результаты

Успешны компиляция одного нового чистого файла и прямая проверка origin context.
`--run` и отдельный `--verify`: **739 assertions**, **159 неизменённых входных
source/metadata identities**, receipt/result совпадают.

Проверено:

- Подмена будущих цен на нечисловой sentinel не меняет deployment-признаки,
  scaler plan или training feature rows. При этом неизменённый snapshot SHA
  такую подмену отклоняет; при новом snapshot visible-prefix pins сохраняются.
- Изменение цены последнего доступного deployment-бара меняет deployment-признаки,
  но не training bank/scaler plan.
- При нулевой задержке значения двух признаков и proposal совпадают с замороженным
  старым чистым kernel на первом/последнем training-примере и deployment-решении.
  Это синтетическое совпадение, не доказательство эквивалентности historical runtime.
- Неправильный training_asof, решение вне origin-бара, выход за coverage,
  пропущенный/повторный бар, visible-price substitution, поздняя публикация,
  ранний arrival claim и смена revision отклоняются.
- Поздний exit для одного символа исключает обе строки соответствующего candidate.
  Пустой training bank остаётся без fit. Flat/abstention примеры не удаляются из
  scaler bank лишь потому, что они не предлагают сделку.
- Origin через 18 баров получает новый training bank, а не состояние старого scaler.

В fixture `B=4h`, origin=100B, training_asof=99B, cutoff=92B, два SYN-символа.
При lag=0 получено 46 строк, последняя training decision=72B, последняя dependency
deployment=99B. При lag=1ms или 120000ms: 44 строки, границы 71B/98B. При lag=B:
те же границы; при lag=B+1ms: 42 строки и 70B/97B. Это иллюстрирует дискретный
сдвиг сетки даже при небольшой задержке, а не измеренные рыночные задержки.
В следующем origin=118B получено 82 строки с новым row-bank SHA. Minimum
opportunities/ESS этим не аттестуются; например, 46 fixture-строк не доказывают
прохождение зарегистрированного минимума 64 opportunities.

Первый запуск остановился до создания результата: новая fixture использовала
имя revision, несовместимое с форматом старого kernel. Fixture исправлена на
`synthetic_revision_1`; соответствующее ограничение сохранено и в адаптере.
Замороженные исходники/контракты не менялись.

## Ограничения и следующий шаг

Parquet reads=0; scaler fit=0; model fit=0; financial evaluation=0. Pytest,
collector, сеть, фоновые процессы и trading runtime не запускались. Все девять
safety flags false. NumPy, pandas и exia не загружались.

Сейчас доказаны свойства синтетического адаптера, не реальный arrival, compute
completion, модельный SHA или causal full-refit на рынке. Действующий draft
availability-контракт остаётся с незаданными ingestion lag, evidence и revision
policy; costs и остальные launch-блокеры прежнего research-контракта не закрыты.

Следующий содержательный этап — связать этот origin/prefix plan с отдельной
спецификацией завершённых executable-label rows и training-matrix manifest,
не допуская label columns в deployment и не выполняя fit. Перед единственным
historical engineering smoke нужно явно зафиксировать допустимые timing/revision
и cost допущения, а также разрешить запуск. Нельзя выдавать такой smoke за
prospective и нельзя объявлять проверенный scaler-input plan обученным scaler.

Локальные результаты: `result.json`, `completion_receipt.json`. Этот Markdown
находится вне receipt. Публикация/commit/push не выполнялись.
