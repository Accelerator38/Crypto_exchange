# V8 Clean — engineering protocol draft и offline full8 feature adapter

Дата: 2026-10-07. Реализован data/feature-компонент для рыночного формата строк,
но рыночная панель не декодировалась. Preflight NO-GO для fit/финансовой оценки
не снят. Это не новый поиск или тест доходности модели.

## Реализация

- `configs/genetic_primus_offline_market_engineering_protocol_v1.json`: отдельный
  draft с SHA старого origin_09 manifest, history/origin/train_asof, двумя прежними
  кандидатами и budget максимум двух pipeline runs, включая failures.
- `src/exia/genetic_primus/offline_market_feature_adapter_v1.py`: принимает полные
  бары физического full8 формата с `timestamp,symbol,OHLCV,source_rows`; pandas,
  NumPy, PyArrow, торговые пакеты и filesystem для данных не нужны.
- `tools/check_genetic_primus_offline_market_feature_adapter_v1.py`: проверяет
  metadata bindings и fabricated full8-shaped rows. Это **не** цены Bitget.

В draft оставлены незаданными 14 полей: три availability, три timing и восемь
accounting. Старые frozen записи не редактируются. Эти поля пересекаются с
предыдущими registration obligations, а не являются ещё 14 независимыми gates.
Raw parquet read/fit под новым draft не разрешены; девять safety flags false.

## Найденное отличие legacy

Статически прочитанный `rolling_linear_retrain_v1.build_origin_examples` строит
пример каждые 18 баров от origin-anchored calendar. Старый label — open начала
окна → close последнего бара, тогда как новый контракт требует delayed
open → delayed open. Его нельзя переиспользовать без изменения семантики.
Legacy модули импортируют NumPy/pandas; они **не импортировались и не исполнялись**.

Calendar нового adapter сохраняет 18-барный шаг. Предыдущие маленькие synthetic
fixtures с примерами каждые 4h были только тестовой сеткой, не регистрацией
нового training opportunity schedule. Это явно закреплено в draft.

## Работа feature state

Adapter требует последовательные полные full8-бары с фиксированного history start,
source_rows=240, без пропусков/дубликатов. Полностью проверяет бар до commit:
ошибка одного символа не продвигает состояние остальных.

Время доступности явно задаётся как start+4h+scenario ingestion lag. Null не
превращается в ноль. Недоступный бар не влияет на цены/хэш/state и может быть
повторно подан после его scenario publication. Snapshot требует полный доступный
prefix к decision asof; устаревший или ретроспективный запрос отклоняется.

EMA12/48 вычисляются последовательно с first-close seed, без перезапуска на
каждом origin. Хранятся 13 строк и четыре EMA12-значения на символ, а не вся
история. `return_12`/`efficiency_12` используют 13 закрытий; proposal — весь
закреплённый prefix, minimum48. Это feature state, **не** warm-start scaler/model.
Flat path даёт нулевые признаки и abstention.

Prefix identity — отдельная по-барная SHA-chain, связанная с manifest/history/
revision-policy identity; она не выдаётся за старый SHA JSON-массива. Declared
revision-policy digest не является доказательством оригинальных ревизий.

## Проверки

Сначала скомпилирован один новый pure file и напрямую проверен calendar.
`--run` и отдельный `--verify` успешны: **435 assertions**, **182 неизменённые
source/metadata inputs**, receipt/result совпадают.

- Формулы признаков и EMA proposal сопоставлены со старым pinned pure EMA kernel.
- При fabricated lag 0/1ms/4h+1ms на одинаковом asof доступно 100/99/98 баров.
  Непубликованная sentinel-цена не меняет snapshot или SHA.
- После публикации валидного следующего бара state продвигается ровно на один бар.
  Ретроспективное извлечение snapshot из уже более позднего state отклоняется.
- Нарушения source_rows, пропуски, дубликаты, некорректные значения, недостаточный
  warmup, null/bool lag и подмена candidate calendar/authorization отклоняются.
- При сбое bar commit полностью атомарен. Tail хранит только 8×13 строк.
- Synthetic calendar origin=100B/history=0 возвращает 64B и 82B после warmup,
  не все 4h timestamps. Это проверка правила, не рыночные training opportunities.

## Ограничения и дальнейшая работа

Parquet reads=0, scaler/model fit=0, market financial evaluation=0. Цены fixtures
все вымышленные; в артефактах только hashes/counters. Сеть, collector, pytest,
фоновые процессы и runtime не запускались. Доходность/NoTrade не сравнивались.

Этот компонент сам не загружает и не аутентифицирует parquet и не доказывает,
что переданные caller строки происходят из pinned manifest. Metadata binding
проверен, raw-файлы на этом шаге не перепроверялись. В будущем decoder должен
проверять size/SHA каждого разрешённого файла до/после чтения, собирать целые
отсортированные full8-бары и фиксировать derived-row provenance.

Delayed labels, per-origin scaler/model refit, price/cost bridge и непрерывный
net ledger пока не подключены к этому market adapter. Calendar не проверяет
label maturity/purge: это отдельная обязательная стадия перед training bank.
Arrival/compute completion не измерены. Нельзя заполнить null параметры без
отдельной регистрации допущений и назвать результат calibrated.

Следующий практический этап — подключение проверенного file-by-file decoder к
этому adapter и создание market feature/label manifests **без fit** после
явного разрешения такой materialization и фиксации ingestion/revision scenario.
Использовать только старые 11 файлов; не добавлять datasets, seeds, модели или
новый sealed interval. Перед обучением дополнительно закрыть net objective,
cost/funding/terminal accounting, timing и разрешение fit.

Артефакты: `result.json`, `completion_receipt.json`. Markdown вне receipt.
Изменения только новые scoped V8 research files. Старый runtime/ключи/ордера,
замороженные contracts/results не затронуты. Публикации/commit/push не было.
