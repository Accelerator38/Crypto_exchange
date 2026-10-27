# Execution input audit: full8 августа–сентября 2026

Дата: 2026-10-08. Локальная проверка входов; не обучение и не финансовый replay.

## Результат

Зарегистрированный bundle: `Retrodate/bitget_full8_postclose_execution_bundle_v1/manifest.json`.
Наличие: False. Structural issues: `['registered_execution_bundle_missing']`.
48 component/symbol требований рассмотрено; **0 квалифицированных для нового post-close replay**.
Это статус registered input gate, не утверждение об отсутствии всех данных на диске.
10 прямых pure checks прошли без pytest/collector.

## Funding: полезная находка без новых загрузок

Исходный normalized файл не изменён: 1464 reported-rate строк,
по `{'ADAUSDT': 183, 'BNBUSDT': 183, 'BTCUSDT': 183, 'DOGEUSDT': 183, 'ETHUSDT': 183, 'LINKUSDT': 183, 'SOLUSDT': 183, 'XRPUSDT': 183}`; terminal timestamp 1 октября в нём отсутствует (0 строк).
По frozen pins проверены 57 funding receipt inputs/outputs и 48 raw file pins.
В 24 сохранённых raw pages (2160 строк) найдены unique
reported rates на 2026-10-01 00:00 UTC для 8/8 активов.
Создана отдельная immutable выборка `terminal_funding_reports.json` со source file/row hashes.
Старые rates/ledger/результаты не исправлялись задним числом.

Записи — **reported funding rates, не наблюдаемые денежные settlements**.
Settlement marks present=False; independent schedule
verified=False.
У каждой новой записи strategy_available_us=null и settlement_mark=null;
consumable_as_funding_cashflow=false. October requestTime не подменяет историческую
доступность ставки в August/September. Имя settlement_ms старого normalized
формата тоже само по себе не доказывает debit/credit или contemporaneous mark.

## Что имеется и чего не хватает

- Funding: rates для 8/8 и terminal reports есть, но marks/settlements/availability не квалифицированы.
- Marks: старый audit содержит только `{'BTC': 29, 'ETH': 29, 'SOL': 29}` common 4h sample candles,
  не continuous full8 marks всего 61-дневного периода и terminal хвоста.
- Fees: current default taker proxy `{'BTCUSDT': '6.0000', 'ETHUSDT': '6.0000', 'XRPUSDT': '6.0000', 'ADAUSDT': '6.0000', 'LINKUSDT': '6.0000', 'DOGEUSDT': '6.0000', 'SOLUSDT': '6.0000', 'BNBUSDT': '6.0000'}` bps не доказывает
  historical/account fee tier. Current-rule feasibility не является historical rule provenance.
- L2: recorded BTC архивы декабря 2024 и book ticker января 2025 вне целевого периода;
  не подменяют full8 2026 depth/sequence/coverage и volume replay.
- Closed features: prices/Parquet на диске не доказывают full causal EMA state,
  maturity labels, отдельный scaler и consumer availability. Training history status:
  `not_registered_no_reuse_of_old_clock_labels`.

Отдельно прочитана только старая 1m OHLCV manifest декларация:
506880 строк, 2026-06-15T00:00:00+00:00 — 2026-07-28T23:59:00+00:00.
Её 60474408 byte Parquet **не читался и не хешировался сейчас**.
Этот пример — не доказательство отсутствия более новых OHLCV в других кешах.
Filename inventory ограничен Retrodate и price audit; максимум 8192 entries,
512 directories, 2048 candidate names на root. Coverage по имени не выводится.
Другие directories/Downloads, runtime, ключи и ордера не исследовались.
У четырёх предоставленных gzip проверены только существование/size; содержимое не открывалось.

## Calendar и terminal boundary

Период ledger: 2026-08-01 00:00 UTC — 2026-10-01 00:00 UTC exclusive.
Минимальный зарегистрированный terminal tail: до 2026-10-01T00:00:07.100000+00:00 exclusive.
Funding timestamp ровно на Oct 1 boundary относится к terminal ledger перед fills:
он не должен исчезнуть из-за exclusive normalization end.

Согласно старому postclose plan, terminal closure — отдельное решение;
не новый 3-дневный label и не перенос выхода ledger на Oct 3.
49 баров — минимум features, не разрешение пересоздать EMA только на 49 барах:
нужен зарегистрированный full causal prefix. Training history scope остаётся
отдельным незавершённым входом, не покрывается одной validation выборкой.

## Следующие работы по приоритету

1. Зарегистрировать источники target-period L2/sequence/coverage и contemporaneous marks;
   L2 нужна для decision/terminal execution windows, marks — для непрерывной NAV/DD.
   Границы L2 windows зависят от доказанной/отдельно зарегистрированной availability,
   нельзя считать номинальный bar close реальным временем получения features.
2. Сопоставить найденные rates/terminal reports с settlement marks, signed exposure,
   schedule и availability provenance. Пока не включать их в финансовый ledger.
3. Квалифицировать historical rules/fees и causal training-history/feature clocks.
   Если используется latency/fee model вместо наблюдений — отдельный заранее
   зарегистрированный исследовательский режим, не observed execution qualification.
4. Только после input gate — один ограниченный реалистичный replay, затем полный
   период с непрерывным капиталом/ledger, графиком NAV и costs attribution.
   Не запускать новое обучение на старых next-open labels ради ещё одного ranking.

Все 9 authority flags false: 0 fits, 0 financial labels, 0 market strategy replays,
0 network requests, 0 runtime/orders. Результаты прибыли этого этапа не существуют.
