# Приёмка записанных данных: offline v1

2026-09-11. Новый слой проверяет входной пакет **до расчёта ордеров/PnL**.
Существующие release, matching kernel, portfolio controller и результаты
8–10 сентября не изменяются. Нет сети, collector, pytest, обучения, доступа
к ключам или торговому runtime.

## Что проверяется

- SHA и точный размер каждого исходного JSON; только локальные относительные
  пути внутри выбранного package directory. URL, выход через `..`, абсолютные
  пути, symlink escape и повторное использование одного файла запрещены.
- Строгий JSON: duplicate keys, нестандартные NaN literals и чрезмерная
  вложенность отклоняются. Decimal-поля используют строки; missing features — null.
- Общая полная full8-сетка **всех ожидаемых 4h-границ** внутри `[start,end)`.
  Проверка не ограничивается полнотой уже имеющихся групп: полностью
  потерянная граница решения также обнаруживается.
- Наличие mark/book по каждому инструменту; максимальные разрывы проверяются
  и по exchange time, и по receive time. Уровни стакана положительные, на
  tick/lot grid, правильно упорядочены и не пересекаются.
- Свежий mark перед каждым решением и наблюдаемый свежий book после latency,
  но до TTL/model expiry/конца сегмента. Это проверка наблюдаемости, не требование
  обязательного fill: защитный limit может всё равно не допустить сделку.
- Funding сравнивается с **явно переданным расписанием каждого инструмента**:
  interval, anchor, срок действия и source hash. Нет default «везде каждые 8h».
  Пропущенный settlement не интерпретируется как нулевая комиссия. Неожиданные
  и дублированные settlements отклоняются. Точная повторная доставка одного
  event ID остаётся идемпотентной согласно замороженному portfolio parser.
- Rule/fee source payload действительно присутствует и хешируется; значения
  и сроки действия совпадают с нормализованным bundle и покрывают весь сегмент.
- Каждый signal связан с values SHA, symbol, feature code SHA, source-bars SHA,
  cutoff, максимальным source close и declared availability не позже решения.
- Для каждого scaler/source model проверяются model/data/train-rows/fit-values
  hashes, количество строк, диапазон train, purge cutoff, label end и fitted-at.
  Source model manifest восстанавливается по исходному release contract.
- Классификация synthetic/observed_unqualified согласована у manifest, capture,
  bundle и reference sources. Изменение одного флага не превращает fixture
  в наблюдённые биржевые данные.

Проверка не вызывает `Replay.ingest` или portfolio calculation. Это отдельно
проверяется регрессией с запретом вызова matcher. Только отдельный synthetic
integration test запускает старый общий portfolio на принятом пакете.

## Что НЕ считается доказанным

`admitted_for_offline_replay=true` означает согласованный вход для локального
инженерного расчёта, **не допуск модели и не подтверждение подлинности данных**.

Всегда остаются false: `source_authenticity_verified`,
`feature_computation_verified`, `scaler_recomputed`, `calibration_verified`,
`qualification_eligible`, `trading_authorized`. Хеширование подтверждает
целостность представленных файлов, а не то, что они действительно получены
на бирже. Временные зависимости сопоставляются с declared lineage; raw bars
и median/IQR в этом инструменте не пересчитываются.

Coverage thresholds и funding schedule передаются явно и фиксируются SHA,
но их правильность для биржи/счёта ещё требует независимой проверки. Согласованно
подделанный пакет нельзя разоблачить одним хешированием. Никаких performance
LCB, выбора победителя, DSR/PBO или разрешений paper/live здесь нет.

## Входной формат

Это импортёр **нормализованного JSON package**, не универсальный парсер сырых
Bitget/WebSocket/CSV/Parquet дампов. OHLCV никогда не преобразуется в стакан.
Для конкретного raw recorder может понадобиться отдельный read-only mapping.

Три файла:

1. Manifest, schema `exia.genetic_primus.recorded_package/1`:
   evidence kind, portfolio contract SHA, start/end, coverage policy и
   `artifacts.bundle/provenance` с relative `path`, raw-byte `sha256`, `bytes`.
2. Bundle — прежняя `exia.genetic_primus.portfolio_bundle/1`, без изменения
   схемы замороженного portfolio. Его dataset SHA связывает `{rules,events}`.
3. Provenance, schema `exia.genetic_primus.recorded_provenance/1`:
   `capture`, `feature_lineage`, `scaler_lineage`, `funding_schedule`,
   `rule_sources`, `fee_source`. Profile feature dataset SHA связывает весь
   feature-lineage объект; values SHA связывает конкретные признаки.

Полный пример создаётся явно checker-флагом `--demo` и помечается synthetic.
Результат экспорта — отдельная копия исходного нормализованного bundle.
Оригиналы не изменяются. Все output-файлы создаются эксклюзивно.

Лимит всех трёх файлов вместе — **2 MiB**, manifest — до 64 KiB, вход — до
1024 events. Это короткий сегмент около хотя бы одной 4h-границы, а не streaming
pipeline многодневных HFT-записей. Автоматическое разбиение с переносом состояния
не реализовано; отдельные сегменты нельзя выдавать за непрерывный backtest.

## Команды

Проверка окружения без записей: сверяет замороженные source hashes, не выполняет replay.

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/import_genetic_primus_recorded_v1.py
```

Короткие прямые регрессии — отдельными командами:

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/check_genetic_primus_recorded_admission_v1.py --group coverage
```

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/check_genetic_primus_recorded_admission_v1.py --group provenance
```

Первое явное создание synthetic package (повтор не перезаписывает файлы):

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/check_genetic_primus_recorded_admission_v1.py --group provenance --demo
```

Пример импорта с **новыми** именами outputs; эта команда не считает портфель:

```powershell
& 'E:\Crypto_exchange\.venv\Scripts\python.exe' -X utf8 tools/import_genetic_primus_recorded_v1.py --manifest Reports/Exia/Genetic_Primus/recorded_admission_v1_20260911/package_manifest.json --report Reports/Exia/Genetic_Primus/recorded_admission_v1_20260911/admission_new.json --export Reports/Exia/Genetic_Primus/recorded_admission_v1_20260911/bundle_new.json
```

## Реальный блокер следующего прогона

Подтверждённый согласованный recorded package не предоставлен. Ограниченный
поиск по именам book/depth/mark/recorded-manifest в `Retrodate` и
`Reports/Exia/Genetic_Primus` (исключая benchmark-файлы) не дал соответствующих
JSON/Parquet записей. Это не доказательство отсутствия дампов под другими
именами/в другом каталоге. Известный старый funding manifest сам по себе
не заменяет book/mark и происхождение features/scaler.

Для перехода к рыночному расчёту нужен путь к реальным согласованным записям
и их reference data либо отдельно согласованный способ получения данных.
Новый genetic search или дополнительные synthetic profits этот блокер не снимают.
