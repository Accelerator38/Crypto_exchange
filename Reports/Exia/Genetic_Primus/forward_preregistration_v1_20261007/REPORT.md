# Genetic Primus: единый preregistration decision record

Дата: 7 октября 2026.

Создан [record.json](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/forward_preregistration_v1_20261007/record.json)
с фиксированными определениями, привязками к источникам, классификацией evidence,
неисполняемым инженерным планом и явными unresolved requirements.
Это регистрация **определений и плана**, а не регистрация подтверждённых данных,
готовая модель или разрешение на расчёт. Старые contracts не изменены.

## Что теперь фиксировано

- Bars 4h; deploy/retrain каждые 3 дня; purge 7 баров по label availability.
- Linear long/flat overlay: return12/efficiency12 плюс intercept — 3 коэффициента;
  ridge alpha 0.5; hidden layers 0; GA выключен; hyperparameter search 0.
- Scaler median/IQR, unweighted linear-interpolated quantiles, epsilon 1e-12,
  clip ±8; scaler fit только по selected training opportunity features,
  без фильтрации по знаку outcome. Полный scaler/coefficient refit каждого origin.
- Два кандидата: uniform и half-life90d с anchor20%; возраст берётся от конца
  label, потери также получают вес. Minimum opportunities64/weight ESS40,
  без forcing minimum trades.
- Target регрессии — gross delayed open-to-open return. Это **не** net profit.
  Пользовательская цель остаётся net >= 0 после всех costs с допустимыми
  временными просадками; отсутствие убытков не гарантируется.
- NoTrade — исследовательский constant cash reference; EMA opportunity явно
  определён по прежнему kernel как descriptive reference с той же execution/cost
  семантикой. Эквивалентность старым EMA результатам не утверждается.
- Требуются model SHA каждого решения, entry-model SHA переносимой позиции,
  scaler-fit rows, actual fit/model availability timestamps и hashes cutoff/
  folds/resample/audit. Наличие определения не является actual ledger.

Executable threshold rule **не зарегистрирован**. Старая формула flat-entry
через общий round-trip budget и held-continue=0 автоматически не переносится
в новые costs/funding semantics. Safe fallback — administrative no-fit/no-orders,
не распоряжение торговому runtime или существующей позиции.

## Инженерное окно выбрано по покрытию, не по доходности

Исходные metadata: full8 OHLCV coverage 2022-01-01 → 2026-09-28 UTC exclusive;
старое 3-day расписание начинается 2026-09-01. Цены не читались.

При decision внутри первого bar origin и delay < bar первый planned open
может сдвинуться на 2 bars. Для полного 3-day label обе границы сдвигаются,
и required exit open должен быть **строго внутри** архива:

`origin + 3 days + 2 bars < coverage_end_exclusive`.

| Nominal period | Latest required exit open | Статус metadata envelope |
| --- | --- | --- |
| 2026-09-25 → 2026-09-28 UTC | 2026-09-28 08:00 UTC | Вне архива, не выбирается |
| 2026-09-22 → 2026-09-25 UTC | 2026-09-25 08:00 UTC | Последний подходящий origin старого расписания |

Предложенный origin — **22–25 сентября 2026**, не новый prospective interval.
Record создан после этого периода и не называется preregistration исторических
outcomes. Выбор использует calendar/coverage, а не performance; engineering
results запрещено использовать для выбора лучшей модели или доказательства edge.

Training-asof запланирован за один bar до origin, cutoff — ещё на 7 bars раньше.
Это явное новое planning rule, **не** измеренная latency или подтверждение arrival.
History expanding от metadata coverage start. Venue/product исходных цен не
подтверждён: они не объявляются Bitget perpetual data на основании symbol notation.
Actual availability, gaps, duplicates, revisions и labels всё ещё не проверены.

## Budget и запрет дополнительных настроек

Один engineering origin, два pipelines, максимум два **планируемых** pipeline
runs; failed pipeline runs тоже занимают бюджет. Authorized market runs и fits
по-прежнему **0**. Metadata regression не является training attempt.

Нельзя добавлять seeds/features/retries, выбирать winner по engineering smoke
или менять определения после outcomes под тем же SHA. Common resample bank и
independent audit обязательны для будущих соответствующих сравнений, но сейчас
они не построены. Historical trial count/DSR/PBO/global registry не подтверждены.

## Что остаётся открытым

21 timing/data/cost field сохранено unresolved, ничего не заменено нулями.
Дополнительно явно выделены 9 protocol requirements:

1. Executable decision threshold rule.
2. Drawdown budget.
3. Concentration/turnover/drift gates.
4. Confidence/multiplicity rule.
5. Historical global trial count.
6. Power/minimum detectable effect/block design.
7. Actual folds/resample/audit banks.
8. New sealed start.
9. New sealed end.

Это разные фазы: например, power design не нужен для проверки schema файла,
но нужен для будущего статистического qualification. Нулевая economic floor
сама по себе не определяет sample size. Старый funding, borrow и instrument
evidence не становится подтверждённым от появления нового record.

## Проверки

Сначала успешно скомпилирован один новый модуль и выполнена чистая calendar
проверка, без prices/import trading modules. Первый checker обнаружил неверный
negative test: достаточную историю ошибочно сочли недостаточной. Исправлен только
test case до публикации; никакой model run/market attempt не запускался.

После исправления:

- **59 assertions passed**;
- **87 source/metadata/doc inputs неизменны** до/после, включая predecessor bank;
- `--verify` подтвердил receipt и **3 JSON outputs**;
- проверены exclusive coverage boundary, неверные grids/timeframes, source SHA,
  attempts to reopen authority и изменения record с пересчитанным self-hash.

Данные record, definitions и engineering plan имеют отдельные semantic hashes;
исходные файлы — byte hashes. Validator сверяет запись с bound definitions, а не
доверяет одному корректному self-hash.
Report создан после receipt и в input bank не входит.

Нет pytest, collector, WebSocket, archive/parquet reads, fits, network calls,
financial evaluations, background processes, paper/live или git push.
Все safety flags остаются false. Доходность модели на этом этапе не изменилась
и превосходство над NoTrade не доказано.

## Следующий шаг — реальные inputs, не новый mock audit

Нужна отдельная явная авторизация **bounded offline input verification** старых
parquet, перечисленных в
[data_manifest_origin_09.json](E:/Crypto_exchange/Reports/Exia/Genetic_Primus/prospective_3d_v1_202609/data_manifest_origin_09.json):
по одному файлу, identity/schema/coverage/grid/duplicates/gaps и timestamp basis,
с кратким агрегированным выводом без prices/API keys. Training, сеть, collector,
новый sealed interval и финансовый backtest остаются запрещены.
Такая авторизация не подразумевается текущим preregistration record.

После проверки inputs — согласовать research timing/cost assumptions и actual
lineage, затем отдельно разрешить единственный ограниченный инженерный refit.
Для real-money release сохраняются snapshot IOC/book/mark/funding/calibration
и instrument/risk gates; OHLCV fallback, paper/shadow/canary не разрешены.
