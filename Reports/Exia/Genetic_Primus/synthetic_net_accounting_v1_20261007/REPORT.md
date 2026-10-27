# V8 Clean — net accounting и cost scenarios

Дата: 2026-10-07. Завершена синтетическая инженерная проверка. Это не обучение,
не расчёт доходности на рынке и не калибровка Bitget.

## Реализация

Добавлены отдельные контракт, pure adapter и checker:

- `configs/genetic_primus_synthetic_net_accounting_contract_v1.json`;
- `src/exia/genetic_primus/synthetic_net_accounting_v1.py`;
- `tools/check_genetic_primus_synthetic_net_accounting_v1.py`.

Поддержаны только синтетические Long/NoTrade и линейные единицы SYN_LINEAR_USDT.
Reference price явно обозначает **synthetic mid**, не OHLCV-open и не фактический
fill. Старые labels и замороженные результаты не переквалифицированы.

Для входа и выхода отдельно задаются fee, half-spread и slippage в bps:
buy становится хуже на entry half-spread+slippage, sell — на exit компоненты.
Half-spread уже односторонний: повторно на два не делится. Fee каждой стороны
считается от её собственного adjusted fill notional. Spread/slippage не вычитаются
второй раз после учёта adjusted prices.

Allocation — доля общего quote-capital. Quantity определяется как allocation
quote / (entry fill × (1+entry fee rate)): входная комиссия помещается внутрь
выделенного бюджета. Это исследовательская convention, не биржевое исполнение
margin/lot rounding. Net quote = reference gross − execution drag − fees − signed
funding debit − явный borrow debit. Нормализация — общий quote-capital, не объём
отдельной сделки. При stress заполненное quantity может меняться внутри того
же quote budget; правила sizing неизменны.

Funding считается по явным settlements: quantity × mark × signed rate.
Положительная ставка для long списывает, отрицательная начисляет. Нужен явный
полный синтетический список; `[]` — объявленный fixture без settlements, не
доказательство нулевого funding рынка. Timestamp на границе входа/выхода
отклоняется: порядок относительно fill не реализован. Неизвестные расходы
не заменяются нулём. Borrow также задаётся явно.

NoTrade не имеет allocation, prices, funding events или borrow; все денежные
результаты нулевые. Paired-агенты используют одинаковый capital, период,
comparison key и cost scenario; два Long — ещё и одинаковые reference prices
и funding inputs. Разная allocation допускается. Delta = challenger net return
− incumbent net return, без bootstrap или разрешения promotion.

## Результаты

Сначала скомпилирован один новый pure file; затем напрямую проверен перевод bps.
`--run` и отдельный `--verify` прошли: **124 assertions**, **171 неизменённая
source/metadata identity**, receipt/result подтверждены.

- Независимый oracle через cash proceeds совпал с PnL decomposition на растущем,
  падающем и плоском synthetic path для zero/base/stress fixtures.
- Комиссии входа/выхода сверены отдельно с соответствующим notional.
- NoTrade остаётся нулевым во всех трёх сценариях; одинаковые агенты дают paired
  delta=0, Long−NoTrade совпадает с Long net return при общем capital.
- В случаях **без funding** более дорогой stress дал худший net результат.
  Это не универсальное утверждение при произвольных funding credits и sizing.
- Проверены signed funding, borrow debit, граничные/повторные settlement timestamps,
  неизвестный funding, отсутствующие rates и несовпадающие paired inputs.
- Неподдерживаемые market instrument, Short, OHLCV-open/observed-fill semantics,
  нарушение allocation, подмена контракта или включение authority отклоняются.
- Decimal precision=50 даёт повторяемую арифметику; изменение внешнего decimal
  context не изменило account result. Exchange rounding этим не подтверждён.

Synthetic base/stress rates — намеренно асимметричные тестовые константы, не
рекомендуемые комиссии и не измерения ликвидности. Старые 16/24/40 bps остаются
**aggregate roundtrip budgets**: adapter не принимает их как side schema и не
выводит из них отдельные fee/slippage. В старом research-контракте все десять
cost registration fields остаются null.

## Статус и дальнейшие действия

Parquet reads=0; scaler/model fit=0; рыночных финансовых оценок=0. Синтетическая
денежная арифметика выполнялась. Pytest, collector, сеть, фоновые процессы и runtime
не запускались. NumPy/pandas/exia не загружались; девять safety flags false.

`market_calibration=null`, fills_verified=false и market net objective registered
false сохранены. Gross OHLCV labels не превращены в mid-based net targets.
Не реализованы rolling portfolio marks, terminal policy, lot/tick rounding,
inverse instruments, margin/liquidation, реальный funding coverage или причинная
доступность cost/funding predictors. Прибыльность и превосходство над NoTrade
не проверялись.

Следующий шаг — **единый launch preflight**, а не расширение search/experiment
матрицы: свести оставшиеся timing/revision, price semantics, costs/funding,
terminal policy и objective requirements для одного historical engineering smoke.
Отделить измеренные сведения от разрешённых sensitivity-допущений и явно
зафиксировать, чего ещё не хватает. Только после регистрации такого сценария
и разрешения fit можно рассчитывать модель; этот этап такой допуск не даёт.

Локальные артефакты `result.json` и `completion_receipt.json` содержат hashes,
счётчики и ограничения, без цен/PnL-примеров. Markdown вне receipt. Предыдущие
исходники и результаты не менялись, публикация/commit/push не выполнялись.
