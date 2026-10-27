# Compact net replay v1

Offline-only исследовательский replay. Не изменяет торговый runtime или release candidate. Все authority flags false означают отсутствие downstream права запуска, не запрет явно заказанных пользователем локальных диагностических расчётов.

## Фиксированные параметры

- Капитал: 10 000 USDT; восемь long/cash sleeves, target 10% на актив, без перераспределения и заимствования.
- Прогноз: 3 дня; переобучение/scaler каждые 3 дня, только созревшие labels, purge 7 баров.
- Архитектуры: linear 6→1, 7 параметров; MLP 6→8→1, 65 параметров, фиксированный ансамбль seeds 11/29/47, 195 коэффициентов.
- Uniform / half-life 90 дней с настоящей 20% uniform mixture; убытки получают веса наравне с прибыльными исходами.
- Порог: 20 bps выбран из 20/50/100 на linear uniform, 20 июня — 26 июля 2026. Даже лучший вариант убыточен; порог предназначен только для исследовательского benchmark, default остаётся NoTrade.
- Benchmark: 1 августа — 30 сентября 2026 включительно; 20 ранее замороженных origins, новых benchmark fits нет.
- Комиссия: 6 bps/fill; adverse drag normal 2 / stress 8 bps/fill. Зарегистрированные допущения, не подтверждённый account tariff.
- Funding: фактические знаковые rates; valuation по OHLCV open того же timestamp — proxy, не exchange mark.

## Семантика риска и ограничения

Drawdown измерен по 4h reference-open NAV, не по intrabar extrema, liquidation mark или исполнимым стаканным ценам. Поэтому 9.31% в replay не доказывает соблюдение 10% live drawdown limit. Бюджеты 10/20/30% — исследовательские post-evaluation gates, не stop-loss, не intraday risk engine и не согласованный риск денег.

Целевой cash reserve не поддерживается непрерывным ребалансированием: неизменный target сохраняет units, actual weight дрейфует. Исполнение — следующий 4h open после доступности закрытого бара; это не обещание исполнения сразу после закрытия. Реальные latency, minimum order/notional, quantity ticks, стакан, market impact и funding marks не доказаны.

Aug19–22 имеет необычный одновременный скачок всех восьми активов. Окно сохранено, но независимая проверка исходных цен ещё необходима. Положительный итог открытого benchmark не заменяет такой проверки, net-сигнала на предшествующем периоде, multiplicity control или нового prospective.

Forecaster обучен на gross MSE; выбор порога и финансовая оценка используют net log-capital growth. Это не direct net-growth обучение нейросети. GA/random search не запускались: prebenchmark net-сигнал не подтверждён, global trial registry неполон.

## Воспроизведение без повторения существующих расчётов

Использовать только `E:\Crypto_exchange\.venv\Scripts\python.exe`, последовательно, один origin/команду. Никакого pytest, collector, фоновых процессов или сети.

1. `tools/run_genetic_primus_compact_net_replay_v1.py prepare --folder <new-folder>` — фиксирует sources/config/model hashes до fits.
2. `... fit --folder <new-folder> --index N` — один linear calibration fit, N=0..11. Повтор существующего fit запрещён.
3. Исходный `calibrate` сохранил шесть financial ledger, затем остановился на bootstrap: trailing cash slice 1 день короче block 3 дня. Не повторять его и не редактировать frozen runner.
4. `tools/finish_genetic_primus_compact_net_replay_v1.py register` и затем `calibrate` — immutable continuation, использует уже готовые ledger. Последний короткий остаток объединяется с предыдущим descriptive fold. Seeds/block/objective/models неизменны.
5. `tools/run_genetic_primus_compact_net_replay_v1.py benchmark --policy <policy>` — один из четырёх compact arms, NoTrade, EMA48_long_only; два cost scenarios. Повтор существующего output запрещён.
6. `tools/finish_genetic_primus_compact_net_replay_v1.py report`, затем `tools/audit_genetic_primus_compact_net_replay_v1.py` — shared independent bootstrap banks, отчёты, графики, сверка ledger без повторных financial calls.

Для нового folder последующие команды также получают `--folder`; audit по умолчанию относится к завершённому текущему artifact. Не менять период/пороги в существующем контракте или output. Новый экономический вариант требует новой registration и учёта testing budget.

Текущий artifact: `Reports/Exia/Genetic_Primus/compact_net_replay_v1_20261007`. Его REPORT_RU.md, results.json, reconciliation.json и equity_ru.png содержат результат. Все исходные source hashes и outputs сохранены; bootstrap repair отдельно описан в continuation_registration.json.
