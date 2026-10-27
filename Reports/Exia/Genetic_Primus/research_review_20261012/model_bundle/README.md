# Genetic Primus offline research shortlist

Дата публикации 12 октября 2026. Это исследовательские snapshots и чистые численные функции, **не trading release**. Default допуска — NoTrade. Все девять runtime authority-флагов false.

## Состав

- `Huber_H3.json`: полный сохранённый fit и scaler 30 сентября 2026 00:00 UTC. Линейная регрессия, 7 параметров, Huber delta 1.345, L2 0.01. Labels ежедневно, forecast 3 дня; решения и retrain раз в 3 дня. Вход gross forecast >20 bps, удержание прежних units. Исследовательский кандидат, проверен уже, чем C_V.
- `Ridge_C.json`: сохранённый fit той же даты; linear6, 7 параметров, uniform, L2 0.01. Ежедневные решения, retrain раз в 3 дня. Контроль для C/C_V, не подтверждённая alpha.
- `offline_models.py`: проверка SHA, численный prediction, последнее значение шести признаков, строго lagged covariance и target weights EMA_V/Passive_V/Ridge_C_V. Никаких orders, exchange client, collector или fit на import.
- `release_metadata.json`: источник, model/scaler SHA, cutoff, training identity, статус каждого кандидата. Latest mature label обоих snapshots — 28 сентября 00:00 UTC.
- `test_vectors.json` и `verify_offline_bundle.py`: четыре набора численных входов с сохранёнными эталонными выходами; 9 самостоятельных проверок, не требующих raw archive или старых модулей.

EMA_V и Passive_V — независимые алгоритмические baselines, а не обученные нейросети. Их risk controller: 60 дневных returns, конец истории на день раньше решения, 20% diagonal covariance shrinkage, floor daily vol 0.2%, target daily portfolio vol 0.5%, max 10%/asset и 80% gross, только уменьшение экспозиции. Эти числа исследовательские, не согласованный риск реальных денег.

## Корректные входы

Порядок: ADA/USDT, BNB/USDT, BTC/USDT, DOGE/USDT, ETH/USDT, LINK/USDT, SOL/USDT, XRP/USDT. Времена — UTC миллисекунды **закрытия и доступности**, не open timestamps свечей. `latest_features` требует весь registered EMA history anchor и не допускает незаметного укорочения истории. Значение anchor должно поступать из доверенного manifest, не произвольно назначаться вызывающим кодом.

Признаки: distance_ema48, return_18, efficiency_18, relative_return_18, volatility_48, breadth_ema48. Прогноз — gross outcome закрытие→закрытие за 3 дня, не net PnL. Перед inference веса и scaler проверяются по SHA. Ошибка данных должна приводить к отказу, а не подмене признаков нулями.

Функция `research_targets` возвращает только диагностические target weights. Она **не** реализует переходы units, fees, funding, жизненный цикл заявки или историю retrain. Нельзя взять snapshots последнего origin и использовать их для всего августа: это внесёт future training leakage. Для исторического replay нужны соответствующие каждому origin модели.

Результаты последних исследований, ограничения и графики: [отчёт о состоянии](../REPORT_1_CURRENT_STATE_RU.md). Новые чистые проверки: [verification](../verification.json); исходные 800 проверенных hashes и итоговые метрики: [evidence](../evidence.json).

## Воспроизводимость и ограничения публикации

Численная зависимость — NumPy; точная использованная версия записана в verification. Пакет самодостаточен для prediction и target inference на предоставленных правильных входах. В него намеренно не входят торговые ключи, raw market archive, все исторические fits и execution runtime. Поэтому эта небольшая публикация **не** обеспечивает полный train-and-backtest-from-scratch без исходного frozen research workspace. Pins позволяют проверить наличие и неизменность требуемых исходников в таком workspace.

Самостоятельная проверка из корня данного workspace:

```powershell
& 'E:/Crypto_exchange/.venv/Scripts/python.exe' -X utf8 'Reports/Exia/Genetic_Primus/research_review_20261012/model_bundle/verify_offline_bundle.py'
```

Ожидается `standalone_checks_passed: 9`. Это проверка корректности переноса, не доказательство доходности. На другом компьютере нужен собственный Python с NumPy; путь выше относится к исследовательскому ПК.

Builder `tools/build_genetic_primus_review_20261012.py` читает существующие исследования и exclusive-create генерирует audit/export/charts. Это не downloader и не trainer; stages нельзя бездумно повторять поверх запечатанных выходов. На Windows использовался только `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8`, один foreground процесс и один BLAS thread.

Политики выбраны ретроспективно. Ни conditional comparison, ни новый seed, ни эта публикация не превращают раскрытые периоды в prospective. Ни одна модель не получила real-money approval.
