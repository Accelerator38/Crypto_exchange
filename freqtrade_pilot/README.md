# Official Freqtrade Bitget pilot

Пилот заменяет сложный runtime Pantheon готовым официальным движком Freqtrade
2026.7. Исходный код движка не копируется и не изменяется: точный upstream
commit закреплен в `upstream.lock.json` и подключен Git submodule.

`UpstreamSampleDryRunStrategy` наследует сигналы официальной `SampleStrategy`
без изменений и добавляет два fail-closed ограничения:

- запуск запрещен при `dry_run != true`;
- stop loss для Bitget futures размещается на бирже как market stop по mark
  price.

Это проверка готового торгового движка, а не заявление о прибыльности штатной
примерной стратегии. Положительный dry-run startup не разрешает paper/live.

## Native Windows setup

```powershell
git submodule update --init --recursive
py -3.12 -m venv .venv-freqtrade
.venv-freqtrade\Scripts\python.exe -m pip install --no-deps --editable vendor\freqtrade
.venv-freqtrade\Scripts\python.exe -m pip install -r vendor\freqtrade\requirements.txt
```

Скачать публичные Bitget futures candles, проверить данные, выполнить base и
cost-stress backtest, runtime live-guard и короткий dry-run startup smoke:

```powershell
.venv-freqtrade\Scripts\python.exe tools\run_freqtrade_pilot.py --download-days 60
```

Повторный прогон без скачивания данных:

```powershell
.venv-freqtrade\Scripts\python.exe tools\run_freqtrade_pilot.py
```

Команда не использует ключи биржи и никогда не включает реальные ордера.
Результат сохраняется в `Reports/FreqtradePilot/pilot_report_v1.json`.

## Результат первого full8 прогона

- движок Freqtrade и официальная интеграция Bitget работоспособны;
- 56 сделок при 0.04% fee/fill: expectancy отрицательная, profit factor < 1;
- fee-stress 0.08%/fill также отрицательный;
- максимальная просадка мала, но это следствие малого stake и не компенсирует
  отсутствие edge;
- в сырых 5m данных есть пропуски, а публичная история funding короче 60 дней.

Вердикт: движок можно оставить как простую execution/backtest основу, штатную
`SampleStrategy` следует отклонить. До paper/live нужна отдельная стратегия с
положительными OOS и cost-stress метриками.

## Первый фиксированный кандидат

`LongHorizonTrendStrategyV1` является отдельным low-turnover long/short
кандидатом на canonical full8 1h dataset 2022-2026. Параметры закреплены в
`candidates/long_horizon_trend_v1.json`; sweep и автоматическая оптимизация не
используются.

Единый ретроспективный запуск:

```powershell
.venv-freqtrade\Scripts\python.exe tools\run_freqtrade_long_horizon_v1.py
```

Launcher проверяет SHA исходных CSV, автоматически строит изолированный
Freqtrade dataset и выполняет development/validation/OOS/sanity при base и
stress costs. Старый dataset не содержит funding history, поэтому только для
этого ретротеста явно используется `futures_funding_rate=0`. Независимо от
результата launcher не имеет promotion authority и не включает paper/live.

Если canonical dataset отсутствует, он восстанавливается из публичного Bitget
API перед запуском кандидата:

```powershell
.venv\Scripts\python.exe tools\build_exchange_futures_retrodate.py `
  --exchange BITGET `
  --symbols BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,ADA/USDT,BNB/USDT,LINK/USDT `
  --start-date 2022-01-01 --end-date 2026-07-14 --timeframe 1h `
  --source-api bitget-v3 --parallel-workers 8 `
  --output-dir Retrodate\bitget_futures_history_v3_2022_20260714 `
  --report Retrodate\bitget_futures_history_v3_2022_20260714\build_report.json `
  --integrity-manifest Retrodate\bitget_futures_history_v3_2022_20260714\integrity_manifest.json
```

Первый зафиксированный прогон завершился
`TERMINAL_REJECTED_RETROSPECTIVE`. Средняя доходность сделки была положительной
на development/validation/OOS, но LCB оставался отрицательным во всех окнах и
для обоих направлений. В sanity-2026 средняя net expectancy стала отрицательной:
`-2.38 bps` при base costs и `-10.37 bps` при stress. Ни один символ не прошел
per-symbol LCB. Профиль `long_horizon_trend_1h_v1` больше не настраивается и не
перезапускается как новый кандидат.

Docker опционален. На машине разработки Docker не был установлен. Compose
закреплен на `freqtradeorg/freqtrade:2026.7`.
