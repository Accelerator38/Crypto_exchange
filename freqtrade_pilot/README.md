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

Docker опционален. На машине разработки Docker не был установлен. Compose
закреплен на `freqtradeorg/freqtrade:2026.7`.
