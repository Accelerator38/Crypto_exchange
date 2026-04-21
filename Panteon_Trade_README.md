# Panteon_Trade

`Panteon_Trade.py` — основной live-entrypoint для мультибиржевого торгового контура `Panteon`.

## Актуальная структура

- `Panteon_Trade.py` в корне — live-торговля и внутренний shadow-контур.
- `API_COMBO_TRADE.py` — тонкий generic-entrypoint, который запускает `Panteon_Trade`.
- `Start_MEXC.py` и `Start_BITGET.py` — биржеспецифичные launchers.
- `Retrodate_cryptotrade/crypto_exchange.py` — расчёты на исторических данных.
- `Genetics_DL_Agents/crypto_genetics.py` и `Genetics_DL_Agents/run_genetics.py` — обучение genetics-агентов.

Связанные данные и артефакты:

- `Results/<EXCHANGE>/...` — live/API результаты по конкретной бирже.
- `Retrodate_cryptotrade/CriptoData/` — исторические CSV и рыночные события.
- `Genetics_DL_Agents/Agents/` и `Genetics_DL_Agents/results/` — геномы, логи и результаты обучения.

## Поток запуска

1. Инициализируется exchange runtime, статистика и директория сессии.
2. Создаётся реальный игрок `Panteon` и набор shadow-агентов.
3. Выполняется `warmup` на исторических барах.
4. После warmup сбрасываются stale-позиции, таймеры и shadow-портфели.
5. `inject_live_state_into_player(...)` подтягивает реальные открытые позиции биржи в `Panteon`.
6. Стартует live-цикл и параллельный shadow-мониторинг.

## Текущие игроки

Канонические имена:

- `Panteon`
- `PanteonResearch`
- `PanteonNextResearch`

- `PANTEON_REAL_PLAYER=Panteon`
- `PANTEON_REAL_PLAYER=PanteonResearch`
- `PANTEON_REAL_PLAYER=PanteonNextResearch`

Для `PanteonNextResearch` дополнительно нужен:

- `PANTEON_NEXT_LIVE_OPT_IN=1`

## Shadow-имена

Канонические shadow-labels:

- `V_Panteon_shadow`
- `V_PanteonResearch`
- `V_PanteonNextResearch`

## Что проверять после запуска

- `combo_trading.log` — основной ход live-торговли, warmup, rotation, sync.
- `leaderboard.json` — состояние shadow-лидерборда и текущий режим.
- `status.json` — краткая сводка по реальному live-боту.
- `all_signals.csv` — все реальные и shadow-сигналы для оффлайн-анализа.
- `final_combo_report.json` — финальный отчёт с `promotion_gate` и сравнением shadow-игроков.
