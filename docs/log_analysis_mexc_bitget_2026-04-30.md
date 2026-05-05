# Отчет по текущим логам MEXC и BITGET

Дата анализа: 2026-04-30, активные логи примерно до 14:22 MSK.

## Источники

| Источник | Путь |
|---|---|
| MEXC live log | `C:\Work\Crypto_exchange\Results\MEXC\2026-04-29_21-29-48\trading.log` |
| MEXC status | `C:\Work\Crypto_exchange\Results\MEXC\2026-04-29_21-29-48\status.json` |
| MEXC signals | `C:\Work\Crypto_exchange\Results\MEXC\2026-04-29_21-29-48\all_signals.csv` |
| MEXC leaderboards | `C:\Work\Crypto_exchange\Results\MEXC\2026-04-29_21-29-48\leaderboard_agents.json`, `leaderboard_players.json` |
| BITGET live log | `C:\Work\Crypto_exchange\Results\BITGET\2026-04-29_21-29-53\trading.log` |
| BITGET status | `C:\Work\Crypto_exchange\Results\BITGET\2026-04-29_21-29-53\status.json` |
| BITGET signals | `C:\Work\Crypto_exchange\Results\BITGET\2026-04-29_21-29-53\all_signals.csv` |
| BITGET leaderboards | `C:\Work\Crypto_exchange\Results\BITGET\2026-04-29_21-29-53\leaderboard_agents.json`, `leaderboard_players.json` |

## Краткий вывод

Система в целом работает: оба процесса находятся в `live_futures`, бары идут без пропусков, `status.json`, `all_signals.csv`, графики и `dashboard.html` обновляются. MEXC сейчас работает штатно и почти в нуле. BITGET работает, но качество исполнения и устойчивость хуже: за текущий запуск просадка `-2.51%`, много защитных закрытий, были 2 failed order и один свежий сбой batch-цен.

Главный технический риск найден не в самих агентах, а в связке `snapshot/reconcile -> inherited_position_policy -> PositionGovernor`. После временной потери/искажения snapshot позиции удаляются из `_open_pos`, затем добавляются обратно как внешние с `stale_bars=0`, и следующий бар может закрыть их как `STALE(1bars)`. Это уже проявилось на обеих биржах.

## План/факт по операциям

| Метрика | MEXC | BITGET |
|---|---:|---:|
| Uptime | `16:51:22` | `16:51:07` |
| Bar range в активном логе | `5761 -> 6763` | `5734 -> 6707` |
| Пропуски баров | нет | нет |
| Live bars | `1002` | `973` |
| Current balance | `123.2892 USDT` | `44.0816 USDT` |
| PnL | `-0.0922 USDT`, `-0.0748%` | `-1.1331 USDT`, `-2.5061%` |
| Max drawdown | `0.6118%` | `3.1345%` |
| Real signals | `63` | `139` |
| Trades | `30` | `54` |
| Current positions | 3 | 0 |

Операционный статус:

- MEXC: цикл идет примерно раз в минуту, обрабатываются 18 символов, funding обновляется, сейчас `HOLD`, открыты `ADA SHORT`, `TAO SHORT`, `ZEC LONG`.
- BITGET: цикл идет, обрабатывается 51 символ, сейчас `HOLD`, открытых позиций нет после защитного закрытия в 14:13 MSK.
- Дашборды обновлялись сегодня: `C:\Work\Crypto_exchange\Results\dashboard.html` и PNG-графики в активных папках бирж.

## Найденные ошибки и аномалии

### 1. BITGET: failed order при размещении RAVE и MYX

В 01:03 MSK два шорта не были размещены:

- `RAVE SHORT amount=3.0 ... code=-1 msg=bitget POST ... place-order`
- `MYX SHORT amount=9.0 ... code=-1 msg=bitget POST ... place-order`

Перед первым ордером был warning `position mode fetch failed ... fallback=hedge_mode`. После failed order в 01:05 появились строки `[reconcile] RAVE удалён из _open_pos` и `[reconcile] MYX удалён из _open_pos`, то есть внутренний учет успел считать позиции открытыми, хотя биржа ордера не приняла.

Оценка: это баг синхронизации состояния. `_open_pos` не должен обновляться до подтвержденного `order_id`, а failed order должен сохранять полный ответ биржи.

### 2. Snapshot/reconcile может порождать ложные внешние позиции

Примеры:

- MEXC 01:02-01:05: DNS/timeout по ценам, затем `TAO` и `ZEC` удалены из `_open_pos`, добавлены обратно как внешние и закрыты как `STALE(1bars)`.
- MEXC 08:03-08:05: `NEAR` открыт, затем удален/добавлен reconcile и закрыт как `STALE(1bars)`.
- BITGET 14:11-14:13: ошибка `spot batch`, snapshot на один цикл показал `позиций=0`, затем 4 позиции вернулись как внешние и были закрыты как `STALE(1bars)`.

Корень:

- `settings.txt`: `inherited_position_stale_bars = 0`
- `src\panteon_runtime\exchange_api_runtime.py`: `inherited_position_policy()` назначает внешним позициям `stale_bars` из этой настройки.
- `src\panteon_runtime\agent_safety.py`: `PositionExitGovernor` закрывает позицию, если `current_bar - bar_opened > stale_bars` и движение меньше порога.

Оценка: это главный баг/риск текущего запуска. Он превращает краткий сбой API в реальное закрытие позиций.

### 3. BITGET: свежая ошибка batch-цен

В 14:11 MSK:

`Ошибка цен Bitget (spot batch): bitget GET https://api.bitget.com/api/v2/spot/market/tickers`

После нее был описанный выше цикл `0 позиций -> 4 позиции -> close_all`. Ошибка сама по себе может быть внешней сетевой/биржевой, но реакция системы на нее слишком агрессивная.

### 4. MEXC: сетевые сбои без остановки процесса

В 01:02-01:39 MSK были `NameResolutionError` и `Read timed out` по `api.mexc.com`. Бары не пропали, процесс восстановился, но в тот же период сработал риск `STALE(1bars)` по переобнаруженным позициям.

Оценка: сам retry работает, но destructive reconcile во время нестабильного API надо блокировать.

### 5. BITGET: много защитных закрытий

Сводка `PositionGovernor` по активному логу BITGET:

| Причина | Количество |
|---|---:|
| SL | 65 |
| TP | 22 |
| STALE | 21 |
| TRAIL | 14 |

Большая часть warning-линий здесь не аварии, а штатный риск-менеджмент. Но концентрация SL по низколиквидным/волатильным символам (`BSB`, `ORCA`, `AIGENSYN`, `NAORIS`, `PRL`) говорит, что профиль BITGET слишком агрессивен для текущего баланса около 45 USDT.

## Оценка агентов

### MEXC

Лучшие по текущему leaderboard:

- `V_LiveAfterShock`: `+0.3604%`, 19 закрытых сделок, Sharpe `9.5866`.
- `V_NeutralRangeScalper`: `+0.1224%`, 5 закрытых сделок, Sharpe `9.2148`.
- `V_LiveVolCompress`: около нуля `+0.0061%`, 32 закрытых сделки.

Слабые:

- `V_LiveRegimePullback`: `-1.0801%`, Sharpe `-32.1001`.
- `V_LiveCrashHunter`: `-0.9824%`, win rate `14.29%`.
- `V_LiveTrendFollow`: `-0.9441%`.
- `V_LiveMeanRev`: много сделок, win rate высокий `55.32%`, но итог `-0.7397%`, что похоже на плохое соотношение profit/loss или издержки.

Агенты `V_GeneticsNeutral/Bullish/Bearish` имеют высокий unrealized PnL, но `closed_trades=0`, поэтому их нельзя считать доказанными для live-лидерства.

### BITGET

Лучшие:

- `V_LiveAfterShock`: `+2.7290%`, 96 закрытых сделок, win rate `53.12%`.
- `V_LiveOIBreakout`: `+1.7883%`, 16 закрытых сделок, win rate `100%`.
- `V_CarryFlowAgentV2`: `+0.4155%`, 21 закрытая сделка.

Слабые:

- `V_PlayerFunding`: `-10.0640%`, drawdown `11.4%`.
- `V_MomentumScalper`: `-7.0142%`, уже в `quarantine`.
- `V_LiveTrendFollow`: `-6.8474%`.
- `V_FundingArb`: `-5.34%`, уже в `quarantine`.

Вывод: на BITGET стоит усиливать `LiveAfterShock`, `LiveOIBreakout`, `CarryFlowAgentV2` и резко ограничить `PlayerFunding/FundingArb/TrendFollow/MomentumScalper`.

## Оценка игроков

### MEXC

Лучшие игроки:

- `V_SoloLiveVolCompress`: `+0.0061%`, 32 закрытых сделки, лучший из достаточно насыщенных live-игроков.
- `V_PanteonMeanRevResearch`: `-0.0913%`, 24 закрытых сделки, просадка низкая.
- `V_SoloFundingArb`: `+0.9153%`, но только 2 закрытых сделки, статистика недостаточна.

Слабые:

- `V_SoloLiveRegimePullback`: `-1.0801%`.
- `V_SoloLiveCrashHunter`: `-0.9824%`.
- `V_SoloLiveTrendFollow`: `-0.9441%`.
- `V_PanteonResearch/Consensus/Defensive/Next` сейчас отрицательные.

### BITGET

Лучшие игроки в shadow/live leaderboard:

- `V_PanteonDefensiveResearch`: `+3.0220%`, 41 закрытая сделка, Sharpe `35.6643`.
- `V_PanteonNextResearch`: `+2.8527%`, 60 закрытых сделок.
- `V_PanteonTrendResearch`: `+2.6033%`, 68 закрытых сделок.
- `V_PanteonConsensusResearch`: `+2.5799%`, 58 закрытых сделок.

Слабые:

- `V_PlayerFunding`: `-10.0640%`, `purgatory`.
- `V_SoloLiveTrendFollow`: `-6.8474%`, `purgatory`.
- `V_SoloFundingArb`: `-5.34%`, `purgatory`.
- `V_SoloLiveCrashHunter`: `-2.2623%`, `purgatory`.

Важное расхождение: лучшие BITGET-игроки в leaderboard плюсовые, но реальный портфель BITGET в минусе `-2.5061%`. Значит, есть проблема не только в выборе модели, но и в live execution layer: округления объема, failed orders, комиссии/проскальзывание, premature stale-close, и/или разница между shadow PnL и реальной биржей.

## Оценка Пантеона

MEXC:

- Пантеон действовал консервативно: 63 real signal, 30 trades, текущий PnL около нуля.
- Основные real contributors: `Panteon_shadow` 31 сигнал, `SoloLiveVolCompress` 25, `PositionSafety` 7.
- Последний режим `HOLD` выглядит корректно: активных сигналов нет, открытые позиции удерживаются.

BITGET:

- Пантеон слишком активен для малого счета: 139 real signal и 54 trades за тот же uptime.
- Основные contributors: `PanteonNextResearch` 82, `Panteon_shadow` 22, `PositionSafety` 20, `SoloLiveVolCompress` 12.
- Текущий `HOLD` после закрытия всех позиций корректен, но закрытие 4 позиций в 14:13 было спровоцировано технической цепочкой `snapshot -> external -> STALE`, а не полноценным торговым решением.

## Рекомендации

### P0: исправить безопасную синхронизацию позиций

1. Не удалять позиции из `_open_pos` по одному плохому/пустому snapshot. Требовать 2-3 последовательных подтверждения `positions=0` и здоровый ответ balance/positions.
2. Если в последние 1-2 минуты были `price error`, `snapshot zero`, `position mode fetch failed` или `balance error`, запрещать destructive reconcile и `STALE`-закрытия по переобнаруженным позициям.
3. Разделить происхождение позиции:
   - `source=panteon_order` для позиций, открытых ботом и подтвержденных `order_id`;
   - `source=manual_external` для реально внешних/ручных позиций;
   - `source=recovered_after_snapshot_gap` для восстановленных после сбоя.
4. Для `source=recovered_after_snapshot_gap` не применять `inherited_position_stale_bars=0`.

### P0: не обновлять internal position book при failed order

1. `_open_pos` должен обновляться только после успешного ответа биржи и наличия `order_id`.
2. Failed order должен писать полный HTTP/status/body/code/msg, а не только `code=-1`.
3. Для BITGET добавить retry только на безопасные transient-ошибки, но не дублировать ордер, если статус неизвестен.

### P1: изменить stale-политику

Текущая настройка:

`inherited_position_stale_bars = 0`

Рекомендуемое значение:

`inherited_position_stale_bars = 30` минимум, лучше `60-90` для live futures.

Также лучше добавить отдельную настройку:

`recovered_position_stale_bars = 120`

Так восстановленные после API-сбоя позиции не будут закрываться через один бар.

### P1: ужесточить BITGET risk profile

1. Снизить `trade_fraction` для BITGET с `0.06` до `0.03-0.04`.
2. Ограничить одновременные позиции BITGET до 2-3, пока real PnL не выйдет из минуса.
3. Поднять `liquidity_min_adv` для BITGET. Текущее `5000` слишком мягкое для микрокапов.
4. Ввести per-symbol блокировку/понижение веса для символов с серией SL: `BSB`, `ORCA`, `AIGENSYN`, `NAORIS`, `PRL`.
5. Не пускать в real symbols с `closed_trades < 3` и `win_rate < 50%`, даже если shadow-агент дает сигнал.

### P1: перераздать веса агентов

MEXC:

- Увеличить вес `LiveAfterShock`, `NeutralRangeScalper`, умеренно `LiveVolCompress`.
- Снизить или отправить в probation `LiveRegimePullback`, `LiveCrashHunter`, `LiveTrendFollow`.
- Не промотировать genetics-агентов по unrealized PnL без закрытых сделок.

BITGET:

- Увеличить вес `LiveAfterShock`, `LiveOIBreakout`, `CarryFlowAgentV2`.
- Оставить `PlayerFunding`, `FundingArb`, `TrendFollow`, `MomentumScalper` в quarantine/purgatory до восстановления.
- Больше доверять `PanteonDefensiveResearch`, `PanteonNextResearch`, `PanteonTrendResearch`, `PanteonConsensusResearch`, но проверять их сигналы через live execution filters.

### P2: улучшить метрики и наблюдаемость

1. В `recent_trades` сейчас `fee=0.0`. Для реальной оценки нужны комиссии и funding из биржи.
2. Разделить warning-уровни:
   - `WARNING` для технических проблем;
   - `INFO` или отдельный `RISK` для штатных SL/TP/TRAIL.
3. В dashboard добавить блок `data health`: возраст snapshot, число ошибок API за 10 минут, число reconcile-add/remove, число failed orders.
4. Для каждого real signal сохранять `selected_player`, `selected_agents`, `order_result`, `order_id`, `position_source`.

## Итог

MEXC можно считать условно штатным: есть сетевые сбои, но PnL и drawdown контролируемые. Главная правка для MEXC - защитить reconcile от ложного удаления/переобнаружения позиций.

BITGET требует вмешательства: процесс живой, но real PnL отрицательный, много SL, есть failed orders и свежий кейс premature close после сбоя API. До исправления `snapshot/reconcile/stale` и снижения риска BITGET лучше держать в более консервативном режиме.
