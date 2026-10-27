# Проверка скачка цен и исполнения — 8 октября 2026

Сверены 57 600 локальных минут: OHLC всех 240 четырёхчасовых свечей воспроизведены точно; пропусков/дубликатов нет. Склейка источников перед 18 августа проверена.
Повторный публичный Bitget подтвердил 0/8 активов; независимый Binance — 8/8 в пределах зарегистрированного диагностического критерия (difference return <100 bps, price basis <200 bps). Это проверка наличия крупного движения, не тождественности цен бирж.

## Скачок

| Актив | Локальный open/open 3d | Bitget return diff, bps | Binance return diff, bps |
|---|---:|---:|---:|
| ADA | 39.35% | нет подтверждения | 7.963 |
| BNB | 19.36% | нет подтверждения | -2.293 |
| BTC | 21.89% | нет подтверждения | 3.919 |
| DOGE | 36.59% | нет подтверждения | 3.377 |
| ETH | 31.53% | нет подтверждения | 2.674 |
| LINK | 27.69% | нет подтверждения | -3.950 |
| SOL | 25.76% | нет подтверждения | 4.410 |
| XRP | 56.38% | нет подтверждения | 3.565 |

## Реалистичность исполнения

OHLCV open — цена первой сделки бара, не гарантированная цена IOC-ордера, bid/ask или доступная ликвидность. Наличие такого движения не подтверждает исполнимость полученной прибыли.
Decision→entry в текущем benchmark — 4 часа, не сразу после закрытия. Labels/replay согласованы между собой, но это иной deployment clock, чем исполнение сразу после закрытия.
Модель исполнения не округляет quantity/price по историческим grids, не проверяет min notional, не моделирует partial fill, reject, глубину/участие, latency и рыночное воздействие.
Публичные текущие contract rules проверены только диагностически; они не являются исторической спецификацией августа–сентября или account fee evidence.
Normal/slippage 2 bps и stress 8 bps — assumptions, не измеренные из стакана данного периода. Полноценная квалификация требует исторического L2/trades/mark с event/receive-time и общего IOC/liquidity adapter.
DD по 4h trade-open NAV не равен непрерывной просадке по mark; budgets 10/20/30% — research gates, не stop-loss и не real-money authorization.

## Диагностика quantity

| Модель | Fills | Off current grid | Below current min | Decision→entry |
|---|---:|---:|---:|---:|
| linear6_uniform | 82 | 82 | 0 | [14400000] ms |
| linear6_half_life_90_anchor20 | 34 | 34 | 0 | [14400000] ms |
| mlp8_uniform | 56 | 56 | 0 | [14400000] ms |
| mlp8_half_life_90_anchor20 | 52 | 52 | 0 | [14400000] ms |
| EMA48_long_only | 62 | 62 | 0 | [14400000] ms |

## Вывод и следующие действия

Сначала отделить источник движения от модели исполнения. Даже подтверждённый скачок остаётся концентрированным единичным событием; исходные отрицательные audit-LCB не изменяются.
Не удалять это окно и не перенастраивать модель на нём. Следующий engineering change — общий исторический IOC-adapter с quantization, latency, bounded depth, reject/partial fill, signed funding и observed mark. Запуск финансового пересчёта после отдельной регистрации сценариев, не автоматически из этой проверки.
Публичных запросов: 20; retries=0; credentials/orders/WS/runtime=0; fits/replays=0. Все старые модели, ledger, hashes и risk authority сохранены.

Документация: [официальный источник 1](https://www.bitget.com/docs/catalog/market/market-data), [официальный источник 2](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data), [официальный источник 3](https://www.bitget.com/docs/catalog/classic-contract-market/classic-contract-market)
