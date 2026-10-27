# Финальная проверка скачка и исполнения — 8 октября 2026

## Результат

Крупный рост 19–22 августа подтверждён по всем восьми монетам свежими историческими данными Bitget и независимым Binance. Признаков ошибки агрегации/склейки в данном окне не обнаружено.
Это устраняет подозрение в искусственном ценовом скачке, но не устраняет концентрацию прибыли в одном событии и не превращает отрицательные audit-LCB в положительные.

Локально: 57 600 минут, 240 свечей 4h, точное совпадение всех OHLC при новой агрегации; объёмы согласованы, пропусков/дубликатов нет.
Bitget: 29 общих свечей на каждую монету, все OHLC совпадают точно. Полная 19-свечная сетка event entry 19.08 04:00 → exit 22.08 04:00 покрыта, 3d returns совпадают.
Binance: полная 30-свечная пятидневная сетка; на всех восьми активах крупное движение подтверждено. Межбиржевые цены не используются как исполнимые цены Bitget.
Первоначальный audit_results.json оставлен неизменным: он требовал полную пятидневную сетку Bitget и показал 0/8 из-за граничного бара. При start/end на -1ms Bitget вернул окно 17.08 20:00 → 22.08 16:00. Это несовпадение границ ответа, не цен. Уточнение использует только сохранённые ответы, без повторных GET.

| Актив | Bitget 3d рост | Binance 3d рост | Разница outcome, bps | Max open basis, bps |
|---|---:|---:|---:|---:|
| ADA | 39.349% | 39.429% | 7.963 | 17.301 |
| BNB | 19.361% | 19.338% | -2.293 | 16.769 |
| BTC | 21.889% | 21.928% | 3.919 | 3.022 |
| DOGE | 36.592% | 36.626% | 3.377 | 11.130 |
| ETH | 31.532% | 31.559% | 2.674 | 2.347 |
| LINK | 27.695% | 27.655% | -3.950 | 6.890 |
| SOL | 25.762% | 25.806% | 4.410 | 4.547 |
| XRP | 56.381% | 56.417% | 3.565 | 5.377 |

## Funding mark proxy

Доступные исторические 4h mark candles BTC/ETH/SOL согласованы со trade-open. Максимальный абсолютный basis на funding timestamps:
- BTC: 1.3722 bps.
- ETH: 1.8495 bps.
- SOL: 0.7802 bps.

Это узкая диагностика 18–22 августа по трём активам, не проверка всего 61-дневного funding ledger, не tick-level mark и не фактические исполнения. Разница mark/trade не доказывает допустимое slippage.

## Почему исполнение ещё не реалистично квалифицировано

1. Decision→entry в текущем benchmark равен 4 часам. Пользовательский желаемый режим «сразу после закрытия» требует нового clock: первый доступный quote после закрытия + зарегистрированная latency, без чтения цен до доступности сигнала. Нельзя незаметно заменить entry и сохранить старые labels/результаты.
2. Все 82 reference fills победившего linear uniform имеют дробные quantity вне текущего size grid Bitget. Они не готовы для отправки в API без quantization. Минимальные размеры по текущим rules проходят, но сами rules не доказывают исторические ограничения августа. Floor-dust не является готовой оценкой net PnL.
3. Комиссия 6 bps совпадает с публичным текущим default takerFeeRate всех восьми контрактов. Account/VIP/акционные и исторические ставки не подтверждены; доступ к аккаунту или ключам не использовался.
4. Slippage 2/8 bps не калиброван на L2/trades периода. OHLCV open не гарантирует доступный ask, глубину, размер IOC, общий participation budget или очередь. Partial fill, reject, limit price, minimum notional/qty/price ticks и реальная latency не смоделированы.
5. Исследовательский DD основан на 4h trade-open NAV, не на непрерывном observed mark. 9.31% не доказывает соблюдение live drawdown budget 10%.
6. Существующий release contract относится к другой старой политике/модели, требует observed L2/mark и запрещает OHLCV fill fallback. Компактная модель текущим replay не квалифицирована к этому контракту; runtime не менялся.

## Дальше

Первый engineering этап: зарегистрировать clock после закрытия, history/version instrument rules, qty/price quantization, min notional и IOC state machine (reject/partial/timeout/closed position), затем проверить на deterministic offline fixtures.
Для полноценного финансового пересчёта на августе–сентябре нужны L2/trades и mark для нужного периода, event/receive-time и fee evidence либо явно зарегистрированные conservative assumptions. Без этих данных не называть расчёт реалистичным и не открывать live/paper.
Запросов по зарегистрированному плану 20, все публичные, последовательные, без retries. При уточнении 0 новых GET; 0 fits/financial replays. Исходные результаты и safety flags сохранены.

Округление временных границ описано в [Bitget market docs](https://www.bitget.com/docs/catalog/market/market-data). Исторические mark candles и поля contract fee/size описаны в [Bitget classic market docs](https://www.bitget.com/docs/catalog/classic-contract-market/classic-contract-market). Независимые candles получены из [Binance futures REST](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data).
