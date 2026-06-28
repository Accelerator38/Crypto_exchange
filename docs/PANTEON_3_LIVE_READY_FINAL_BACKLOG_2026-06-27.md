# Panteon 3.0: финальный live-ready backlog

Дата: 2026-06-27

Это список оставшихся доработок перед расширением или перезапуском реальных ордеров. Он намеренно строгий: текущая replay matrix все еще не проходит promotion gates и не доказывает положительное live expectancy.

## Текущие доказательства

- Последняя replay matrix: `Reports/Panteon3PreLiveMatrix/codex_2026_06_20_26_dynamic_best`.
- Данные: восстановленный публичный Binance hourly spot за 2026-06-20..2026-06-26.
- Результат candidate: 2 filled signals, 1 closed trade, realized PnL -0.1990 USD.
- Promotion verdict: failed.
- Лучший standalone benchmark component в этом окне: `LiveVolCompress`, +1.7087 USD, 5 closed trades, 80% win rate.
- Flash parity diagnosis: `LiveVolCompress` почти всегда `HOLD/inactive`; actionable rows в основном `no_evidence` или `insufficient_closed_trades` до убыточной controlled-exploration сделки.

## P0: обязательно до расширения real-order live

1. Заменить spot-only replay validation на MEXC/BITGET futures replay data.
   - Нужны exchange-specific futures candles, fees, funding, min-notional, precision, spread/slippage.
   - Текущий Binance hourly dataset оставить только как smoke test, не как доказательство edge.

2. Построить causal component-memory feed для Flash routing.
   - `component_benchmark_report` сейчас post-run evidence; его нельзя использовать как ex-post oracle.
   - Нужно сохранять prior-bar component stats по actor/symbol/regime/action.
   - Routing должен использовать только статистику, доступную до текущего бара.

3. Починить inactive specialist agents на источнике сигнала.
   - `CarryFlowAgentV2`: почти всегда `HOLD/inactive`; проверить funding/carry inputs и feature requirements.
   - `MomentumScalper`: почти всегда `HOLD/inactive`, иногда insufficient sample.
   - `LiveCrashHunter`: редкий по природе crash specialist; не использовать как источник базовой частоты.
   - `LiveVolCompress`: benchmark положительный, но actionable Flash rows редкие; нужно связать его standalone signal stream с Flash candidate evidence.

4. Добавить live shadow canary gates перед любым увеличением размера.
   - Требовать nonzero signals/orders/fills в shadow/paper.
   - Требовать positive rolling expectancy после fees/funding/slippage.
   - Требовать прохождение max daily loss, max open exposure, max trades/day и exchange reconcile.

5. Держать минимальный real-order risk cap.
   - Dynamic promotion и controlled exploration должны оставаться behind flags.
   - Размер не увеличивать до минимум 20 filled и 10 closed real/paper trades на биржу с положительным expectancy.

## P1: нужно для повышения частоты торгов

1. Dynamic best-component promotion должен использовать causal memory, а не current-run benchmark summaries.
2. Расширить compact Flash audit include labels для всех расследуемых candidate components.
3. Вывести per-bar `why_not_promoted` в обычные reports, а не только в diagnostic scripts.
4. Разделить diagnostics по exchange, symbol, regime, action, owner и source stream.
5. Добавить daily no-trade funnel thresholds: run должен падать, если actionable candidates есть, но все исчезают из compact audit или превращаются в HOLD.

## P2: cleanup перед длинными live runs

1. Архивировать старые reports/results, которые не используются как baselines.
2. Оставить только задокументированные launch scripts для MEXC/BITGET Panteon 3.0 modes.
3. Документировать точные live commands и flags, разрешенные для real orders.
4. Добавить pre-flight command, который запрещает live, если latest matrix/paper verdict устарел или failed.

## Текущее решение

Не расширять live size сейчас. Следующая цель реализации: causal component-memory routing плюс futures replay data. Только после этого имеет смысл перезапускать или увеличивать MEXC/BITGET real-order canary.
