# План выхода к живым торгам Bitget

Прибыль не может быть гарантирована. Каждый этап ниже имеет бинарный gate; при
непрохождении работа возвращается к данным или модели, а риск не повышается.

## 1. Заморозить контракт

- Не добавлять новые decision-слои до завершения первого canary.
- Единственный runtime: `pantheon_players_v1`.
- Единственные режимы: `multi` и `singleton`.
- В live допускается один real-игрок; остальные только shadow.

Статус: **выполнено**. Последний полный прогон после изоляции snapshot,
исправления preflight, player-only dashboard и public-only bridge: 1860 тестов
и 17 subtests.

## 2. Собрать Bitget-историю

- Получить закрытые OHLCV, funding и доступный derivatives context для
  согласованного списка USDT futures.
- На каждом запуске получать contract config биржи: minimum amount, precision,
  size multiplier и статус символа нельзя хардкодить.
- Хранить источник, время получения и контрольную сумму каждого набора.
- Разделить данные по времени на train/warm-up, validation и полностью
  отложенный final sanity период.

Статус OHLCV: **выполнено**. 2022-01-01 — 2026-07-14, 8 символов,
317 952 строки; дыр, дубликатов и off-grid точек нет. Carry-flow/funding/OI tape
продолжает накапливаться отдельно и не нужен для текущего singleton.

## 3. Walk-forward multi

Запустить `tools/run_player_efficiency_retro.py` на последовательных временных
окнах с фактическими комиссиями, slippage stress и минимальными объёмами Bitget.
Сравнивать:

- `multi` против `NoTrade` и каждого singleton;
- результат по каждому режиму, символу и месяцу;
- expectancy после costs, max drawdown, turnover и стабильность выбора;
- base costs и стресс-сценарии с увеличенным slippage/fees.

Gate: положительная out-of-sample expectancy, приемлемая просадка, отсутствие
зависимости результата от одного символа/месяца и достаточная выборка. Конкретные
пороги фиксируются до запуска, а не подгоняются после результата.

Статус: `multi` — **FAIL** (`-46.72 USD`, 2/5 положительных окон).
`singleton(ResearchValidatorAgent)` — **PASS** (`+153.27 USD`, expectancy
`+1.31 USD/сделку`, DD `4.25%`, 4/5 положительных окон, +5 bps stress
`+141.50 USD`). В demo допускается только прошедший singleton.

## 4. Bitget demo/paper canary

- Использовать отдельный Demo API key и demo header/endpoint.
- Проверить client order id, idempotency, reconnect, REST reconciliation,
  reduce-only close, one-way/hedge mode и восстановление после рестарта.
- Дождаться warm-up player×regime памяти; `NoTrade` во время warm-up нормален.
- Canary artifact обязан подтвердить player-only contract, shadow игроков,
  ненулевые signals/orders/fills, положительную expectancy после costs и нулевые
  позиции на завершении.

Gate: два последовательных 30-дневных canary-периода (или дольше до суммарных
6 закрытий) без desync, orphan orders, повторных открытий и нарушений ownership;
ledger полностью совпадает с Bitget, завершение с нулевой позицией. Первый
период проверяет механику, второй — воспроизводимость. Статус: **в работе**.

Технический статус: `demo_futures` реализован fail-closed и покрыт тестами.
2026-07-15 live credentials успешно проверены read-only: equity
`39.51944491 USDT`, открытых позиций `0`, отправленных ордеров `0`. На том же
live market data запущен безопасный `paper_live_feed` canary с
`singleton(ResearchValidatorAgent)`, капиталом `39.51944491 USDT` и
`FakeExchange`: контрольный run `2026-07-15_19-35-12_v2` перешёл в
`running/active`, а реальная биржа в исполнении не участвует. После аудита
bridge активный canary перезапущен как `2026-07-15_19-49-47_v2`: live-feed
bridge получает только публичные рыночные данные и вообще не получает API
credentials; live-ключи используются отдельным read-only account probe.

Этот запуск проверяет feed, player-only выбор, shadow и внутренний real-маршрут,
но не заменяет биржевой Demo canary: в `.env.local` по-прежнему отсутствуют
отдельные Demo credentials. Live-ключи не используются как fallback для Demo.
После создания Demo API key добавить в `.env.local`:

```dotenv
BITGET_TRADING_MODE=demo_futures
BITGET_DEMO_API_KEY=<demo key>
BITGET_DEMO_SECRET_KEY=<demo secret>
BITGET_DEMO_PASSPHRASE=<demo passphrase>
```

Сначала выполнить read-only probe; он не содержит order-вызовов:

```powershell
python tools/check_bitget_demo_access.py
```

## 5. Micro-live

- Новый API key без withdrawal permission, с IP allowlist.
- Isolated margin, один символ/одна позиция на старте, минимально исполнимый
  notional, биржевой stop и жёсткий дневной loss kill-switch.
- Перед каждым order получать актуальные contract constraints; всегда задавать
  уникальный `clientOid`, а закрытие делать reduce-only там, где это требует
  выбранный position mode.
- Автоматическое повышение капитала запрещено.

Gate: заранее заданное число закрытых сделок с положительной live expectancy,
совпадение ledger с биржей и просадка внутри бюджета.

## 6. Постепенное масштабирование

Повышать риск только одной ступенью после независимого review. При ухудшении
expectancy, росте slippage, desync или нарушении canary-гейтов возвращаться на
предыдущую ступень. Добавление нового игрока сначала проходит retro, затем
shadow, demo и только потом micro-live.

## Команды

```powershell
# Проверка Demo-плана запуска без ордеров
$env:BITGET_TRADING_MODE = "demo_futures"
python Start_panteon.py --only BITGET --dry-run `
  --trade-regime "singleton(ResearchValidatorAgent)"

# Player-only retro
python tools/run_player_efficiency_retro.py `
  --data-dir Retrodate\bitget_futures_history_v3_2022_20260714 `
  --years 2022,2023,2024,2025,2026

# Фиксированный baseline
python tools/run_player_efficiency_retro.py --mode singleton `
  --fixed-player ResearchValidatorAgent --shadow-scope fixed `
  --data-dir Retrodate\bitget_futures_history_v3_2022_20260714 `
  --years 2022,2023,2024,2025,2026
```

Bitget указывает, что параметры futures orders должны соответствовать
`minOrderAmount`, `sizeMultiplier` и precision, рекомендует передавать
`clientOid`, а ограничения контракта получать через Contract Config. Для demo
API нужен отдельный Demo key и специальный demo-заголовок. Документация:

- https://www.bitget.com/api-doc/uta/trade/Place-Order
- https://www.bitget.com/api-doc/contract/market/Get-All-Symbols-Contracts
- https://www.bitget.com/api-doc/uta/guide
