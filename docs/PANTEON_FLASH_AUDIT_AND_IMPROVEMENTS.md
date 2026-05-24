# Panteon Flash: аудит кода, баги и план улучшений

Дата: 2026-05-21
Изученные модули: `src/panteon_v2/selection/{flash_allocator,voting,player,promotion_manifest,agent,scoring,strategist}.py`, `src/panteon_v2/app/{main_loop,agent_bootstrap}.py`, `src/panteon_v2/domain/types.py`.

Документ построен в три блока:
1. найденные баги и архитектурные дефекты с точными ссылками на файл/строки;
2. предложения по улучшению агентов, игроков и Пантеона;
3. конкретный план, как заставить Пантеон стабильно торговать не хуже своих составляющих.

---

## 1. Баги и архитектурные дефекты

### 1.1. Ranking использует `score`, а гейт `min_score_to_trade` использует `base_score`

`flash_allocator.py:311-314,399-401`. Когда `shadow_confirmation_enabled=True`, переменная `score` перезаписывается на `shadow.score`, и затем к ней прибавляется `actionable_bonus`. Но порог `min_score_to_trade` сравнивается с `base_score` (исторический production score), а не с финальным `score`, который сортирует кандидатов. Это рассинхронизация двух «правд»: ранжирование идёт по shadow, а гейт допуска — по production. На audit-выходе reason будет `score_below_threshold`, но фактически отброс произошёл по совершенно другой шкале. Поведение неинтуитивно, отлаживать тяжело.

Фикс: либо использовать единый `effective_score = blend(base, shadow) + bonuses`, либо в audit писать обе величины (`base_score`, `effective_score`, `gate_score`).

### 1.2. `regime` фактически не используется в `regime_score`

`scoring/scoring.py:99-161`. Аргумент `regime` принимается, но в формуле не появляется ни разу — учитывается только `metrics`, которые уже per-regime. Имя обещает «учёт режима», а функция этого не делает. Если завтра кто-то решит дать BULLISH меньший pnl_weight чем CRASH — он подумает, что это уже работает.

Фикс: либо удалить параметр, либо реализовать regime-specific конфиг через словарь `Dict[Regime, ScoringConfig]`.

### 1.3. `MarketSnapshot.regime_confidence` игнорируется скорингом

`domain/types.py:174`, `scoring/scoring.py`. Поле есть, валидируется в `[0,1]`, но ни один decision-слой его не читает. При `regime_confidence=0.3` Flash вынесет такое же решение, как при `1.0`. Это потерянный сигнал и потенциально источник убытков на переходах режимов.

Фикс: умножить score на `regime_confidence`, либо повысить `min_score_to_trade` пропорционально `1 / regime_confidence`.

### 1.4. `_component_score` усредняет `pnl_pct` без веса по выборке

`flash_allocator.py:578-602`. Когда ensemble не имеет своих данных, его score собирается из компонентов: `pnl = sum(pnl_pct) / len(scored)`. Это арифметическое среднее: агент с 200 закрытыми сделками и 1% PnL и агент с 3 закрытыми сделками и 50% PnL получают одинаковый вклад. Маленькая выборка с шумом доминирует. При этом `closed_trades` суммируются — то есть `Metrics` получается несамосогласованным: `pnl_pct` усреднён, а `closed_trades` суммированы.

Фикс: использовать weighted average по `closed_trades` либо явно по trade_pnl суммам.

### 1.5. `Inactivity penalty` срабатывает при наличии closed_trades

`scoring/scoring.py:158-159`. Условие `signals == 0 and entries == 0`. Если у агента в текущем баре `entries=0`, `signals=0`, но `closed_trades=5` (всё закрылось), штраф −`inactivity_penalty=0.65` уезжает в score. Это сваливает score прибыльного, но прошедшего без новых entries актора, что портит ранжирование.

Фикс: учитывать `closed_trades` в условии, либо применять штраф к свежей активности (`recent_signals_window`).

### 1.6. WeightedConsensus: close-канал не симметричен open-каналу

`voting.py:142-167`. Для opens используется direction `net = long - short`. Для close — `score_close >= close_thr` без вычитания «голосов против close». Любой close-вес выше порога закроет позицию, даже если 80% весов хотят держать. Это «асимметричная» политика, и она агрессивно закрывает.

Фикс: ввести `close_net = score_close - score_keep` либо явно требовать `close_share = score_close / total_active_weight >= threshold`.

### 1.7. Tie-breaker в `FlashAllocator` смещён к `agent` против `ensemble`

`flash_allocator.py:427-435`. Сортировка кандидатов:
```
(rejected, -score, -closed_trades, actor_type, label)
```
`actor_type` — строка, `"agent" < "ensemble" < "no_trade"`. При полностью равных score и closed_trades агент всегда побеждает ensemble. Это не нейтральный выбор, особенно когда ensemble намеренно собран как стабилизатор шума.

Фикс: использовать явный иерархический tiebreak (например, по `signals` или по `pnl_per_trade`), либо random tie-break на seed от bar.

### 1.8. `actor_signal_cap` режет по алфавиту символа, а не по качеству сигнала

`flash_allocator.py:529-567`. Цикл идёт по `decisions` в порядке `sorted(symbols)` (заданном в `decide`). Если actor вышел победителем на BTC/USDT и AVAX/USDT с разными score, при `cap=1` режется тот, кто пришёл «позже», то есть BTC (по алфавиту BTC > AVAX). Лучше резать ту, у которой score ниже.

Фикс: сортировать decisions внутри cap-цикла по `(actor, -score)` и обрезать с конца.

### 1.9. `actionable_bonus` не нормирован

`flash_allocator.py:313-314`. `+0.25` к score. Когда production score кандидатов имеет дисперсию порядка единиц — это шум; когда score близок к 0 — это доминирующий вклад. По сути bonus меняет ранжирование непредсказуемо. Это нелинейный «магический» множитель.

Фикс: либо нормировать bonus к стандартному отклонению score по пулу кандидатов, либо превратить в мультипликативный (`score *= 1 + actionable_bonus`).

### 1.10. `NoTradePlayer` бесполезен как кандидат

`player.py:67-88`, `flash_allocator.py:691-723`. `NoTradePlayer.vote` всегда возвращает `[]`. В Flash `_player_outputs` создаёт rows с `Action.HOLD` для всех символов. В `_decide_symbol` row с `is_hold` маркируется rejected с reason="inactive". То есть NoTrade на уровне игрока никогда не попадает в audit как реальный кандидат — он отбрасывается тем же фильтром, что и любой бездействующий агент. Декоративная сущность.

Если NoTrade задумывался как «контрольная альтернатива» (раздел 7.2 документации), его нужно сделать первоклассным кандидатом с собственным score (например, `expected_pnl_no_trade = 0 - estimated_slippage * avg_trade_size`). Сейчас «no_trade as decision» — это просто `selected is None`.

### 1.11. `EnsemblePlayer.last_vote_errors` мутирует не-frozen dataclass

`player.py:91-115,159`. Поле перезаписывается в `vote()`. Player перестаёт быть pure. Если `vote()` вызывается из нескольких потоков (например, real + shadow на одном экземпляре), `last_vote_errors` гонка.

Фикс: возвращать errors отдельно (`vote → (signals, errors)`), либо хранить ошибки в caller-овом контексте, а не в Player.

### 1.12. Overextension guard — глобальные пороги для всех монет

`flash_allocator.py:508-527`, defaults `-8%/+8%`. Для BTC ±8% за 12 баров — экстрим (вероятность 1-2% в моменте), для альтов — обычный шум. Защита либо не сработает на «спокойных» монетах, либо слишком часто будет давить «волатильные».

Фикс: ATR-нормированные пороги (`return / ATR_lookback >= z_threshold`), либо per-symbol калибровка.

### 1.13. `Signal.price` форс-перезаписывается из `market.prices.get(symbol, 0.0)`

`flash_allocator.py:451-460`. Если `market.prices` не содержит symbol, `.get(..., 0.0)` вернёт 0.0, `Signal.__post_init__` пропустит (валидация `price >= 0`), но 0-цена приведёт к делению на ноль в risk_limits/position_tracker. Молчаливая ошибка.

Фикс: явная проверка `market.has_price(symbol)`; иначе пропускать решение с reason="missing_price".

### 1.14. `promotion_manifest_enabled` проверяет gate только для `is_open`

`flash_allocator.py:392-398`. Close-сигналы пропускаются — это логично (нельзя оставить открытую позицию). Но при включении manifest позиции, открытые до promotion, попадут под close через закрытие сигнала, которое имеет другой signal_key (close action). Накопленная статистика close-сигналов часто меньше open-сигналов, что искажает promotion для close. Лучше явно отделять gating логики open/close.

### 1.15. `max_signals_per_actor` молчаливо подавляет лучших

`flash_allocator.py:555-563`. Когда cap исчерпан, кандидат заменяется на NoTrade с reason="actor_signal_cap". В audit `candidates` остаётся прежний (то есть выбран actor), а в `signal` — None. То есть `selected_actor` рассинхронизирован с `signal`: видно «выбран X», но сигнала нет. Отладка путается.

Фикс: после обрезки полностью обновлять FlashDecision (selected_actor="NoTrade", reason="actor_signal_cap"), уже сделано в `replace(...)` — хорошо. Но при этом теряется audit кандидата (он стирается). Лучше сохранять и `original_selected_actor` для трассировки.

### 1.16. Shadow score кладётся с разными ключами одновременно

`main_loop.py:1191-1259`. В `_flash_shadow_confirmation_scores` для каждой пары `(label, symbol)` записываются и кортеж `(label, symbol)`, и `(label, symbol, action_name)`. На стороне FlashAllocator `_shadow_confirmation_for` ищет по приоритету `(label,symbol,action) → (label,symbol) → ...`. Если scorer не сумел отдать stats для конкретного action и вернул `None`, мы перезапишем `(label, symbol)` нулевыми данными при следующей итерации? Сейчас в коде нет — `scores[(label, symbol)]` записывается один раз, потом overwrite не происходит. Но при ошибке в одном scorer и обновлении в другом семантика «какой ключ ответил» нестабильна.

### 1.17. `regime_score` теряет монотонность в `pnl_pct` при `closed_trades < 2`

`scoring.py:131-133`. При <2 closed возвращается `pnl_pct * 0.40 * confidence`. `confidence` при closed=0 равен 0 → score = 0 независимо от pnl. То есть для одного закрытого с PnL=+5%, confidence(1)=1/6 ≈ 0.167, score ≈ 0.33. Для нуля закрытых и тех же signal-данных — score = 0. Это поведение мешает promoted/blocked logic, особенно когда «новый» агент имеет 1-2 сделки с большим выигрышем.

### 1.18. Solo-safety проверка может выкинуть рабочий agent из-за shadow-only состояния

`main_loop.py:1083-1091`. Каждый агент перед попаданием в Flash оборачивается в `Solo_<label>` и проверяется fallback_safety. Если safety drop срабатывает для Solo-обёртки — агент удаляется из пула. Но тот же агент может быть «здоров» в составе ensemble. Удаление по wrapper-уровню убирает участие в любых ensemble.

Фикс: safety check должен быть атрибутом агента/ensemble, а не wrapper-а.

### 1.19. Скоринг не учитывает торговые издержки

`scoring.py`. `pnl_pct` берётся как есть, без вычета комиссий/проскальзывания/funding. Агент с высоким turnover и нулевым net-edge выглядит положительно gross. На реальной бирже его доходность станет отрицательной первой.

Фикс: PerformanceMemory должен возвращать `pnl_net_pct` и `pnl_gross_pct`, scoring — использовать net.

### 1.20. Отсутствует penalty за churn (частая смена выбранного актора)

`flash_allocator.py`. Каждый бар — независимое решение. Если на t score-разрыв 0.01 в пользу A, на t+1 в пользу B, на t+2 снова A — система будет дёргать execution layer (двойные fees, налог на «вход и выход»). Это убыточно даже при правильном «long-term winner».

Фикс: ввести `selection_inertia` — bonus к ранее выбранному актору, либо требовать `score(new) - score(prev) >= switch_margin`.

### 1.21. AgentRegistry не thread-safe

`agent.py:49-103`. `_agents: Dict` мутируется в `register/unregister`, читается в `decide`. Если bootstrap/promotion работают параллельно с main loop — гонка. На реальном live trading это тихие баги.

### 1.22. `ActionFilterAgent` без allowed_regimes возвращает оригинал, но при пустом allowed_actions пропускает все

`agent_bootstrap.py:236`. `if allowed_actions and action not in allowed_actions: continue` — при пустом `allowed_actions` фильтр отключён. Это документировано (см. `ResearchValidatorNeutralOnly` с `actions=()`), но скрыто. Легко перепутать «разрешить все» и «не разрешать ничего».

### 1.23. Promotion manifest требует жёсткие фиксированные пороги

`promotion_manifest.py:11-19`. `min_full_closed_trades=50`, `min_win_rate_pct=52.0`. При 50 закрытых сделках доверительный интервал на win_rate очень широкий. Промоушн по таким порогам — bias к «случайно повезло». Никакой статистики значимости.

Фикс: Bayesian промоушн с posterior-распределением edge и нижней границей доверительного интервала.

### 1.24. main_loop.py — 3657 строк, тесты практически невозможны

`app/main_loop.py`. Монолит, сильное coupling. Любая правка одного фильтра требует ревизии всех 4 веток (real/shadow/fallback/flash). Это инженерный долг, который непрямо превратится в торговый убыток.

### 1.25. Voting policies без teста на одинаковое поведение под одинаковые votes

`voting.py`. Нет инварианта `for votes V, weights W, equal_threshold: WeightedConsensus == RiskParity(equal vols)`. Сейчас разные политики могут давать разные результаты под одинаковыми entries, что усложняет shadow/real comparison.

---

## 2. Улучшения по уровням

### 2.1. Агенты

1. **Декларативная спецификация edge**. Каждый агент должен публиковать `expected_edge_signature`: список `(regime, symbol_class, action_class, side)`. Это позволяет Пантеону отбирать кандидатов до выполнения `act()` и сильно уменьшает шум.
2. **Net-PnL внутри агента**. Агент возвращает не только action, но и `expected_cost_bps` (gross-to-net скидка). Score должен сравнивать net edges.
3. **Confidence per signal**. Расширить ответ агента: вместо `Dict[symbol, Action]` — `Dict[symbol, AgentVote(action, confidence, expected_holding_bars)]`. Это даёт voting policy явный материал для взвешивания и Flash — материал для score.
4. **ATR/vol normalisation**. Все агенты должны принимать `MarketSnapshot` с расширенными метриками: `ATR_14, realized_vol, current_funding_z`. Не дублировать вычисления внутри агентов.
5. **Изоляция через `ActionFilterAgent`**. Создавать controlled variants для каждого крупного семейства: `*_LongOnly`, `*_ShortOnly`, `*_NeutralOnly`, `*_HighLiquidityOnly`. Это уже частично сделано; распространить на все агенты.
6. **Walk-forward calibration**. Гиперпараметры агентов (пороги breakout, длина окна) — не «прибиты», а калибруются на скользящем окне.

### 2.2. Игроки

1. **WeightedConsensus → закрыть close-канал асимметрию**. См. п. 1.6.
2. **Заменить EnsemblePlayer на «soft voting»**. Сейчас агент даёт `Action`. Если расширить до `AgentVote(action, confidence)`, ensemble может собирать «softmax over actions» и брать argmax с уверенностью — это математически идентично logistic regression stacking.
3. **Динамические веса**. `EnsemblePlayer.weights` сейчас frozen. Сделать `weights` функцией от текущего режима и недавнего PnL каждого компонента (`weights = softmax(recent_signal_key_pnl)`).
4. **RotatingAgentPlayer + fallback**. Сейчас порядок прибит к regime. Добавить watchdog: если выбранный агент даёт `is_hold` N подряд баров, попробовать следующий из listа.
5. **«Anti-ensemble»**. Игрок, который выводит действие только когда `set(votes) == {Action.HOLD, X}` (то есть один агент кричит, все молчат). Это полезный edge для редких событий.
6. **NoTradePlayer как первоклассный кандидат**. Дать ему синтетический score = `expected_cost_savings - opportunity_cost`. Тогда «не торговать» — это решение, которое можно ранжировать рядом с реальными.

### 2.3. Пантеон / FlashAllocator

1. **Bayesian rolling score per signal key**. Хранить `(alpha, beta)` Beta-priors для win/loss и нормальные posterior для PnL. Решение использовать нижнюю границу доверительного интервала (LCB).
2. **Thompson sampling**. На каждом баре сэмплировать из posterior каждого кандидата и выбирать максимум. Это даёт regularized exploration: новые кандидаты получают шанс, но не за счёт денег.
3. **Hysteresis / switch_margin**. Не менять выбранного актора, пока новый не лучше старого на `switch_margin`. См. п. 1.20.
4. **Concentration penalty**. Снижать score кандидата пропорционально Herfindahl index уже выбранных акторов на этом баре. Это distribute риск по нескольким источникам, даже если один formally лучший.
5. **Cost-aware ranking**. Score = `pnl_net - λ * gross_volume`. λ калибруется по средним fee/slippage на бирже.
6. **Кэш `regime_score`**. Сейчас вычисляется на каждый бар × кандидат × symbol. Дёшево, но при тысячах signal keys — заметно. Memoize по `(label, regime, perf_version)`.
7. **Audit с двумя score**. Записывать `base_score`, `shadow_score`, `effective_score`, `gate_score`. Тогда reason="score_below_threshold" станет интерпретируемым.
8. **Promotion как posterior gate**. См. п. 1.23. Промоушн только при `P(edge > 0 | data) >= 0.9`.
9. **Per-symbol calibrated overextension**. См. п. 1.12. Хранить `recent_ATR[symbol]` в market snapshot.
10. **Кросс-валидация promotion**. Promotion manifest должен делиться на «in-sample фитнес» и «out-of-sample гейт» — те же 50 closed trades должны быть на holdout-периоде.
11. **Decision diff log**. Каждое отличие выбранного актора от предыдущего бара логируется с причиной (margin, urgency, hysteresis trigger). Это резко упрощает диагностику.

---

## 3. Как сделать Пантеон не хуже своих составляющих

### 3.1. Почему Пантеон чаще хуже своих составляющих

Greedy «выбрать лучшего по score» имеет три фундаментальные проблемы:
1. **Selection bias**. Лучший на in-sample становится средним на out-of-sample (winner's curse).
2. **Late switching**. Score основан на rolling статистике — он реагирует на деградацию с задержкой. К моменту, когда score откатывает agent A, A уже потерял на out-of-sample.
3. **Кост-неаккуратность**. Каждая смена актора — это дополнительные сделки. На длительной серии churn съедает edge.

Соответственно, любой Пантеон, который не управляет этими тремя факторами, будет в среднем проигрывать своему лучшему компоненту.

### 3.2. Минимальный набор изменений, после которых Пантеон догонит лучших

1. **Lower confidence bound вместо greedy maximum**. Вместо `argmax(score)` выбирать `argmax(LCB(score, n))`. LCB = mean − k·std/√n. При k=2 — пессимистичный отбор, при k=0 — greedy. Это снимает selection bias.
2. **Switch margin / hysteresis**. См. п. 1.20. Эмпирическое значение: `0.5 * median_intraday_score_volatility`.
3. **Cost-aware net score**. См. п. 1.19. Без этого все улучшения подавляются комиссиями.
4. **Sticky bonus уже выбранному**. Простой механизм: `score(prev_winner) += sticky_bonus`. Эквивалентно switch margin, но проще встраивается.
5. **Per-symbol weighted ensemble (а не winner-takes-all)**. Вместо «один best actor per symbol» — top-K акторов с весами `softmax(score / T)`. Когда edge у одного актора реально лучший, T мала и поведение сходится к winner-takes-all. Когда edge размыт — Пантеон диверсифицирует, что снижает variance.
6. **Decay стороннего шума**. Метрики каждого signal key должны иметь exponential half-life. Сейчас «full history» и «recent» — два дискретных режима; половинный спад полезнее.
7. **Out-of-sample promotion gate**. См. п. 2.3.10.

### 3.3. Доказательство «не хуже» — операционный контракт

Зафиксировать как часть CI:

1. **Test: panteon_lcb_dominance**. На каждом walk-forward слое: `equity(Panteon) >= equity(best_standalone) * 0.95` после комиссий.
2. **Test: churn_budget**. `avg_actor_switches_per_day < 2 * best_standalone_avg_switches`.
3. **Test: regime_floor**. В каждом из 4 режимов хотя бы 30 закрытых сделок и `pnl >= 0` на out-of-sample.
4. **Test: notrade_value**. Сравнить equity при `min_score_to_trade=current` и `min_score_to_trade=-inf`. Если NoTrade-режим вычитает PnL — фильтры слишком строгие.
5. **Test: ensemble_lift**. Для каждого ensemble player посчитать `lift = pnl_ensemble - pnl_best_member`. Если ни один ensemble не даёт lift > 0 на out-of-sample, упразднить ensemble layer.

### 3.4. Метрика «победа Пантеона»

Один scalar для CI и человеческой инспекции:

```
PanteonAdvantage = (pnl_panteon_net - pnl_best_standalone_net) / max_dd_panteon
```

Если `PanteonAdvantage < 0` — Пантеон вреден, отключить production-путь до фикса.

### 3.5. Roadmap (приоритезирован)

| Приоритет | Изменение | Эффект | Сложность |
| --- | --- | --- | --- |
| P0 | Net-PnL в scoring | Убирает иллюзию edge на high-turnover агентах | S |
| P0 | LCB вместо max | Снимает selection bias | S |
| P0 | Switch margin | Сокращает churn-related fee bleed | S |
| P0 | Per-symbol overextension через ATR | Меньше ложных блокировок | M |
| P1 | Bayesian promotion gate | Истинная статистическая значимость промоушна | M |
| P1 | Softmax ensemble per symbol | Снижение variance Пантеона | M |
| P1 | regime_confidence в score | Меньше торговли на неуверенных переходах | S |
| P1 | Concentration penalty (Herfindahl) | Устойчивость к коррелированной просадке | M |
| P2 | Soft voting (AgentVote вместо Action) | Унификация ensemble и FlashAllocator | L |
| P2 | Walk-forward calibration гиперпараметров | Устойчивость на разных режимах | L |
| P2 | Refactor main_loop.py | Тестируемость, скорость итераций | L |

P0 — изменения с малой сложностью и сильным эффектом. P1 — средний уровень риска. P2 — структурные, требуют отдельной ветки.

---

## 4. Краткие рекомендации по приоритетам действий

1. Сначала исправить P0-баги (1.1, 1.5, 1.6, 1.7, 1.8, 1.12, 1.13, 1.19, 1.20). Это технический долг, который смазывает любые последующие изменения.
2. После — внедрить LCB-ranking и switch_margin (раздел 3.5).
3. Только после стабильного in-sample/out-of-sample gap < 5% на walk-forward — заводить новые экспериментальные wrapper-варианты агентов через shadow.
4. Promotion manifest пересмотреть как Bayesian gate (раздел 1.23, 2.3.8).
5. Перенести часть монолитного `main_loop.py` в отдельные модули (один на каждое решение: real, shadow, flash, fallback). Это сделает остальные изменения дешёвыми.

---

## 5. Что точно нельзя оставлять «как есть» в production

- `regime_confidence` игнорируется → real money риск на переходах.
- Overextension thresholds глобальны → блокировка нормальных сигналов на BTC, пропуск экстремумов на альтах.
- Скоринг без вычета издержек → промоушн агентов с отрицательным net edge.
- Отсутствие switch_margin → churn-driven fee bleed.
- Bug 1.13 (Signal.price может стать 0) → потенциальный divide-by-zero в downstream.
- AgentRegistry без блокировки → race в live.

Каждый из этих пунктов на длинной дистанции стоит больше, чем суммарная польза от любого нового агента.
