# Глубокий аудит Пантеон v3 — логика, память, игроки, агенты

Дата: 2026-06-07 · Ветка: `panteon_v3` · Базовый коммит сборки: `eb87b34`

Аудит охватывает: `selection/` (composer, voting, selector, player, strategist,
flash_allocator), `scoring/`, `memory/` (performance, retro_prior, quarantine),
`app/` (main_loop, decision_paths/flash, degradation_tracking),
`analysis/` (retrodate_market_runner, retrodate_validator, soft_allocator),
`shadow/`, `execution/`.

Симптом, заявленный пользователем: **«Пантеон либо не торгует, либо торгует хуже
своих составляющих игроков/агентов»** — и в ретротестах, и в живых торгах.

> Пометки достоверности: **[verified]** — подтверждено прямым чтением кода в этом
> аудите; **[reported]** — найдено суб-агентом с точной ссылкой file:line,
> архитектурно согласовано, но не перечитано вручную построчно.

---

## ЧАСТЬ 1. Как устроена система

### 1.1 Слои
```
Agent (стратегия)  →  Player/EnsemblePlayer (ансамбль агентов + VotingPolicy)
   →  Strategist (legacy: один лидер) ИЛИ FlashAllocator (v3: победитель на символ)
   →  Executor → Exchange (MEXC / v1-futures)
```
- **Agent** — атомарная стратегия, `act(market) -> {sym: Action}`.
- **EnsemblePlayer** (`selection/player.py`) — берёт top-k агентов от `AgentSelector`,
  агрегирует их голоса `VotingPolicy.aggregate`, выдаёт `Signal`-ы. Веса нормированы в 1.0.
- **PlayerComposer** (`selection/composer.py`) — фабрика игроков из `PlayerProfile`.
- Есть **два пути принятия решения**, выбор в `main_loop._run_one_bar:676`:
  - **Flash path** (v3, прод): `FlashAllocator.decide` — на каждый символ выбирает
    ровно одного актора (агент / Solo-обёртка / ансамбль), winner-take-all.
  - **Strategist path** (legacy): `consider_switch` выбирает одного лидера-игрока.

### 1.2 Память / рейтинг
- Класс `PerformanceMemory` (`memory/performance.py`). **Два отдельных инстанса**
  создаются в `app/bootstrap.py:482`: `virtual_perf` (шэдоу, читается селектором/
  стратегистом/аллокатором) и `real_perf` (живые сделки). **Сегрегация по бирже
  достигается двумя инстансами**, а не ключом.
- Основной ключ состояния — `(label, regime)`. `label` — это и агент, и игрок.
- Есть вторичный контекст-ключ `(exchange, symbol, regime, label)`, но в живом пути
  **`exchange=` нигде не передаётся** — контекст-состояние практически мёртвый код **[reported]**.
- **Нет decay/half-life**: счётчики и `returns` копятся вечно; «свежесть» есть только
  в session-overlay (дельта к baseline), но не как затухание.

### 1.3 Скоринг (`scoring/scoring.py`)
`regime_score(metrics, regime)`:
- `score = (pnl·0.62 + sharpe·0.34 + dd·(−0.18) + activity·0.008 + win_bonus) · confidence`
- `confidence = min(closed, 6)/6` — **линейная**, без LCB/Байеса.
- `has_data == False → score = 0.0` (строка 128).
- `inactivity_penalty = −0.65` применяется только если `has_data == True`,
  но `closed==signals==entries==0` (узкий случай execution_failures).

### 1.4 Селекция и голосование
- `AgentSelector.select` (`selector.py:155`) — скорит всех не-карантинных, оставляет
  `score > threshold` (**строгое `>`**, threshold по умолчанию `0.0`), сортирует, top-k.
- `PlayerComposer._build` — вес агента = `score**1.60` (+bias), нормировка в 1.0.
- `WeightedConsensus.aggregate` (`voting.py:113`) на каждый символ:
  `net = Σ w_long − Σ w_short`; открытие если `|net| ≥ open_thr`
  (`open_multi=0.26` при ≥2 голосующих, иначе `open_single=0.34`, пол `0.20`).

### 1.5 Аллокация (Flash, прод)
- `FlashAllocator.decide` — НЕ делит капитал, а выбирает одного актора на символ.
- Каскад гейтов допуска (`_decide_symbol:1650-1852`), порядок сверху-вниз:
  карантин → genetics → solo-suppression → not-hold → regime/symbol deny →
  `has_data` → `closed ≥ min_closed_trades_to_trade(3)` → `pnl_pct ≥ min_pnl_pct_to_trade(0)`
  → семейство shadow-confirmation (closed ≥ **50**, score, win-rate, downside, **LCB**)
  → technical/trend → overextension → degradation/deny → `gate_score ≤ min_score_to_trade(0)`.

### 1.6 Ретротесты
- `retrodate_market_runner.py` гоняет **тот же** `_run_one_bar`, что и лайв (отличие —
  `FakeExchange` и конфиг). Поэтому те же гейты душат и ретро, и лайв.
- Память греется только накоплением shadow-сделок в `virtual_perf` по ходу прогона.
- Сравнение «Пантеон vs составляющие»: лучший компонент — ex-post максимум по PnL за
  весь прогон, без минимума по числу сделок, из отдельных виртуальных книг
  с полным капиталом у каждого.

---

## ЧАСТЬ 2. Корневые причины двух симптомов

### Симптом A: «Пантеон НЕ торгует»
| # | Причина | Где | Дост. |
|---|---------|-----|------|
| A1 | Голосование = доля нормированного веса; одиночный сильный агент даёт лишь `1/N` < `open_single=0.34`; встречный голос вычитается из `net` | `voting.py:158` | **[verified]** |
| A2 | Cold-start: пустой агент → `score=0.0`, селектор берёт строго `>0.0` → отсеян; ансамбль из пустых → `compose=None` | `selector.py:193`, `scoring.py:128`, `composer.py:292` | **[verified]** |
| A3 | Гейт `pnl_pct < min_pnl_pct_to_trade` сравнивает **кумулятивный** PnL с порогом 0.0 → любой исторически-минусовой актор отвергается навсегда | `flash_allocator.py:1708` | **[verified]** |
| A4 | `shadow_confirmation_min_closed_trades = 50` на актора; редко-торгующий ансамбль не набирает 50 shadow-сделок → вечный `shadow_unconfirmed` | `flash_allocator.py:243,1719` | **[verified]** |
| A5 | Score-гейты используют `<=` при пороге 0.0 → актор с ровно нулевым (нет shadow-истории) score отвергается | `flash_allocator.py:1850`, shadow score gate | **[reported]** |
| A6 | Current-actionability gate: кандидат допустим только если в этот же бар уже сработал какой-то компонент → Пантеон не может опередить составляющие по входам | `strategist.py:1305` | **[reported]** |
| A7 | Latched kill-switch (`disabled_reason`) без авто-восстановления → после транзиентной просадки бот навсегда в manage-only | `main_loop.py:6118` | **[reported]** |
| A8 | `min_agents`-гейт: профили требуют 2–3 агента; после фильтра composable их может не хватить → `compose=None` | `composer.py:292`, профили | **[verified]** |

### Симптом B: «Пантеон ХУЖЕ своих составляющих»
| # | Причина | Где | Дост. |
|---|---------|-----|------|
| B1 | Score ансамбля без своей истории = **среднее** скоров агентов (`sum/len`) → регрессия к среднему, всегда ниже лучшего компонента (legacy-путь) | `strategist.py:2412` | **[verified]** |
| B2 | `_component_score` **суммирует** fee/funding/pnl и берёт `max(dd)` по компонентам → синтетический ансамбль переплачивает ~N× комиссий и наследует худшую просадку | `flash_allocator.py:3696` | **[reported]** |
| B3 | LCB (`mean − z·std/√n`) считается по **по-баровым агрегатам**, не по сделкам; у ансамбля меньше баров/больше дисперсия смешения → LCB структурно ниже, чем у частого агента | `soft_shadow_score.py:180`, `flash_allocator.py:1180` | **[reported]** |
| B4 | Solo-обёртка подавляет «сырой» агент, но сама обязана пройти полный shadow-каскад; если обёртка отвергнута — не торгует никто, а дешёвый агент уже убран | `flash_allocator.py:1660` | **[reported]** |
| B5 | `score_close > score_keep` + open-голоса добавляются в `score_keep` → ансамбль дольше держит убыточные позиции, чем закрыл бы агент | `voting.py:172` | **[verified]** |
| B6 | StrongConsensus требует единогласия ВСЕХ активных (HOLD = вето) → `BombermanStrong` почти не открывается | `voting.py:224` | **[verified]** |
| B7 | Сравнение в ретро структурно занижает Пантеон: ex-post лучший компонент без минимума сделок, отдельные книги с полным капиталом vs одна общая книга с гейтами; нулевой slippage сильнее раздувает частых агентов | `retrodate_market_runner.py:4169`, `execution/exchange.py:144` | **[reported]** |
| B8 | Switch-гистерезис (`switch_margin=0.30`, `cooldown=30`, `streak=2`); сброс streak при малом margin → лидер «залипает», прибыльный челленджер не входит | `strategist.py:958-1103` | **[reported]** |

> **Главный вывод.** Симптомы — не один баг, а системная мис-калибровка. Ансамбль
> поставлен в заведомо проигрышные условия относительно своих компонентов на ТРЁХ
> уровнях сразу: (1) скоринг ансамбля = среднее/сумма компонентов; (2) пороги
> голосования и допуска калиброваны под одиночного агента, но применяются к
> разбавленному ансамблю; (3) метрика сравнения в отчётах — ex-post оракул.
> Плюс несколько «глобальных» гейтов (кумулятивный PnL, 50 shadow-сделок,
> строгие `>`/`<=`, current-actionability) душат торговлю у всех, но у ансамбля — сильнее.

---

## ЧАСТЬ 3. Прочие баги, уязвимости, нестыковки

| # | Severity | Проблема | Где | Дост. |
|---|----------|----------|-----|------|
| C1 | **CRITICAL** | Не-атомарная запись снапшота памяти (`open(w)`+`json.dump`); креш при записи рвёт JSON → вся память теряется, бот стартует «холодным» (снова A2) | `app/migration.py:243` | **[reported]** |
| C2 | **HIGH** | Retro-prior строится с `trade_fraction=1.0`, а лайв — `0.10`; `restore` перезаписывает живой `trade_fraction` → seed-PnL и последующий учёт расходятся в ~10× | `memory/retro_prior.py:90`, `performance.py:515` | **[reported]** |
| C3 | HIGH | `is_hopeless_in_all_regimes`: `dominant_loss`-ветка карантинит актора, прибыльного в других режимах; пороги крошечные (−0.30%, 5 сделок) → шум выбивает агентов, пул худеет (→ A8) | `scoring.py:233` | **[verified]** |
| C4 | HIGH | Контекст-состояние per-exchange — мёртвый код в лайве (`exchange=` не передаётся); латентный footgun, если кто-то положится на ключ | `performance.py:181`, `selector.py:241` | **[reported]** |
| C5 | MEDIUM | Priors отбрасывают `returns` → `sharpe=0` у seed-меток, занижают именно те агенты, что должны бутстрапиться | `retro_prior.py:263` | **[reported]** |
| C6 | MEDIUM | Нет decay/half-life; старая «всё-время» память доминирует над текущим эджем; `returns` растёт безгранично | `performance.py` (весь) | **[verified]** |
| C7 | MEDIUM | Quarantine recompute (`qm.recompute`) после композиции, но `_drop_quarantined` вызывается лишь если degradation-gate не no-op → свеже-карантинный игрок проскакивает в strategist-путь | `main_loop.py:629-656` | **[reported]** |
| C8 | MEDIUM | `EnsemblePlayer.vote` вызывает `agent.act` **по разу на символ** (O(N²)); если `act` стейтфул/недетерминирован — голоса по символам несогласованы | `player.py:178-200` | **[verified]** |
| C9 | MEDIUM | Genetics/promotion-метрики: `pnl += sum`, но `sharpe = mean` → несовместимые семантики, гейты пропускают незаслуженно или режут заслуженно | `strategist.py:1923` | **[reported]** |
| C10 | MEDIUM | Недетерминизм ретро при `shadow_parallel_workers>1` (serial-путь без lock) → невоспроизводимый выбор лидера | `shadow_tournament.py:256` | **[reported]** |
| C11 | MEDIUM | Регим BTC и по-символьный регим используют разные окна lookback; первые 24 strided-бара — принудительный NEUTRAL → специалисты режима не матчатся рано | `retrodate_market_runner.py:3358-3404` | **[reported]** |
| C12 | LOW | Neutral-seed: отсутствующее состояние режима подставляет NEUTRAL → размывает «независимую память на режим» | `performance.py:102,348` | **[reported]** |
| C13 | LOW | Empty-run в отчёте неотличим от сбоя: при отсутствии Panteon-PnL бенчмарк не пишется вместо «0 сделок» | `retrodate_market_runner.py:4277` | **[reported]** |
| C14 | LOW | Несогласованные компараторы в profitability-гейтах (`>=` с tolerance vs строгий `>`) → off-by-epsilon на граничных прогонах | `flash_profitability_gates.py:206-252` | **[reported]** |
| C15 | LOW | Unit-mismatch: `pnl_delta` в USD сравнивается с `metric_tolerance_pct` (проценты) | `flash_technical_validation_gates.py:263` | **[reported]** |

### Безопасность (вне основной логики)
- **S1.** В рабочем дереве лежал `ключ мехс.txt` (82 байта, ключевые слова api/secret/key)
  и **не** был в `.gitignore`. В этом аудите добавлены правила в `.gitignore`
  (`ключ*.txt`, `*- legacy/`), секрет в репозиторий **не** попал. **Рекомендация:**
  переместить ключи в `.env`/секрет-хранилище, считать этот ключ скомпрометированным
  и **ротировать** на бирже, т.к. он лежал в открытом виде в рабочей папке.
- **S2.** 3 ГБ legacy-снапшот в рабочем дереве — исключён из git, но стоит вынести из репо-папки.

---

## ЧАСТЬ 4. План устранения и роста прибыльности

Принцип: сначала убрать **системные перекосы против ансамбля** и **глобальные душащие
гейты**, потом — точечные баги, потом — улучшение эджа. Каждый шаг — за отдельным
тестом и за фичефлагом, чтобы можно было A/B сравнивать на ретро.

### Фаза 0 — Безопасность и устойчивость (день 1)
1. Ротировать MEXC-ключ; перевести секреты в `.env`/keyring (S1).
2. Атомарная запись снапшота: `tmp`+`os.replace`, опц. `.bak` (C1).
3. Авто-восстановление kill-switch по времени/восстановлению equity (A7).

### Фаза 1 — Убрать перекос «ансамбль ≤ компонентов» (наибольший рычаг по B)
4. **Скоринг ансамбля от компонентов** заменить `mean` на max-смещённую агрегацию,
   например `0.7·max(agent_scores) + 0.3·weighted_mean` (B1, `strategist.py:2412`).
   То же для `_component_score`: **взвешенные средние** pnl/fee/funding (не суммы),
   портфельная (не `max`) просадка (B2, `flash_allocator.py:3696`).
5. **Унаследовать допуск от компонентов**: засчитывать ансамблю `Σ closed_trades`
   и положительный shadow-LCB компонентов в гейты `min_closed_trades`/shadow-confirm
   (A4, B4) — ансамбль не должен заново зарабатывать историю, которую уже имеют его агенты.
6. **LCB по сделкам, а не по барам**: накапливать пер-трейд PnL из
   `closed_position_outcomes`; либо знаменатель `√(число баров)` согласовать с числителем (B3).

### Фаза 2 — Разблокировать торговлю (симптом A)
7. **Голосование**: открывать, если `score_long_open ≥ open_thr` (направленно, без
   вычитания встречного из `net`), ИЛИ если max-конвикция одного агента сама бы открыла;
   перекалибровать `open_single/open_multi` под нормированные веса; сделать HALF-полосу
   достижимой при одном голосующем (A1, `voting.py`).
8. **Гейт PnL — по `pnl_per_trade`, не по кумулятивному `pnl_pct`**; единицы — % за сделку (A3).
9. **Cold-start**: `score >= min_eligible_score` (нестрогое) или явный exploration-floor
   для нулевых меток, чтобы хотя бы `min_agents` выживали (A2, A8, `selector.py:193`).
10. Привести score/shadow-гейты к согласованной строгости (`<` вместо `<=`) или порогам `>0` (A5).
11. **Decouple current-actionability** от eligibility: лидер сам выдаёт сигнал в обычном
    vote-пути, а не дисквалифицируется из-за отсутствия совпадающего shadow-бара (A6).
12. Снизить `shadow_confirmation_min_closed_trades` (или масштабировать к частоте актора)
    и/или включить bootstrap-путь по умолчанию (A4).

### Фаза 3 — Память и карантин (устойчивость эджа)
13. **EWMA/half-life** на `returns` (или кольцевой буфер) + затухание счётчиков (C6) —
    рейтинг следит за текущим эджем, а не за древними режимами.
14. Карантин: квантовать по **агрегату across-regime** (не выбивать net-положительных),
    поднять `quarantine_hard_min_closed` до статзначимого (C3).
15. Привести retro-prior к тому же `trade_fraction`, что и лайв; не давать `restore`
    приоритета перезаписывать живой `trade_fraction`; синтезировать `returns` или
    помечать seed как low-confidence (C2, C5).
16. Починить или удалить per-exchange контекст-состояние (C4); согласовать regime-ключи (C7).

### Фаза 4 — Достоверность ретротестов (чтобы решения не были иллюзорными)
17. Сравнивать Пантеон с **causally-selectable** лучшим компонентом (трейлинг-окно),
    а не ex-post оракулом; минимум сделок для «лучшего»; единая модель капитала/
    позиционных лимитов для компонентов и ансамбля (B7).
18. Включить реалистичный slippage/spread в `FakeExchange` ретро, совпадающий с лайвом (B7).
19. Warm-up исключать из окна сравнения; писать в `run_summary.json` число дропнутых
    кандидатов по каждому гейту и причины пустых прогонов (A6, C11, C13).
20. Запретить/детерминизировать `shadow_parallel_workers>1` для бенчмарков (C10).

### Фаза 5 — Рост прибыльности (после стабилизации)
21. **Веса**: заменить жёсткий `score**1.6` на тюнингуемый параметр; перейти к
    риск-скорректированным весам (по shadow-Sharpe/LCB), а не по сырому score (C6, composer).
22. **Регим-специализация**: давать affinity-флор в селекции, чтобы специалист режима
    выживал cold-start в своём режиме (вместо отсева по `score≈0`).
23. **Адаптивные пороги** открытия/закрытия по волатильности и режиму вместо констант.
24. Ввести непрерывную метрику **«ансамбль ≥ best-causal-constituent»** как KPI A/B —
    цель фаз 1–2 считать достигнутой, когда она ≥ 0 на большинстве окон.

### Порядок и проверка
- Каждый шаг — отдельный коммит + юнит-тест + ретро-прогон до/после на одном и том же
  наборе лет; сравнивать PnL, число сделок, alpha-vs-constituents, maxDD.
- Рекомендуемая последовательность по ROI: **Фаза 0 → 1 → 2**, затем 4 (чтобы видеть
  эффект честно), затем 3 и 5.

---

## Приложение. Самые нагруженные строки (быстрый доступ)
- `voting.py:158` — `net = score_long_open − score_short_open` vs `open_thr` (A1, B5)
- `selector.py:193` — `if row.score > threshold` (A2)
- `scoring.py:128` — `if not metrics.has_data: return 0.0` (A2)
- `scoring.py:233` — `dominant_loss` карантин (C3)
- `composer.py:253-267` — `_dynamic_score_weight`, power 1.60
- `strategist.py:2412` — `base_score = sum(agent_scores)/len` (B1)
- `flash_allocator.py:1708` — `pnl_pct < min_pnl_pct_to_trade` кумулятивно (A3)
- `flash_allocator.py:243` — `shadow_confirmation_min_closed_trades = 50` (A4)
- `flash_allocator.py:3696` — `_component_score` суммирует fee/pnl, `max(dd)` (B2)
- `soft_shadow_score.py:180` — LCB по по-баровым агрегатам (B3)
- `migration.py:243` — не-атомарная запись снапшота (C1)
- `retro_prior.py:90` — `trade_fraction=1.0` vs лайв `0.10` (C2)
- `retrodate_market_runner.py:4169` — ex-post «лучший компонент» (B7)
