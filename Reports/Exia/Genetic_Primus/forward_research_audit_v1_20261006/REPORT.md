# Итоги следующего шага — 6 октября 2026

Создан отдельный SHA-pinned **проект** forward research contract и выполнен
статический аудит выбранного пути data → labels → scaler → decisions.
Контракт неисполняемый: нового обучения, рыночного расчёта или допуска к деньгам нет.

## Проверенный результат

- 5 исходников, 16 exact AST anchors; код моделей не исполнялся.
- 50 детерминированных metadata assertions прошли.
- 13 проверок сохранённых артефактов прошли.
- 4 прежних config-файла unchanged по SHA до/после публикации.
- 0 model fits, 0 market strategy evaluations; сеть и runtime выключены.

Ни одна из этих величин не является показателем доходности или статистической
независимости наблюдений. Это scoped аудит, не полный аудит репозитория.

## Главные находки

| Путь | Наблюдение | Последствие |
|---|---|---|
| Linear retrain v1 | Purge labels, train-only median/IQR, веса включают убытки выбранных opportunities | Хорошая основа, но data revision и availability не аттестованы |
| Linear retrain v2 | Полный fit в каждом origin, active model SHA в schedule | Нет fit_started/completed/model_available timestamps |
| Shared examples → deploy | Для построения examples нужны terminal prices будущего окна | Выделить deploy features без labels и без требования будущих цен |
| PreparedWindow | Previous-bar shift, signal timestamp равен execution timestamp | Zero-latency reference, не реалистичное исполнение |
| Continuous prospective | Scaler refit, supplied genomes остаются входом | Нельзя называть это полным переобучением и смешивать с linear retrain |

Источник риска общей таблицы:
[построение будущих цен](E:/Crypto_exchange/src/exia/genetic_primus/rolling_linear_retrain_v1.py:50)
и [выбор deploy строки](E:/Crypto_exchange/src/exia/genetic_primus/rolling_linear_retrain_v2.py:62).
Это не доказательство чтения label прогнозом: в просмотренном prediction path
используются features/proposal. Проблема — отсутствие структурного разделения
и причинного построения deploy payload без будущих outcomes.

Различие алгоритмов:
[full linear fit](E:/Crypto_exchange/src/exia/genetic_primus/rolling_linear_retrain_v2.py:123)
против [supplied genome schedule](E:/Crypto_exchange/src/exia/genetic_primus/prospective_continuous_v2.py:122).
Рейтинг торговых результатов не обновлялся.

## Новый проект исследования

Только linear long/flat overlay: два features и intercept, всего 3 коэффициента,
ridge 0.5, expanding history, refit каждые 3 дня, purge 7 четырёхчасовых баров.
Два proposed variants: uniform и half-life 90 дней + uniform anchor 20%.
Параметры — ограничение предлагаемого дизайна, не доказанный optimum.
Новых GA/search trials нет; разрешённых market runs — 0.

Цель пользователя сохранена: неубыточный итог после costs, временные просадки
допустимы. Economic floor 0 bps/calendar day; proposed independent paired LCB
строго > 0 после stress-costs. Не гарантирует будущую неубыточность.
Нулевой floor не определяет длительность: для power нужны ненулевой detectable
effect и зарегистрированные covariance/block/multiplicity assumptions.

Не заполнены execution-aligned labels, latency budgets, реальная привязка data
snapshots/revisions, cost adapter и risk gates, EMA reference, objective registry,
power design и новые dates. Они остаются blockers, а не считаются нулевыми.
Обязательные per-origin hashes/timestamps перечислены в проекте.

## Следующий шаг без нового обучения

Реализовать изолированный causal origin payload и metadata guards. Разделить
completed labels и deploy features; отвергать future/неполные/revised inputs,
несогласованные asof/cutoff и model/scaler identities. Проверить на небольших
синтетических fixtures. Только после этого переходить к отдельной интеграции;
новый sealed interval сейчас не нужен и не открывался.

`market_research_launch_ready=false`, `real_money_ready=false`.
Все credentials/orders/execution/paper/live/promotion/training/financial_evaluation/
network flags false. Ни collector, ни pytest не запускались. Commit/push не было.

Файлы: [проект контракта](E:/Crypto_exchange/configs/genetic_primus_forward_research_proposal_v1.json),
[методика](E:/Crypto_exchange/docs/GENETIC_PRIMUS_FORWARD_RESEARCH_AUDIT_V1.md),
`proposal.json`, `audit.json`, `summary.json`, `completion_receipt.json` рядом
с этим отчётом. Markdown — пояснение; опубликованный receipt связывает JSON.
