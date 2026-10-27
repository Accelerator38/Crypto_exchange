# Post-close clocks и preflight — 8 октября 2026

Новый контракт времени реализован и проверен: **21/21** price-free unittest.
Deployment/retrain каждые 3 дня; выход — следующий scheduled origin,
а не entry fill + 3 дня. Entry после готовности закрытого бара и compute;
reference → latency → отдельный свежий post-arrival snapshot.
Неполные clocks не дают нулевой label. Training cutoff = origin − 8 баров:
один бар до training asof и ещё purge 7 баров после созревания исхода.
Max availability включает late funding/cost/mark dependencies.

Синтетический пример: signal +100 ms, reference +200 ms, arrival +450 ms,
endpoint +500 ms; exit на close +3 дня с той же политикой clocks.
Источник ровно на cutoff проходит time gate; на cutoff +1 ms — не проходит.
В обоих случаях training_ready=false: clocks не доказывают реальный fill.

## Данные и реальный блокер

Registered full8 execution bundle present: **False**.
Финансовый replay ready: **False**. Требуются verified L2/depth/sequence,
marks, funding, historical rules/fees и closed-feature prefix для full8.
Есть локальный funding manifest на **1464**
rates за запрошенный период, но rates без исторических availability/observed
marks не закрывают execution qualification. Terminal funding на 1 октября
не следует терять: он может присутствовать в raw, здесь это не проверялось.

Четыре ранее указанных Download-файла найдены по stat; их имена относятся
к BTC в 2024/2025. Содержимое gzip не читалось, coverage не доказана;
это не substitute для full8 за август–сентябрь 2026. Inventory ограничен
известными источниками: отсутствие registered bundle не доказывает отсутствие
всех подходящих файлов на диске. Полный verification L2/marks ещё не выполнен.

Replay data window: 1 августа — 1 октября exclusive UTC, плюс engineering
tail 7.1 s для terminal close. Минимум 49 баров для six features не заменяет
full causal EMA prefix. Для training нужны отдельные зрелые execution labels
по зарегистрированному историческому scope, не ретроспективная замена SHA
старых next-4h-open labels. Latency/participation пока не откалиброваны.

## Следующая работа

1. Подготовить bounded offline parser/reconstruction smoke на имеющихся
   BTC L2/trades/derivative файлах 2024 года; это только проверка формата.
2. Зарегистрировать реальные источники full8 для нужного validation и
   training scope. Шаблон bundle сохранён, все evidence поля незаполнены.
3. Проверить coverage, gaps, sequence и source availability; затем построить
   execution/cost labels, новые hashes и causal scalers. Только после этого
   новый financial replay с непрерывным ledger и equity curve.

Старые прибыльности, objective, validation history и model rankings не
изменены. Новый period не называется prospective. Предыдущие IOC evidence:
10 source hashes и 9
artifact hashes сверены без изменений. New fits/market replays/network
requests = 0. Все девять authority flags false, runtime не открывался.
