# Forward research proposal и статический аудит v1

Дата: 2026-10-06. Объект — следующий отдельный проект исследования, не изменение
замороженных/prospective контрактов. Статус `draft_not_executable`.

## Что создано

- `configs/genetic_primus_forward_research_proposal_v1.json`: SHA-pinned проект
  причинности, моделей, измерений, lineage, ограничений и testing budget.
- `src/exia/genetic_primus/forward_research_audit_v1.py`: чистые stdlib функции
  проверки проекта и поиска точных AST anchors. Анализируемый код не исполняется.
- `tools/build_genetic_primus_forward_research_audit_v1.py`: bounded чтение
  пяти исходников, metadata binding предыдущего readiness и эксклюзивная публикация.
- `tools/check_genetic_primus_forward_research_audit_v1.py`: детерминированные
  негативные проверки и отдельная сверка опубликованных артефактов.

Публикация: `Reports/Exia/Genetic_Primus/forward_research_audit_v1_20261006`.
Receipt связывает hashes исходников, проекта, четырёх прежних контрактов,
metadata предыдущего этапа и трёх новых JSON outputs. Предыдущий dependency
fingerprint целиком повторно не аттестован: проверены только bindings его
summary/readiness к receipt. Новая версия не доказывает historical dependency closure.

## Проект алгоритма

Два proposed pipelines: uniform и half-life 90d с uniform anchor 20%.
Long/flat overlay, признаки `return_12`, `efficiency_12`, intercept и два
коэффициента, ridge alpha 0.5, без hidden layers и GA. Expanding history,
полный refit коэффициентов и scaler каждые 3 дня, 4h bars и purge 7 bars.
Веса применяются ко всем отобранным opportunity outcomes, включая убытки.
ESS и minimum opportunities не превращаются в требование совершать сделки.
Это ограничение дизайна, а не вывод о превосходстве такого варианта.

Train cutoff должен зависеть от `training_asof`, а не автоматически от
deployment start. Цепочка времени:

`training_asof <= fit_started_at <= fit_completed_at <= model_available_at
<= decision_asof <= submitted_at <= filled_at`.

Также `label_available_at <= training_asof - 7 * bar_duration` и
`feature_available_at <= decision_asof`. Реальные budgets latency, fill semantics
и новая исполняемая label definition ещё не зарегистрированы; они оставлены null,
а не выданы за нулевые costs/delays. Один terminal liquidation в конце полного
интервала, carry без refill на неизменном target, без forced-flat при retrain.

Предлагаемые baseline — NoTrade и отдельно зарегистрированный EMA reference.
EMA definition, cost adapter, risk gates, interval и power design пока не заданы.
Запланировано 2 candidate pipelines, 0 hyperparameter search trials,
0 разрешённых market runs. Историческое общее число trials неизвестно,
DSR/PBO не подтверждены.

## Что действительно установлено по исходникам

16 точных AST anchors найдены в 5 файлах. Это наблюдения структуры, не
доказательство полного dataflow, prefix invariance или корректности исполнения.

- `rolling_linear_retrain_v1.py`: выбранные завершённые labels, purge, train-only
  scaler, recency weights; нет независимых revision/availability guards.
- `rolling_linear_retrain_v2.py`: full fit на каждом origin и active model SHA;
  нет measured fit completion/availability metadata.
- Общая examples table содержит labels. Её построение требует terminal prices
  будущего origin; deploy выбирает строку из этой же таблицы. Прочтение будущего
  label при прогнозе не установлено, но online feature builder этим не доказан.
- `features.py` содержит trailing EMA; причинность на пропусках, revisions и
  точном grid этим не проверена.
- `measurement_lab_v2.py` сдвигает признаки на следующий бар, но записывает
  одинаковый timestamp для signal и execution — zero-latency reference.
- `prospective_continuous_v2.py` пересчитывает scaler и использует supplied
  genomes. Это другой алгоритм, не full linear retraining.

## Критерий результата

Экономический floor — 0 bps/calendar day относительно NoTrade после stress-costs;
просадки допустимы по ответу пользователя. Предлагаемая статистическая проверка —
paired block-bootstrap LCB строго выше 0 на зарегистрированном независимом анализе.
Calendar days включают abstention, общие resample banks нужны для сравнения,
независимый audit bank — для финальной проверки. Это не обещание неубыточности.

Floor 0 не позволяет сам по себе рассчитать мощность: нужен ненулевой эффект,
который исследование должно уметь обнаружить, оценка зависимости и блоков,
confidence/multiplicity rule. Минимально обнаруживаемый эффект для power planning
не обязан менять пользовательский экономический floor. Старые sealed intervals
автоматически не используются и не открываются.

## Проверки и границы

50 metadata assertions прошли; отдельно 13 artifact assertions прошли.
Четыре прежних config-файла проверены unchanged по bytes до/после публикации.
Рыночные parquet, обученные модели, collector и runtime не загружались;
pytest, сеть, обучение и финансовая оценка не запускались. Все authority flags false,
market launch и real-money readiness false. Новые SHA — не доказательство alpha.

## Следующая реализация

Изолированный causal origin payload и чистые metadata guards: отдельно completed
training labels, отдельно deploy features без future label columns; обязательные
availability timestamps, asof cutoff, fit/model/scaler hashes и fail-closed
проверки будущих/неполных/revised inputs. Сначала синтетические metadata fixtures
без обучения; затем только отдельная интеграция по закрытым требованиям.
Не нужно новое sealed окно или дополнительный GA search на этом шаге.
