# Timing / execution / data: отдельный исследовательский контракт

Это проект для forward-research, а не изменение замороженных контрактов и не
допуск к fit, paper или реальным деньгам. Все полномочия остаются `false`.
Никакая числовая latency, fee, slippage или funding не выбрана автоматически.

## Разделение двух задач

- Новый OHLCV-контракт: **проверка чувствительности** к заранее зарегистрированной
  задержке. Планируемый open не считается реальным fill. Нет intrabar ликвидности,
  IOC partial fills или подтверждения исполнимости цены open.
- Старый release-контракт: IOC taker, первый свежий snapshot после latency,
  наблюдаемый mark, явный funding; fallback на OHLCV запрещён.
  Новый исследовательский код не подменяет этот контракт.

Старый `linear_latency_sensitivity_v1` сдвигал targets на один бар без refit.
Это не подтверждение latency биржи и не проверка полного causal retraining.
Его значения costs не переносятся в новый проект автоматически.

## Согласование горизонта

Пусть `D` — decision-asof, `d` — зарегистрированная задержка, `B` — 4 часа.
Планируемый вход — первый bar open **строго после** `D + d`:

`entry = (floor((D + d) / B) + 1) * B`.

Следующее решение планируется в `D + 3 дня`. При той же задержке
`exit = entry + 3 дня`: обе границы сдвинуты, держание не сокращается.
На точном совпадении ready с open используется следующий open — это явно
консервативное правило, а не предположение о мгновенном исполнении.
Задержка этой ограниченной sensitivity должна быть меньше одного бара;
decision находится в первом баре origin. Изменение политики требует новой версии.

Label — gross open-to-open на этом интервале, **не** legacy open-to-last-close.
Availability — максимум времени поступления наблюдений обоих open, не только
timestamp exit. Допуск в train: availability <= training-asof − 7 баров.
Исторические labels с другой семантикой не переименовываются и не пересчитываются
здесь. Нет reads цен и нет реализации market label builder в этом этапе.
Моделируемый exit нужен для label, но не предписывает реальное закрытие позиции:
неизменный target должен переноситься между refit без лишнего fill.

## Что нужно зарегистрировать

1. Timing: compute budget, decision-to-ready delay, SHA основания выбора бюджета.
   Измеренные start/complete/available каждого будущего fit проверяются отдельно.
2. Data: SHA manifest, timestamp/arrival semantics, availability evidence,
   revision policy, universe, feature definition, training-prefix manifest,
   snapshot cutoff. Полный архив может входить в provenance, но не в идентичность
   train, если содержит будущие deploy-цены.
3. Costs: adapter SHA, normal/stress fee/slippage **per side**, calibration SHA,
   funding evidence, основание применимости borrow, terminal и instrument specs.
   Неприменимость funding/borrow также должна иметь явное основание, не default 0.

Здесь 21 обязательное поле остаётся `null`. `registration_complete` означает
лишь полноту декларации. Даже полностью заполненный объект не подтверждает
содержимое файлов, действительное обучение, исполнение или доходность.

## Проверки привязок

`origin_binding_gate` проверяет SHA самого контракта, training-prefix manifest,
opportunity feature rows для scaler (не все label rows), model SHA решения,
folds/resample/audit SHA, purge-cutoff, причинную цепочку timing, fit budget и
отложенные границы. Разные bank SHA **не доказывают** независимость seeds.
Это проверка заявленных metadata; доказательство реальных inputs/outputs ещё нужно.
Полная финальная оценка модели по одному лишь такому ledger запрещена.

## Проверка без сети

Используется только `.venv/Scripts/python.exe -X utf8`; запуск checker не запускает
pytest, collector, fit или market evaluator. Проверяются малые artificial timestamps,
подмены schema/hashes/authority, границы cutoff и неизменность predecessor inputs.
Артефакты создаются только в новой директории, файлы — exclusive create.
Публикуются assessment/summary/receipt. Повторная проверка: флаг `--verify`.

## Следующий содержательный шаг

Сопоставить **существующие локальные** metadata/спецификации исполнения с 21 полем
и сформировать одну обоснованную регистрацию либо список реально отсутствующих
данных. Не плодить mock stages и не запускать search на старом objective.
После этого отдельно нужны actual data verification, обновлённые executable labels,
достаточные opportunities/ESS, явная авторизация ограниченного research fit и
причинный refit/scaler/model ledger. Ни регистрация, ни synthetic checks не делают
модель лучше NoTrade. Новый sealed interval и статистический бюджет не открыты.
