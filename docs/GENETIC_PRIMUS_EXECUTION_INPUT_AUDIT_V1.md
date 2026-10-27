# Bounded execution input audit v1

Следующий этап после exact-us clock — проверка локальных input/provenance,
не новый model fit, objective, financial replay или downloader.
Все authority flags false. Только `.venv` Python; один короткий локальный
шаг за раз, stdout <=1 KB. Нет collector, pytest, network, background и runtime.

## Разделение evidence

Registered postclose bundle отсутствует/проверяется структурно по старому pure gate.
Шесть компонентов × восемь symbols дают 48 требований. Даже исправленная
declaration JSON не выдаёт допуск: нужен independently verified input receipt.

Old normalized funding за August–September сохранён неизменным, его SHA проверяется.
Отдельно извлекаются только reported rates на Oct 1 boundary из уже имеющихся
raw historical responses. Exact strings, timestamp units, source-file SHA,
row ordinal/hash и reported requestTime сохраняются. Конфликт rates не
разрешается first-wins; пропуск явный. Rate/response time не превращается
в settlement cashflow, final rate, mark или consumer available time.

Terminal closure — отдельное решение полного validation ledger 1 октября,
не дополнительный трёхдневный label. Хвост: end+compute+reference wait+TTL.
Funding на правой границе нельзя потерять, пока до terminal fill остаётся экспозиция.
Извлечённые строки остаются consumable_as_funding_cashflow=false до квалификации.

## Ограничения I/O

Большие Parquet/gzip не читаются и не хешируются. Четыре известные user archives
проверяются только stat. Metadata inventory ограничен Retrodate и отдельным
price audit directory: <=8192 directory entries, <=512 dirs и <=2048 candidate
names per root. Symlinks/reparse points не обходятся; остановка на лимите
явно отмечается incomplete. Filename и mtime не доказывают рыночное покрытие.
Нет поиска всего filesystem или чтения неизвестных Downloads/runtime/secrets.

Читаются только выбранные frozen source/receipt JSON/code/small artifacts;
per-file cap=2 MiB, normalized JSON <=1600 events, raw <=24 pages ×100 rows.
Funding receipt inputs/outputs <=128 pins/section, raw_inputs <=48 pins.
Ранее запечатанные SHA сверяются до работы, все использованные pins — перед выводом.
Новый evidence root exclusive; старые данные/код/receipts не изменяются.

Сначала py_compile только нового audit tool, затем минимальная прямая pure
parser function. Далее bounded one-shot auditor с прямыми pure checks,
не pytest/discovery и не выполнение collector.

## Выход

Отдельные terminal funding report candidates, 48 input gaps, metadata inventory,
summary/direct checks, русский отчёт, source/artifact hash receipts.
0 economic labels/fits/financial replays/network requests. Нет графика прибыли:
капитал и сделки не рассчитываются. Не квалифицированные входы не подменяются
synthetic data, старым next-open ledger или default costs.
