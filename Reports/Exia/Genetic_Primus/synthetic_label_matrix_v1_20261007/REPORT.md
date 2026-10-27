# V8 Clean — завершённые labels и training-matrix manifest

Дата: 2026-10-07. Завершён синтетический инженерный этап. Рыночные файлы,
исторические результаты и замороженные контракты не открывались для пересчёта.

## Что добавлено

- Отдельный контракт `configs/genetic_primus_synthetic_label_matrix_contract_v1.json`.
- Чистый builder `src/exia/genetic_primus/synthetic_label_matrix_v1.py`.
- Детерминированный checker `tools/check_genetic_primus_synthetic_label_matrix_v1.py`.

Builder получает проверенный origin/prefix plan и материализует только допустимые
синтетические training-строки. X содержит ровно `return_12` и `efficiency_12`;
значения повторно рассчитаны и сверены с hashes предыдущего feature adapter.
Идентификаторы, число и порядок строк сверены со scaler-input plan.

Y — **синтетический gross diagnostic** `exit_open / entry_open - 1`, не исполненная
сделка и не зарегистрированный net objective. Entry — первый grid open строго
после decision+delay; exit сдвигается вместе с entry и сохраняет 18 баров.
Complete-row publication entry/exit и дополнительный purge на 7 баров проверяются
до использования цен target. Зрелость всей панели и публикация отдельного
символа фиксируются отдельно; разные допустимые времена не считаются ошибкой.

Manifest привязывает SHA X, Y, совместных row IDs/X/Y, исходных entry/exit-строк,
feature dependencies, availability/cutoff, контракта и snapshot. Числовые X/Y и
цены существуют только в памяти синтетической проверки; не выгружаются в результат.
Deployment создаётся отдельным whitelist без target, entry/exit и training-label
полей. Его два числовых признака также только временные, сохраняется общий SHA.

## Результаты

Сначала успешно скомпилирован один новый чистый модуль, затем напрямую проверена
одна функция на отрицательном синтетическом target. `--run` и отдельный `--verify`
успешны: **776 assertions**, **165 неизменённых source/metadata inputs**, receipt
и result совпадают. Команды завершались локально за несколько секунд.

- Для lag=0 матрица 46×2; для 1ms, 120000ms и 4h — 44×2; для 4h+1ms — 42×2.
  Эти lags — тестовые сценарии, не измеренная задержка рынка.
- Flat fixture: сохранены все 46 нулевых targets и abstention-примеры.
- Declining fixture: сохранены все 46 отрицательных targets.
- Mixed fixture: сохранены 36 отрицательных и 10 положительных targets.
  Это счётчики label-исходов, не статистика прибыльности модели.
- Подмена ещё недоступных цен sentinel-значениями не меняет X, Y, paired SHA
  или deployment. Неизменённый snapshot SHA подмену отклоняет.
- Намеренно перепривязанный исторический open меняет Y и paired SHA, но не
  close-only X или deployment. Visible-prefix pins обновляются только в этом
  специально зарегистрированном синтетическом изменении.
- Публикация exit на 1ms после cutoff исключает candidate обеих символов:
  44 вместо 46 строк. Разные публикации внутри cutoff допускаются.
- Пустой eligible bank даёт матрицу 0×2, без fit. Следующий origin получает
  новую матрицу 82×2 и другой paired SHA, не старый scaler/model state.
- Некорректные/неположительные/нефинитные цены, подмена horizon/purge, перевод
  gross в net, нулевой вместо отсутствующего cost model, удаление отрицательных
  или flat/abstention targets и включение authority отклоняются.

## Что этим НЕ подтверждено

Обучение scaler/model: **0**. Parquet reads: **0**. Рыночных финансовых оценок:
**0**. Синтетическая арифметика targets выполнялась, но портфель, costs, turnover,
drawdown, bootstrap, сравнение с NoTrade и promotion не вычислялись.

Pytest, collector, сеть, runtime и фоновые процессы не запускались; NumPy,
pandas и exia не загружались. Все девять safety flags false.

`cost_model=null` означает **отсутствие регистрации**, не бесплатное исполнение.
Фактические fills, исходное arrival evidence, поздние ревизии, compute/model
availability, веса/ESS и minimum opportunities остаются отдельными требованиями.
46/82 строк fixture не доказывают прохождение opportunity/ESS gates.
Этот gross diagnostic запрещено молча использовать как objective новой модели.

## Следующий шаг

Подготовить отдельную явную спецификацию **net accounting и cost scenarios**:
порядок применения комиссий/спреда/slippage, единицы и направление списаний,
нулевую экспозицию NoTrade, одинаковые costs для baseline/challenger и hashes
параметров. Сначала проверить её на синтетике без fit. Не выдавать агрегированные
старые roundtrip budgets за измеренные per-side fees. До historical smoke
зарегистрировать допустимые timing/revision/cost допущения и разрешение запуска;
не расширять экспериментальную матрицу и не называть старый период prospective.

Артефакты: `result.json`, `completion_receipt.json`. Markdown-отчёт вне receipt.
Ни публикации, ни commit/push не выполнялись.
