# Единый forward-research preregistration record

Record фиксирует определения и предложенный исторический engineering plan.
Это **не** готовый исполнимый контракт, разрешение на чтение market inputs,
обучение или статистическая preregistration нового prospective interval.
Дата регистрации — 2026-10-07, позже выбранного historical period.
Исходные frozen contracts и ранее опубликованные receipts не изменяются.

## Зафиксированные определения

4h bars, deployment/retrain каждые 3 дня, purge 7 баров, linear long/flat overlay
с 2 features и intercept: всего 3 коэффициента, ridge alpha 0.5, hidden layers 0.
Два кандидата: uniform и half-life90d с uniform anchor 20%. Все выбранные
completed opportunity outcomes, включая отрицательные, участвуют в весах.
Нет GA, minimum-trades forcing или поиска гиперпараметров.

Scaler: median/IQR по opportunity feature rows, линейная интерполяция quantiles,
IQR <= 1e-12 заменяется на 1, clip ±8. Scaler и коэффициенты refit каждого origin;
intercept не штрафуется ridge. Это нормативная спецификация, основанная на
статическом коде, не тест NumPy runtime и не доказательство actual fit.

Features/opportunity зафиксированы из проверенного synthetic kernel definition.
EMA opportunity становится явно определённым **descriptive** reference той же
семантики, но это не утверждение об эквивалентности историческим EMA результатам.
NoTrade — constant cash equity 1 без позиций/fills в исследовательской конвенции.

Ridge target — gross delayed open-to-open return. Он не равен net после costs.
Экономическая цель пользователя остаётся net >= 0 при допустимых просадках;
нет обещания отсутствия убытков. Executable threshold rule остаётся unresolved:
legacy formula через composite fee не переиспользуется автоматически.

## Исторический engineering plan

Источник — только metadata старого full8 OHLCV manifest. Venue/product этих цен
не подтверждён и не переименовывается в Bitget perpetual market data.
Выбор origin использует только старый calendar и coverage, без outcomes:

`latest origin with origin + 3d + 2 bars < coverage_end_exclusive`.

Два бара — максимальный shift первого open при decision в первом баре origin
и delay < bar. Exit open должен быть внутри архива, не на его exclusive end.
Итог: proposed nominal period **2026-09-22 → 2026-09-25 UTC**; latest required
exit open 2026-09-25 08:00 UTC. Arrival этого open ещё не подтверждён.
Последний nominal origin 2026-09-25 → 2026-09-28 не удовлетворяет этой envelope.

Training-asof планируется за один bar до origin, purge-cutoff ещё на 7 bars раньше.
Это новое явное planning rule, не измеренный timestamp модели/получения данных.
History expanding от coverage start; actual timestamp/revision/availability ещё
не проверены. Планирование не создаёт labels, predictions или scaler values.

Budget: один engineering origin, два pipelines, максимум два planned pipeline
runs с учётом failures. Authorized runs/fit/data reads сейчас 0.
Engineering smoke нельзя использовать для выбора лучшей модели или доказательства
edge. Дополнительные seeds/features/retries требуют отдельного record; нельзя
менять определения после outcomes под тем же SHA.

## Что не закрыто этим record

21 поле timing/data/cost регистрации сохраняется unresolved, плюс 9 требований:
threshold rule, drawdown budget, concentration/turnover/drift gates, confidence/
multiplicity, historical trial count, power/block design, actual folds/resample/
audit banks и две даты нового sealed interval. Они не заменяются нулями.
Это requirements разных фаз, не утверждение, что power или canary gate нужен
для чисто инженерной проверки входного файла. Полная financial pipeline закрыта.

Source byte SHA и semantic hashes definitions/engineering plan/record фиксируются
раздельно. Пересчитанный self-hash не позволяет заменить lineage или authority:
validator пересоздаёт ожидаемый record из bound sources.
Receipt наследует predecessor bank, проверяет неизменность source inputs до/после
и использует exclusive creation. Report не входит в bank (создаётся позднее).

## Следующее действие

Структурные mock-проверки больше не закроют неизвестные параметры. Следующий
содержательный шаг требует явной авторизации **bounded offline input verification**
старого historical source: сначала file identity/schema/coverage/duplicates/gaps
и timestamp semantics, без stdout цен/ключей, без model fit, сети и collector.
Это разрешение сейчас не получено и в record не подразумевается.
Затем отдельно требуется согласовать research assumptions и закрыть actual
lineage прежде, чем разрешить единственный инженерный refit.

Финансовая qualification и release сохраняют строгие требования к instrument/
fee/book/mark/funding и measured latency. OHLCV fallback для release запрещён.
Новый sealed interval не открывается; paper/shadow/canary/promotion выключены.
