# Причинный rolling replay: пакет и состояние этапов

## Подтверждено

Проверены завершённые gross forecast fits, а не выполнено новое обучение. 20 origins, четыре фиксированных arms, 80 checkpoint-записей, 160 сохранённых моделей. Для каждого решения связаны model/ensemble SHA, scaler SHA, training/fold/weight/prefix hashes, train_end, cutoff и фактическая последняя доступная тренировочная метка. Персональные scaler identities различаются на всех 20 origins, общие внутри каждого origin.

Train_end=origin−1 четырёхчасовой бар; cutoff=train_end−7 баров. Labels созревают после завершения exit-бара; availability≤cutoff. Переобучение каждые три дня; entry следующий open после decision, hold 18 баров.

Фиксированный интервал: **2026-08-01T00:00:00+00:00 — 2026-10-01T00:00:00+00:00, конец exclusive**. Полностью включает 30 сентября. Последний terminal 2026-09-30T04:00:00+00:00; последние 20 часов — cash, без неполного origin и октябрьских outcomes.

## Не выполнено — не подменять результат

Финансовый replay, net-устойчивость и equity curve **не рассчитаны**. Forecast-checkpoint ledger не является position/fill/capital ledger. Нет подтверждённого net-сигнала; RMSE не является его заменой. GA и random search не запускались. Prospective не запечатан и не начат.

Текущий проверенный funding archive покрывает 2026-04-28T16:00:00+00:00 — 2026-07-27T08:00:00+00:00, имеет allowed_use=recent_diagnostic_screen_only и не покрывает ни один август–сентябрьский settlement. Нулевой funding не подставлен. Также не зарегистрированы net policy, капитал/аллокация, normal/stress costs и execution delay. Старые net-target gates сохранены, не сняты задним числом.

## Условный GA против random search

Сначала подтвердить net-сигнал простой модели на причинном pre-August development interval: normal/stress absolute и paired LCB>0, независимый audit bank, risk/stability gates и полный trial registry. Все денежные метрики вычислять через unified shared-capital ledger; fitness — net log-growth под research DD10/20/30%, LCB — проверка устойчивости. Открытый август–сентябрь не использовать для выбора GA-параметров.

У обоих методов один зарегистрированный linear feature-mask/L2 search space, folds и bootstrap banks. На origin: 64 unique objective calls, 128 fits, 64 inner-fold evaluations, 0 gradient steps. Budget verifier проверяет реальные counters и уникальные candidate hashes, завершение одинакового бюджета и идентичность data/space/folds/bank. Дубликаты не считаются новыми evaluations; incumbent считается. Space SHA пока null — это план, не выполненное сравнительное исследование.

## Следующий необходимый ввод

1. Полная funding history full8 за 1 августа — 30 сентября 2026 UTC с settlement basis и проверкой пропусков/интервалов.
2. Зафиксированные normal/stress fees и adverse execution costs с происхождением; research assumptions отдельно от наблюдаемого исполнения.
3. Registered predictor→net-policy: threshold/abstention, капитал, exposure/аллокация и cash/borrow, задержка.
4. Независимая проверка пикового окна 19–22 августа, без его исключения для улучшения метрик.

После устранения входных блокеров — один фиксированный replay всех arms + NoTrade, непрерывные net equity/DD curves, cost/funding/turnover/exposure/carry отчёт и common/audit stability. До этого нельзя заявить, что этап replay завершён. После результата — окончательный контракт/registry, заранее выбранная длительность и genuinely unseen future interval с единственным финальным анализом. Никакого автоматического запуска live/paper.

Проверки: 19 новых pure unittest regressions; 57 текущих source/output identities и восемь funding file identities сверены. Этот шаг: model fits=0, финансовые replay=0, network=0, collector=0, pytest=0. Старые 560 market forecast fits (включая superseded 280) не скрыты.

Артефакты: forecast_checkpoint_ledger.json, status.json, input_requirements.json, engineering_checks.json, evidence_receipt.json.
