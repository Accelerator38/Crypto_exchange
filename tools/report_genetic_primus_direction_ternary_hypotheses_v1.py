"""Post-fit evidence audit and Russian report. No estimator fitting or tuning."""
from __future__ import annotations
import run_genetic_primus_direction_ternary_hypotheses_v1 as run
import json
from pathlib import Path
import numpy as np

OUT, ROOT = run.OUT, run.ROOT
BASE_NAMES = ["logit_uniform", "logit_balanced", "boost_uniform", "boost_balanced", "always_neutral", "majority_train", "random_uniform_3", "random_train_prior", "momentum_deadband"]

def metric(cm, name):
    cm = np.asarray(cm, float); diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    if name == "accuracy": return diagonal.sum(axis=-1)/cm.sum(axis=(-2, -1))
    denom = cm.sum(axis=-1)+cm.sum(axis=-2)
    return np.divide(2*diagonal, denom, out=np.zeros_like(diagonal), where=denom != 0).mean(axis=-1)

def audit():
    c, reg = run.registered(); results = run.read(OUT / "results.json")
    weights = []
    for seed in c["bootstrap"]["seeds"]:
        with np.load(OUT / ("bootstrap_bank_%s.npz" % seed), allow_pickle=False) as a:
            expected = run.causal.block_bank(61, 1000, 3, seed)
            assert np.array_equal(a["indices"], expected)
            # Independently count calendar repetitions, not tri.calendar_weights.
            weights.append(np.stack([np.bincount(row, minlength=61) for row in a["indices"]]))
    all_draws = [[], []]; obs = []
    for h in c["timeframes_minutes"]:
        rr = run.read(OUT / ("results_%sm.json" % h)); names = run.methods(c, h)
        with np.load(OUT / ("daily_confusion_%sm.npz" % h), allow_pickle=False) as a: daily = a["daily"]
        with np.load(OUT / ("paired_draws_%sm.npz" % h), allow_pickle=False) as a:
            for bank in range(2):
                saved_draws = a["bank%s" % bank]
                cache = {}; checked = []
                for j, row in enumerate(rr["comparisons"]):
                    asset = c["symbols"].index(row["symbol"])
                    values = []
                    for method in (row["candidate"], row["control"]):
                        key = (asset, method, row["metric"])
                        if key not in cache:
                            cm = np.einsum("bd,dij->bij", weights[bank], daily[asset, names.index(method)])
                            cache[key] = metric(cm, row["metric"])
                        values.append(cache[key])
                    difference = values[0]-values[1]
                    assert np.allclose(difference, saved_draws[:, j], atol=1e-15, rtol=0)
                    checked.append(difference)
                all_draws[bank].append(np.array(checked).T)
        for row in rr["comparisons"]:
            asset = c["symbols"].index(row["symbol"])
            value = metric(daily[asset, names.index(row["candidate"])].sum(axis=0), row["metric"])-metric(daily[asset, names.index(row["control"])].sum(axis=0), row["metric"])
            assert abs(value-row["difference"]) < 1e-15
            obs.append(value)
    obs = np.array(obs); criticals = []
    for bank in range(2):
        draws = np.concatenate(all_draws[bank], axis=1)
        se = draws.std(axis=0, ddof=1); active = se > 1e-12
        critical = max(0., float(np.quantile(((obs[None, active]-draws[:, active])/se[None, active]).max(axis=1), .95)))
        lower = obs-critical*se
        assert np.allclose(lower, [r["simultaneous_lcb_bank%s" % bank] for r in results["comparisons"]], atol=1e-14, rtol=0)
        criticals.append(critical)
    preserved = {}
    old_manifests = ["direction_research_delivery_20261013.json", "direction_hypotheses_delivery_20261013.json",
        "direction_book_multiday_v1_20261013/delivery_manifest.json", "direction_ternary_v2_20261013/delivery_manifest.json",
        "direction_ternary_fixed_v1_20261013/delivery_manifest.json"]
    for name in old_manifests:
        value = run.parent.read(ROOT / "Reports/Exia/Genetic_Primus" / name)
        preserved.update(value.get("artifact_pins", value.get("artifact_sha256", {})))
    for path, sha in {**preserved, **reg["data_pins"]}.items():
        assert run.parent.digest(ROOT / path) == sha, path
    previous = run.read(ROOT / c["previous_registry"]); current = run.read(run.REGISTRY)
    assert current["objectives"][:-1] == previous["objectives"]
    run.parent.write(OUT / "comparison_audit.json", dict(independently_checked_comparisons=len(obs),
        bootstrap_banks=2, independently_recomputed_max_t_criticals=criticals,
        old_artifacts_verified=len(preserved), registered_inputs_verified=len(reg["data_pins"]),
        previous_registry_entries_unchanged=True, estimator_fits=0,
        report_tool_sha256=run.parent.digest(Path(__file__)), **{k: False for k in run.fixed.SAFETY}))
    print(json.dumps(dict(comparisons=len(obs), old_artifacts=len(preserved), registered_inputs=len(reg["data_pins"]), fits=0)))

def report():
    import matplotlib.pyplot as plt
    c, reg = run.registered(); results = run.read(OUT / "results.json"); summary = run.read(OUT / "summary.json")
    audit = run.read(OUT / "comparison_audit.json")
    frame_audits = [run.read(OUT / ("audit_%sm.json" % h)) for h in c["timeframes_minutes"]]
    assert sum(x["models"] for x in frame_audits) == 882
    rows, comparisons = results["rows"], results["comparisons"]
    records = sum([run.read(OUT / ("results_%sm.json" % h))["model_records"] for h in c["timeframes_minutes"]], [])
    medians, leaders = [], []
    for h in c["timeframes_minutes"]:
        for name in run.methods(c, h):
            selected = [x for x in rows if x["minutes"] == h and x["method"] == name]
            medians.append(dict(minutes=h, method=name, accuracy=float(np.median([x["accuracy"] for x in selected])),
                macro_f1=float(np.median([x["macro_f1"] for x in selected])),
                recall=[float(np.median([x["recall_down_neutral_up"][k] for x in selected])) for k in range(3)]))
        expanded = [x for x in medians if x["minutes"] == h and x["method"] not in BASE_NAMES]
        leader = max(expanded, key=lambda x: x["accuracy"])
        base = max([x for x in medians if x["minutes"] == h and x["method"] in c["base_methods"]], key=lambda x: x["accuracy"])
        leaders.append(dict(minutes=h, model=leader["method"], median_accuracy=leader["accuracy"],
            best_base=base["method"], base_median=base["accuracy"], exploratory_difference_pp=100*(leader["accuracy"]-base["accuracy"])))
    fig, axes = plt.subplots(2, 2, figsize=(17, 12), layout="constrained")
    for ax, h in zip(axes.flat, c["timeframes_minutes"]):
        names = [v["id"] for v in run.variants(c, h)]+c["base_methods"]+["majority_train"]
        matrix = np.array([[next(x["accuracy"] for x in rows if x["minutes"] == h and x["symbol"] == coin and x["method"] == name) for coin in c["symbols"]] for name in names])*100
        im = ax.imshow(matrix, vmin=25, vmax=60, cmap="viridis", aspect="auto")
        ax.set_xticks(range(8), [s.split('/')[0] for s in c["symbols"]]); ax.set_yticks(range(len(names)), names, fontsize=9)
        ax.set_title("%s мин — точность трёх состояний, %%" % h)
        for i in range(len(names)):
            for j in range(8): ax.text(j, i, "%.1f" % matrix[i, j], ha="center", va="center", fontsize=8, color="white" if matrix[i,j] < 43 else "black")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Правильные прогнозы, %", shrink=.65)
    fig.savefig(OUT / "accuracy_by_coin.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(19, 7), layout="constrained")
    for ax, h in zip(axes, c["timeframes_minutes"]):
        names = [v["id"] for v in run.variants(c, h)]+["logit_uniform", "boost_uniform", "majority_train"]
        matrix = np.array([next(x["recall"] for x in medians if x["minutes"] == h and x["method"] == name) for name in names])*100
        im = ax.imshow(matrix, vmin=0, vmax=100, cmap="magma", aspect="auto")
        ax.set_xticks(range(3), ["Вниз", "Нейтрально", "Вверх"], rotation=25); ax.set_yticks(range(len(names)), names, fontsize=8)
        ax.set_title("%s мин: медиана recall, %%" % h)
        for i in range(len(names)):
            for j in range(3): ax.text(j, i, "%.0f" % matrix[i,j], ha="center", va="center", fontsize=8, color="white" if matrix[i,j] < 60 else "black")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Полнота истинного класса, %", shrink=.7)
    fig.savefig(OUT / "state_recall.png", dpi=150); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout="constrained")
    for ax, h in zip(axes.flat, c["timeframes_minutes"]):
        names = [v["id"] for v in run.variants(c, h)]; differences = []
        for name in names:
            differences.append([100*(next(x["accuracy"] for x in rows if x["minutes"] == h and x["symbol"] == coin and x["method"] == name)-next(x["accuracy"] for x in rows if x["minutes"] == h and x["symbol"] == coin and x["method"] == "boost_uniform")) for coin in c["symbols"]])
        differences = np.array(differences); central = np.median(differences, axis=1)
        ax.errorbar(central, np.arange(len(names)), xerr=[central-differences.min(axis=1), differences.max(axis=1)-central], fmt="o", capsize=3)
        ax.axvline(0, color="gray", lw=1); ax.set_yticks(range(len(names)), names); ax.invert_yaxis()
        ax.set_title("%s мин — добавка к boost_uniform" % h); ax.set_xlabel("Медиана и min–max по валютам, п.п. (не CI)")
    fig.savefig(OUT / "increment_by_coin_range.png", dpi=150); plt.close(fig)
    max_row = max([x for x in rows if x["method"] not in BASE_NAMES], key=lambda x:x["accuracy"])
    parent_pass = sum(r["simultaneous_lcb_bank0"] > 0 and r["simultaneous_lcb_bank1"] > 0 for r in comparisons if r["control"] in ("logit_uniform", "boost_uniform"))
    lines = ["# Расширенная матрица трёхрежимных прогнозов", "", "Дата: 14 октября 2026. Период исследований: 1 августа — 30 сентября 2026 UTC.", "",
        "## Краткий итог", "", "Расширение не подтвердило общего устойчивого преимущества над базовыми моделями: строгий исследовательский gate прошли **0 из 336** сочетаний модель/валюта/горизонт. Это не доказательство полного отсутствия сигнала: %s отдельных сравнений с базовыми logit_uniform/boost_uniform имеют положительный одновременный LCB в обоих banks, но не выполняют весь набор критериев. Лучшие наблюдаемые медианы трёхклассовой accuracy — 42,86–45,18%%. Ни 80%%, ни 90%% на полном покрытии не получены." % parent_pass, "",
        "Получено 882 модельных артефакта и выполнено 882 исторических estimator fits: 714 одиночных, 84×2 для иерархии; ещё 84 ансамбля повторно используют базовые модели без fit. Пять synthetic fits учтены отдельно. Базовые модели и прежние эксперименты не переобучались и не изменялись.", "",
        "| Горизонт | Лучший новый по медиане accuracy* | Accuracy, % | Лучшая база | База, % | Разница медиан, п.п. |", "|---:|---|---:|---|---:|---:|"]
    for x in leaders: lines.append("| %s мин | %s | %.2f | %s | %.2f | %+.2f |" % (x["minutes"], x["model"], x["median_accuracy"]*100, x["best_base"], x["base_median"]*100, x["exploratory_difference_pp"]))
    lines += ["", "* Постфактум описательный максимум, не заранее выбранный победитель. Разница двух медиан не является парной оценкой эффекта; парные comparisons хранятся отдельно. Максимальная ячейка: %s, %s, %s мин — %.2f%%; выбирать модель по этому максимуму нельзя." % (max_row["method"], max_row["symbol"], max_row["minutes"], max_row["accuracy"]*100), "",
        "## Источники и ограничения данных", "", "Исходные OHLCV memmaps times.npy/closes.npy/features.npy — direction_multiscale_v1_20261012. Pooled-модели обучаются сразу на восьми активах; метрики считаются отдельно для ADA, BNB, BTC, DOGE, ETH, LINK, SOL, XRP. Это не восемь независимо обученных моделей. OHLCV не содержит нового полного потока стакана/агрессоров/on-chain. Все input SHA закреплены в registration.json; сохранённые родительские forecasts и модели — direction_ternary_fixed_v1_20261013.", "",
        "Август–сентябрь уже раскрыт в нескольких исследованиях; пороги также выбраны после их изучения. Прогон зарегистрирован до новых fits, но не является независимой валидацией. Отдельный bootstrap seed не заменяет новые рыночные данные. Неизвестное полное число старых адаптивных попыток не исправляется локальной поправкой этого опыта.", "",
        "## Единая задача и pipeline", "", "1. Для каждого horizon берутся причинные OHLCV-признаки на закрытом интервале.\n2. 21 origin, retrain каждые 3 дня. Rolling 30 дней включает 7 дней calibration: фактический train примерно 23 дня минус purge. Calibration не используется для подбора параметров этого опыта. Строгие endpoints до train/calibration cutoffs, purge 7 баров. Исходная временная выборка совпадает с базой; cap 4000 times.\n3. Scaler mean/std обучается заново на каждом origin только на train; нейтральные исходы не удаляются. Для robust winsor 0,5/99,5 percentiles тоже только train. В иерархии scaler второго этапа обучается только на mature moves; это отдельная условная задача.\n4. Получаются три вероятности; решение argmax на каждом допустимом интервале, без отказов.\n5. Confusion и правильные вероятностные losses накапливаются на полном периоде, по каждой валюте. Общие календарные bootstrap blocks для всех моделей.\n6. Отдельный audit проверяет SHA, folds/cutoffs, scalers, repeat-predict, confusion, bootstrap draws и LCB без обучения.", "",
        "r = 100 × (close[t+h] / close[t] − 1). Down при r < −δ; neutral при |r| ≤ δ (границы включены); up при r > δ. δ: 1m 0,020%; 10m 0,064%; 30m 0,109%; 240m 0,320%. Это фиксированные общие полосы для всех активов/origins, не старый per-asset train_q50. Accuracy = правильные из всех прогнозов; macro-F1 = среднее F1 трёх классов. Нейтральный класс означает движение внутри полосы, не буквально равную цену.", "",
        "## Проверенные гипотезы и архитектуры", "", "| Вариант | Смысл | Признаки | Номинальные параметры / лимит |", "|---|---|---:|---|",
        "| lag_logit / lag_boost | Лаги 1/2/3/6 прошлых баров для трёх OHLCV-компонент | 24 | logit 75; boost до 672 leaf values + splits |",
        "| context_logit / context_boost | Час/сутки/неделя, волатильность, общий рынок, время UTC | 20 | logit 63; boost тот же лимит |",
        "| asset_intercepts | Общие признаки + 7 intercept-индикаторов активов | 19 | logit 60 |",
        "| robust_logit | Train-only winsorization | 12 | logit 39 |",
        "| recency90 | Half-life 90 дней для ВСЕХ трёх исходов | 12 | logit 39 |",
        "| mlp83 | 6→8→3, tanh, LBFGS, alpha=10 | 6 | 83; max_iter=80, max_fun=160 |",
        "| temporal_ensemble3 | Текущий + до двух прошлых logit_uniform; текущие признаки | 12 | 1–3×39; новых fit нет |",
        "| large_train_logit / large_train_boost | Nested cap 16000 вместо 4000 только для 1m | 12 | 39 / тот же boost |",
        "| neutral_direction_hierarchy | move/neutral → up/down | 12 на этап | 2 бинарных logit, всего 26 |", "",
        "Logit: C=0,1, max_iter=120, tol=1e−6. Multiclass коэффициенты указаны номинально, с softmax избыточностью. Boost: 32 итерации, 3 дерева на итерацию, 7 листьев, min leaf 128, learning rate 0,08, L2=10, early stopping отключён. Все варианты uniform по классам; recency меняет только временной вес exp2(−age/90d), scaler невзвешенный. Большая выборка только на 1m: на остальных сетках cap=4000 уже не ограничивал объём. В context 240m часовой lag заменён доступным лагом 4h, это явно зарегистрировано.", "",
        "Binary MLP65 не выдаётся за модель с прежним числом параметров: три выхода дают 83. Старый exact-flat gate с отказами заменён full-coverage band-aware иерархией: P(down)=P(move)×(1−P(up|move)); P(neutral)=1−P(move); P(up)=P(move)×P(up|move). Старые бинарные price-probabilities не используются как новые трёхклассовые targets.", "",
        "## Сходимость и воспроизводимость", "", "**Все 84 исторических MLP fits достигли лимита 80 итераций.** Их результаты воспроизводимы, но оптимизация не сошлась; увеличение лимита после просмотра теста было бы новым опытом. Других warning нет, prior fallback не потребовался. Добавление сложности не дало ожидаемого скачка точности.", "",
        "Пройдены 9 pure checks и 5 synthetic estimator fits. Repeat-predict всех 882 артефактов: максимальная ошибка 0. Confusion пересчитаны независимым накоплением, проверены все 624 metric cells. Второй audit независимо проверил 4704 paired differences, все bootstrap columns в двух banks и max-t LCB. Проверены %s старых artifact SHA и %s зарегистрированных input pins; прежние registry entries неизменны." % (audit["old_artifacts_verified"], audit["registered_inputs_verified"]), "",
        "## Неопределённость и ranking", "", "Для каждого нового метода/актива/горизонта сохранены 95% marginal CI accuracy/macro-F1. Основной gate строже: одновременный односторонний max-t LCB положителен для accuracy и macro-F1 против всех 7 controls (neutral, majority, два random, momentum, logit_uniform, boost_uniform), в обоих banks. Семейство — 4704 comparisons, 1000 повторов, circular календарные blocks 3 дня, 61 день. Block size зафиксирован, но его достаточность при более длинной зависимости не доказана. Нельзя трактовать все сотни тысяч минутных строк как независимые наблюдения.", "",
        "## Все медианы по валютам", "", "| Мин | Метод | Accuracy, % | Macro-F1 | Recall вниз / нейтр. / вверх, % |", "|---:|---|---:|---:|---|"]
    for x in medians: lines.append("| %s | %s | %.2f | %.4f | %s |" % (x["minutes"], x["method"], x["accuracy"]*100, x["macro_f1"], " / ".join("%.1f" % (100*v) for v in x["recall"])))
    lines += ["", "![Accuracy по активам](accuracy_by_coin.png)", "", "Рисунок 1. Трёхклассовая accuracy на одинаковых labels, по каждой валюте; полный набор расширенных вариантов и базовых моделей.", "", "![Recall каждого класса](state_recall.png)", "", "Рисунок 2. Медианная полнота по классам. Преимущество за счёт neutral не означает качественное угадывание up/down.", "", "![Разброс добавки](increment_by_coin_range.png)", "", "Рисунок 3. Медиана и min–max парной добавки к boost_uniform по восьми активам; полосы — межактивный разброс, не доверительные интервалы. Показывает неоднородность улучшений.", "",
        "## Что делать дальше", "", "1. Считать этот цикл завершённым отрицательным результатом для большого расширения; не добирать 80–90% перебором порогов на раскрытом периоде.\n2. Для нового теста оставить компактную базу boost_uniform и logit_uniform. Как ограниченный research shortlist рассмотреть context_boost на 30m и recency90 на 4h — наблюдаемые улучшения небольшие и ещё не доказаны; не объявлять их победителями и не переносить модель в торги.\n3. До нового fit проверить календарную и межактивную стабильность, калибровку вероятностей и признаки на более широком train-only development участке. Утверждать новый контракт и бюджет отдельно.\n4. Запечатать действительно новый временной период, сохранив bands, controls и causal preprocessing; единственный финальный анализ, без настройки по промежуточным результатам.\n5. Для минутных направлений приоритет — многодневные синхронные L2/trades и независимая проверка их availability; однодневный BTC book не заменяет новый период.\n6. Финансовый эффект проверять отдельным экспериментом с actual-transition costs/funding и непрерывным ledger, только после подтверждения предсказательной добавки. Все credentials/orders/paper/live/runtime/network/promotion разрешения остаются false.", "",
        "## Артефакты и границы отчёта", "", "registration.json — контракт, code/data hashes; objective_registry_v23 — append-only objective; origin_*/tf_* — probabilities, model/fold/scaler SHA и cutoffs; results.json — полные 624 строки, 4704 comparisons и gates; summary.json — бюджет; audit_*m.json и comparison_audit.json — проверки; DETAILS_RU.md — таблицы всех методов/валют/горизонтов, интервалы и directed-call precision; delivery_manifest.json — финальные hashes. Таблицы доступны в Markdown/JSON; CSV/XLSX/DOCX не создавались. Навык научно-технических отчётов применён для evidence-first структуры, формул, проверок и ограничений. Прибыль/капитал, торговые fills и live не оценивались: это разрешённый отдельный прогнозный блок без торговли."]
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(lines)+"\n")
    details = ["# Подробные метрики всех трёхрежимных моделей", "", "Период 01.08–30.09.2026 UTC. Все методы на одинаковых допустимых test times и фиксированных bands; покрытие 100%. CI marginal, не поправлены по всему семейству. Порядок recall/распределений: вниз / нейтрально / вверх.", ""]
    for h in c["timeframes_minutes"]:
        for coin in c["symbols"]:
            selected = [x for x in rows if x["minutes"] == h and x["symbol"] == coin]
            actual = selected[0]["actual_down_neutral_up"]; n = sum(actual)
            details += ["## %s — %s мин" % (coin, h), "", "N=%s; истинные классы: %s (%s%%)." % (n, actual, " / ".join("%.2f" % (100*x/n) for x in actual)), "",
                "| Метод | Accuracy, % | Macro-F1 | MCC | Recall вниз/нейтр./вверх, % | Вызовы движения, % | Верный направленный вызов*, % | LogLoss | Brier |", "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
            for row in selected:
                def fmt(value, multiplier=1): return "—" if value is None else "%.4f" % (value*multiplier)
                details.append("| %s | %.2f | %.4f | %s | %s | %.2f | %s | %s | %s |" % (row["method"], row["accuracy"]*100, row["macro_f1"], fmt(row["mcc"]), " / ".join("%.1f" % (100*v) for v in row["recall_down_neutral_up"]), row["directional_call_fraction"]*100, fmt(row["directional_precision_including_false_moves"], 100), fmt(row.get("log_loss")), fmt(row.get("multiclass_brier"))))
            details += ["", "* Верный направленный вызов / все направленные вызовы, включая ложное движение на neutral. Это условная precision, не полная accuracy и не новое покрытие.", "", "| Новый метод | 95% CI accuracy bank0 / bank1, % | 95% CI macro-F1 bank0 / bank1 |", "|---|---|---|"]
            for row in selected:
                if "ci95_by_bank" in row:
                    ci = row["ci95_by_bank"]
                    details.append("| %s | %s | %s |" % (row["method"], " / ".join("[%.2f; %.2f]" % (100*b[0],100*b[1]) for b in ci["accuracy"]), " / ".join("[%.4f; %.4f]" % tuple(b) for b in ci["macro_f1"])))
            details.append("")
    with (OUT / "DETAILS_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(details)+"\n")
    run.parent.write(OUT / "report_statistics.json", dict(medians=medians, exploratory_leaders=leaders,
        max_new_accuracy_cell={k: max_row[k] for k in ("minutes", "symbol", "method", "accuracy")},
        individual_parent_comparisons_positive_both_banks=parent_pass,
        report_tool_sha256=run.parent.digest(Path(__file__)), financial_replay=False))
    print(json.dumps(dict(metric_cells=len(rows), figures=3, report=run.rel(OUT / "REPORT_RU.md"))))

def manifest():
    c, reg = run.registered()
    for name in ("REPORT_RU.md", "DETAILS_RU.md", "accuracy_by_coin.png", "state_recall.png", "increment_by_coin_range.png", "comparison_audit.json"):
        assert (OUT / name).is_file()
    artifact_pins = {run.rel(p): run.parent.digest(p) for p in OUT.rglob("*") if p.is_file() and p.name != "delivery_manifest.json"}
    artifact_pins.update(run.pins()); artifact_pins[run.rel(Path(__file__))] = run.parent.digest(Path(__file__))
    artifact_pins[run.rel(run.REGISTRY)] = run.parent.digest(run.REGISTRY)
    run.parent.write(OUT / "delivery_manifest.json", dict(artifact_pins=artifact_pins,
        figures_visually_inspected=True, independent_validation=False, **{k: False for k in run.fixed.SAFETY}))
    print(json.dumps(dict(final_artifact_pins=len(artifact_pins))))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["audit", "report", "manifest"])
    args = parser.parse_args(); globals()[args.stage]()
