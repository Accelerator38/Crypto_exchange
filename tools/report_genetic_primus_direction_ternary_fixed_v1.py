"""Verified Russian fixed-band three-state forecast report and static figures."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_genetic_primus_direction_ternary_fixed_v1 as run
import numpy as np
import matplotlib.pyplot as plt

SHORT = {"logit_uniform": "Logit U", "logit_balanced": "Logit B", "boost_uniform": "Boost U", "boost_balanced": "Boost B",
    "always_neutral": "Всегда нейтр.", "majority_train": "Majority train", "random_uniform_3": "Random 1/3",
    "random_train_prior": "Random train-prior", "momentum_deadband": "Momentum",
    "legacy_binary_logistic": "Старая binary logit", "legacy_binary_boosting": "Старый binary boosting",
    "legacy_ternary_train_q50": "Старая ternary train_q50", "legacy_ternary_0.1pct": "Старая ternary ±0,1%"}


def pct(value):
    return "—" if value is None else "{:.2f}%".format(100*value).replace(".", ",")


def dec(value, places=3):
    return "—" if value is None else ("{:0."+str(places)+"f}").format(value).replace(".", ",")


def main():
    c, registration = run.config(), run.registered()
    result = run.read(run.OUT / "results.json")
    for h in c["timeframes_minutes"]:
        audit = run.read(run.OUT / "audit_{}m.json".format(h))
        assert audit["status"] == "PASS" and audit["metric_cells_recomputed"] == 104
    rows, horizons, coins = result["rows"], c["timeframes_minutes"], c["symbols"]
    lookup = {(r["minutes"], r["symbol"], r["method"]): r for r in rows}
    def get(h, coin, method): return lookup[(h, coin, method)]
    conditioned, gates, summary_h = [], [], []
    for row in rows:
        cm = np.asarray(row["confusion_down_neutral_up"])
        sub = cm[np.ix_([0, 2], [0, 2])]; count = int(sub.sum())
        conditioned.append({"minutes": row["minutes"], "symbol": row["symbol"], "method": row["method"],
            "n_actual_and_predicted_move": count, "conditional_sign_accuracy": float(np.trace(sub)/count) if count else None,
            "future_conditioned_diagnostic_only": True})
    conditional_lookup = {(r["minutes"], r["symbol"], r["method"]): r for r in conditioned}
    for h in horizons:
        for coin in coins:
            for candidate in c["candidates"]:
                comparisons = [p for p in result["comparisons"] if p["minutes"] == h and p["symbol"] == coin and p["candidate"] == candidate]
                assert len(comparisons) == 10 and {p["control"] for p in comparisons} == set(c["primary_controls"])
                accuracy = all(all(v > 0 for v in p["simultaneous_lcb95_by_bank"]) for p in comparisons if p["metric"] == "accuracy")
                f1 = all(all(v > 0 for v in p["simultaneous_lcb95_by_bank"]) for p in comparisons if p["metric"] == "macro_f1")
                gates.append({"minutes": h, "symbol": coin, "candidate": candidate,
                    "accuracy_positive_vs_all_5_controls_both_banks": accuracy,
                    "macro_f1_positive_vs_all_5_controls_both_banks": f1,
                    "both_metrics_positive": accuracy and f1, "independent_validation": False})
        metrics = []
        for method in run.methods(c):
            selected = [get(h, coin, method) for coin in coins]
            entry = {"method": method, "median_accuracy": float(np.median([r["accuracy"] for r in selected])),
                "median_macro_f1": float(np.median([r["macro_f1"] for r in selected]))}
            if method in c["candidates"]:
                entry.update(joint_positive_tasks=sum(g["both_metrics_positive"] for g in gates if g["minutes"] == h and g["candidate"] == method),
                    median_log_loss=float(np.median([r["log_loss"] for r in selected])),
                    median_brier=float(np.median([r["multiclass_brier"] for r in selected])))
            metrics.append(entry)
        summary_h.append({"minutes": h, "band_pct": run.core.band(c, h), "methods": metrics})
    summary = {"bands_pct": c["neutral_bands_pct"], "fits": result["fitted_estimators"], "prior_only": result["prior_only"],
        "warning_count": result["warning_count"], "metric_cells": len(rows), "family_cells": len(result["comparisons"]),
        "joint_positive_tasks": sum(g["both_metrics_positive"] for g in gates), "total_candidate_asset_horizon_tasks": 128,
        "historical_interval_already_revealed": True, "independent_validation": False, "by_horizon": summary_h}
    run.parent.write(run.OUT / "summary.json", summary)
    run.parent.write(run.OUT / "direction_conditioning.json", {"rows": conditioned, "definition": "sign accuracy conditional on both actual and predicted non-neutral; known-future subgroup, not full-coverage or deployable filtering"})
    run.parent.write(run.OUT / "research_comparison_gates.json", {"rows": gates, "financial_promotion": False, "independent_validation": False})
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    shown = [*c["candidates"], "majority_train", "momentum_deadband"]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.3))
    for h, ax in zip(horizons, axes.ravel()):
        values = np.array([[get(h, coin, method)["accuracy"]*100 for method in shown] for coin in coins])
        image = ax.imshow(values, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(shown)), [SHORT[m] for m in shown], rotation=15)
        ax.set_yticks(range(8), [coin.split("/")[0] for coin in coins])
        ax.set_title("{} мин, полоса ±{}%".format(h, dec(run.core.band(c, h))))
        for i in range(8):
            for j in range(len(shown)):
                ax.text(j, i, "{:.1f}".format(values[i, j]), ha="center", va="center", fontsize=9,
                        color="black" if values[i, j] >= 65 else "white")
    fig.suptitle("Общая точность трёх состояний: новые модели и контроли на одинаковых labels", fontsize=13)
    fig.subplots_adjust(left=.06, right=.88, bottom=.09, top=.91, hspace=.40, wspace=.25)
    color_ax = fig.add_axes([.92, .18, .015, .63]); fig.colorbar(image, cax=color_ax, label="Accuracy, %")
    fig.savefig(run.OUT / "accuracy_by_coin.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5), sharey=True)
    colors = ["#c65d5d", "#a5a8ac", "#4d9b70"]
    for h, ax in zip(horizons, axes.ravel()):
        distribution = np.array([get(h, coin, c["candidates"][0])["actual_down_neutral_up"] for coin in coins], float)
        distribution = distribution/distribution.sum(axis=1, keepdims=True)*100
        bottom, x = np.zeros(8), np.arange(8)
        for state_no, (color, title) in enumerate(zip(colors, ["Вниз", "Нейтрально", "Вверх"])):
            ax.bar(x, distribution[:, state_no], bottom=bottom, color=color, label=title)
            if state_no == 1:
                for k, value in enumerate(distribution[:, state_no]):
                    ax.text(k, bottom[k]+value/2, "{:.1f}%".format(value), ha="center", va="center", fontsize=8)
            bottom += distribution[:, state_no]
        ax.set_xticks(x, [coin.split("/")[0] for coin in coins]); ax.set_ylim(0, 100)
        ax.set_title("{} мин, ±{}%; n={} на валюту".format(h, dec(run.core.band(c, h)), get(h, coins[0], c["candidates"][0])["n"]))
        ax.set_ylabel("Фактическая доля класса, %")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Один фиксированный процент не создаёт одинаковую задачу для разных валют", fontsize=12)
    fig.tight_layout(rect=(0, .06, 1, .94)); fig.savefig(run.OUT / "class_distribution.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.7))
    for h, ax in zip(horizons, axes.ravel()):
        matrix = []
        for candidate in c["candidates"]:
            cm = sum(np.asarray(get(h, coin, candidate)["confusion_down_neutral_up"]) for coin in coins)
            matrix.append(np.diag(cm)/cm.sum(axis=1)*100)
        matrix = np.array(matrix)
        image = ax.imshow(matrix, vmin=0, vmax=100, cmap="Blues", aspect="auto")
        ax.set_xticks(range(3), ["Вниз", "Нейтрально", "Вверх"])
        ax.set_yticks(range(4), [SHORT[m] for m in c["candidates"]]); ax.set_title("{} мин".format(h))
        for i in range(4):
            for j in range(3):
                ax.text(j, i, "{:.1f}%".format(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > 60 else "black")
    fig.suptitle("Recall каждого состояния: что именно распознают модели", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94)); fig.savefig(run.OUT / "state_recall.png", dpi=150); plt.close(fig)

    lines = ["# Фиксированные пороги: полный базовый пересчёт трёхклассовых прогнозов", "",
        "Дата: 13 октября 2026. Период: **1 августа — 30 сентября 2026 UTC**, 61 календарный день. "
        "Режим: раскрытый исторический research, без торговли и без новой независимой валидации.", "",
        "## Итог", "",
        "Применены ровно запрошенные пороги: 1 мин ±0,020%; 10 мин ±0,064%; 30 мин ±0,109%; "
        "4 часа ±0,320%. Каждый порог фиксирован для всех восьми валют и всех origins. Это **не** прежний "
        "адаптивный per-asset train_q50 и не процент точности модели.", "",
        "Заново обучены логистическая модель и boosting, каждая с обычными и сбалансированными весами. "
        "Выполнены {} fit, {} prior-only замен, {} предупреждений. Пересчитаны все 416 metric cells, "
        "включая пять общих контролей и четыре сохранённых старых диагностических прогноза.".format(summary["fits"], summary["prior_only"], summary["warning_count"]), "",
        "Сводка ниже — медианы по восьми валютам, не финансовая доходность и не единый глобальный рейтинг.", "",
        "| Горизонт | Полоса | Logit U | Logit B | Boost U | Boost B | Majority train |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for h in horizons:
        values = [np.median([get(h, coin, method)["accuracy"] for coin in coins]) for method in [*c["candidates"], "majority_train"]]
        lines.append("| {} мин | ±{}% | {} | {} | {} | {} | {} |".format(h, dec(run.core.band(c, h)), *(pct(v) for v in values)))
    lines.extend(["", "U — обычные веса всех обучающих исходов; B — class-balanced веса только по train. "
        "Ни одна модель/полоса не выбрана по максимальной тестовой accuracy. Общая accuracy трёх классов "
        "не равна условной точности знака и не сравнима напрямую с прежней бинарной accuracy на других labels.", "",
        "![Accuracy по валютам](accuracy_by_coin.png)", "",
        "Рисунок 1. Каждая ячейка — доля правильных ответов на полном наборе допустимых интервалов данной "
        "валюты и горизонта. Контроли оцениваются на тех же labels. Не показаны здесь legacy-контроли: "
        "они есть в полных таблицах, но обучались на иных определениях цели.", "",
        "## Измерительная семантика", "",
        "r = 100 × (P после горизонта / P в момент прогноза − 1). Ответ «вниз», если r < −δ; "
        "«нейтрально», если |r| ≤ δ; «вверх», если r > δ. Границы включены в нейтральный класс. "
        "δ=0,020% соответствует 2 bps, 0,064% — 6,4 bps, 0,109% — 10,9 bps, 0,320% — 32 bps. "
        "Нейтрально здесь означает малое, а не обязательно нулевое изменение цены.", "",
        "Все прогнозы имеют одно из трёх состояний, coverage=100% допустимой сетки. Нейтральный ответ "
        "не является отказом от выдачи прогноза и сам по себе не задаёт торговое действие.", "",
        "## Данные, модели и причинность", "",
        "- Данные: замороженные `direction_multiscale_v1_20261012/tf_{1,10,30,240}m/{times,closes,features}.npy`. "
        "Сохранены hashes 348 входов: 12 массивов и 336 legacy forecast NPZ. Сырые котировки заново не загружались.",
        "- Валюты: ADA, BNB, BTC, DOGE, ETH, LINK, SOL, XRP. Прогноз следующей свечи: горизонт, шаг примеров "
        "и частота решений равны 1/10/30/240 минут; retrain — каждые 3 дня.",
        "- 21 origin. Rolling30d включает 7 дней calibration; примерно 23 дня для fit. Purge7 баров на "
        "обеих границах, строгая зрелость labels до cutoff. Общий time sample до 4000 времён × 8 валют, "
        "совпадающий с прежним strict-boundary протоколом.",
        "- Все labels, включая нейтральные, остаются в train. Отдельный unweighted scaler на каждом origin; "
        "одинаковый scaler/labels для четырёх кандидатов. Веса меняют только loss, не scaler или временную выборку.",
        "- Logit: прежние 12 признаков, 3 класса, 39 номинальных коэффициентов, C=0,1, max_iter120. "
        "Boosting: 32 итерации, 3 дерева на итерацию, максимум 7 листьев, min_leaf128, L2=10. "
        "У boosting максимум 672 leaf values плюс splits; это более сложная модель, чем linear.",
        "- Balanced weights = N/(3 × train_class_count), нормированы до среднего 1. Расчёт только по "
        "pooled train. При нехватке 32 исходов любого класса предусмотрен помеченный prior fallback; "
        "в этом расчёте он не потребовался.",
        "- Классы вероятностей всегда идут в порядке вниз/нейтрально/вверх. Ledger содержит SHA модели "
        "через candidate-index lookup для каждого прогноза/валюты.",
        "- Ни bands, ни hyperparameters, ни веса, ни calibration-пороги не подбирались по текущему test. "
        "Сами четыре процента выбраны пользователем после просмотра прежних research-отчётов: это раскрыто "
        "как адаптивная история, а не скрыто под видом полностью нового preregistered holdout.", "",
        "## Состав сравнения", "",
        "Полный **базовый** пересчёт охватывает два обучаемых семейства и два режима весов, все контроли, "
        "восемь валют и четыре горизонта. Расширенная матрица lag/context/recency/MLP/GA в этот ограниченный "
        "прогон не включалась; торговые стратегии не переобучались.", "",
        "Основные контроли: всегда нейтрально, majority train, random с 1/3 на класс, random по train priors, "
        "momentum предыдущей свечи с той же полосой. Случайные числа общие и фиксированные.", "",
        "Legacy diagnostics: сохранённые binary logit/boosting, прежняя ternary train_q50 и ternary ±0,1%. "
        "Их ответы заново оценены **на текущих фиксированных labels**, без нового fit. Поэтому абсолютные "
        "результаты в таблицах сопоставимы по test-исходам, но legacy обучались на другом target. "
        "Их преимущество/проигрыш не доказывает автоматически улучшение самого направления.", "",
        "## Классы и точность по каждой валюте", "",
        "![Распределение исходов](class_distribution.png)", "",
        "Рисунок 2. Одинаковая полоса в процентах даёт разную долю нейтральных исходов для BTC и более "
        "волатильных активов. Нельзя сравнивать 45% на этих labels и 47% прежнего train_q50 как изменение "
        "качества одной неизменной задачи."])
    for h in horizons:
        lines.extend(["", "### {} минут, ±{}%".format(h, dec(run.core.band(c, h))), "",
            "n={} на каждую валюту.".format(get(h, coins[0], c["candidates"][0])["n"]), "",
            "| Валюта | Нейтральных | Logit U | Logit B | Boost U | Boost B | Majority | Momentum |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for coin in coins:
            values = [get(h, coin, method)["accuracy"] for method in [*c["candidates"], "majority_train", "momentum_deadband"]]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(coin,
                pct(get(h, coin, c["candidates"][0])["actual_neutral_fraction"]), *(pct(v) for v in values)))
    lines.extend(["", "## Распознавание направлений и проверка устойчивости", "",
        "![Recall по состояниям](state_recall.png)", "",
        "Рисунок 3. Recall каждого фактического состояния, суммирование по восьми валютам только для "
        "визуализации. Это не восемь независимых временных выборок. Полные per-asset confusion, precision/recall, "
        "macro-F1 и MCC — в `results.json` и [подробных таблицах](DETAILS_RU.md).", "",
        "Условная точность знака среди **и фактически, и предсказанно** ненейтральных исходов дана отдельно. "
        "Она использует известный будущий класс для определения подгруппы, поэтому не является deployable "
        "фильтром или полной точностью прогноза. Нейтральные ошибки не выбрасывались из основного accuracy.", "",
        "Block-bootstrap: 61 день, блок 3 календарных дня, 1000 повторов, два общих seed banks. "
        "Одна simultaneous max-t family: 4 кандидата × 5 основных контролей × 8 валют × 4 горизонта × "
        "2 метрики = 1280 сравнений. Legacy диагностические сравнения не выдаются за дополнительный "
        "независимый подтверждающий тест.", "",
        "| Горизонт | Logit U | Logit B | Boost U | Boost B |",
        "|---|---:|---:|---:|---:|"])
    for h in horizons:
        counts = [sum(g["minutes"] == h and g["candidate"] == candidate and g["both_metrics_positive"] for g in gates) for candidate in c["candidates"]]
        lines.append("| {} мин | {} / 8 | {} / 8 | {} / 8 | {} / 8 |".format(h, *counts))
    lines.extend(["", "В этой таблице считаются валюты, где **и accuracy, и macro-F1** имеют положительную "
        "simultaneous paired LCB против **всех пяти** основных контролей в **обоих** seed banks. Всего {} "
        "из 128 candidate/asset/horizon задач. Это внутренний research-критерий на раскрытом периоде, "
        "не независимая валидация и не допуск к деньгам.".format(summary["joint_positive_tasks"]), "",
        "Дополнительно рассчитаны log loss и multiclass Brier для всех новых моделей. "
        "Class-balanced loss меняет priors и может ухудшать калибровку вероятностей при реальном class balance; "
        "нельзя считать его вероятности автоматически калиброванными или выбирать веса лишь по overall accuracy.", "",
        "## Проверки и файлы доказательств", "",
        "| Требование | Реализация / свидетельство | Статус |", "|---|---|---|",
        "| Четыре точных фиксированных порога | config, checks.json, каждый ledger band_pct | PASS |",
        "| Настоящие три класса у двух семейств | 336 pinned model files, probabilities[...,3] | PASS |",
        "| Причинный scaler и зрелые labels, purge7 | folds/cutoff SHA, audit_*m.json | PASS |",
        "| Общая сетка и все 8 валют × 4 горизонта | 84 forecast ledgers, n на валюту | PASS |",
        "| Uniform/balanced при прочих равных | одинаковые train indices/scaler, class_weight_mass | PASS |",
        "| Старые прогнозы на тех же новых labels | legacy_predictions, source pins | PASS |",
        "| Пересчёт метрик и парного bootstrap без fit | 4 audit-прогона, 416 cells, 1280 columns | PASS |",
        "| Полная новая независимая валидация | этот период уже раскрыт | НЕ ВЫПОЛНЕНА |", "",
        "12 детерминированных проверок, 6 синтетических fit, py_compile отдельных файлов. "
        "Аудит без переобучения повторно подтвердил labels, пороги, model SHA, mean/std, priors, веса, "
        "probability reload parity, log loss/Brier, confusion, accuracy/macro-F1, общие bootstrap banks и "
        "финальные simultaneous bounds. Нет предупреждений о сходимости. Входные hashes и версии Python/numpy/sklearn/joblib зафиксированы.", "",
        "Все семь permissions credentials/orders/paper/live/runtime/network/promotion остаются false. "
        "Нет pytest, collector, WebSocket, сетевых запросов, торгов или фоновых процессов. Старые результаты "
        "и dirty worktree вне новых scoped research-файлов не менялись; registry обновлён новой версией v22.", "",
        "## Ограничения и следующий шаг", "",
        "1. 80–90% точности направления не подтверждены. Рост трёхклассового качества может происходить "
        "за счёт предсказуемости масштаба движения/волатильности, а не сильного прогноза знака.",
        "2. Нет единой лучшей архитектуры на всех горизонтах. Не объявлять победителя по сотым долям "
        "процента или агрегированию, в котором минутные строки подавляют остальные горизонты.",
        "3. Порог выбран после истории многократных исследований. Ни bootstrap seed, ни новый registry "
        "не делают август–сентябрь независимыми. Перед следующим подтверждающим этапом нужен замороженный "
        "контракт и действительно новый интервал; существующий prospective-контракт не переписан.",
        "4. Это прогнозный OHLCV-эксперимент. Стаканные single-day/multiday модели, реалистичное исполнение, "
        "капитал, costs и funding здесь не пересчитывались; прибыли и доступности реальных fills из accuracy не следует.",
        "5. В дальнейшей исследовательской модели имеет смысл разделить вероятность существенного движения "
        "и условный знак, чтобы измерять добавку каждого компонента. Это предложение, не уже выполненное новое обучение.", "",
        "## Артефакты", "",
        "- [Полная статистика всех 416 model/control cells](DETAILS_RU.md).",
        "- `results.json`, `results_*m.json`, `summary.json`: цифры и paired confidence bounds.",
        "- `direction_conditioning.json`, `research_comparison_gates.json`: условная диагностика и строгие research counts.",
        "- `origin_*/tf_*m.npz/.json/.joblib`: решения, модель SHA, folds/cutoff, параметры и вероятности.",
        "- `registration.json`, registry v22, `checks.json`, `audit_*m.json`: воспроизводимость.",
        "- Отчёт Markdown/PNG; DOCX не запрашивался и не создавался.", ""])
    with (run.OUT / "REPORT_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(lines))

    details = ["# Полные результаты фиксированной трёхклассовой задачи", "",
        "Период 01.08–30.09.2026 UTC, раскрытый research. U — обычные веса; B — balanced. "
        "Основная accuracy включает все интервалы; условная точность знака — post-hoc подгруппа с известным будущим."]
    for h in horizons:
        details.extend(["", "## {} минут, ±{}% — обученные модели".format(h, dec(run.core.band(c, h))), "",
            "| Валюта | Модель | Accuracy | Macro-F1 | MCC | Recall вниз / нейтр. / вверх | Условный знак | n условных | Log loss | Brier |",
            "|---|---|---:|---:|---:|---|---:|---:|---:|---:|"])
        for coin in coins:
            for candidate in c["candidates"]:
                row = get(h, coin, candidate); cond = conditional_lookup[(h, coin, candidate)]
                details.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(coin, SHORT[candidate],
                    pct(row["accuracy"]), pct(row["macro_f1"]), dec(row["mcc"]), " / ".join(pct(v) for v in row["recall_down_neutral_up"]),
                    pct(cond["conditional_sign_accuracy"]), cond["n_actual_and_predicted_move"], dec(row["log_loss"]), dec(row["multiclass_brier"])))
        details.extend(["", "### {} минут — все основные и legacy-контроли".format(h), "",
            "| Валюта | Метод | n | Accuracy | Macro-F1 | Факт вниз / нейтр. / вверх, n | Recall вниз / нейтр. / вверх |",
            "|---|---|---:|---:|---:|---|---|"])
        for coin in coins:
            for method in [*c["primary_controls"], *c["legacy_diagnostics"]]:
                row = get(h, coin, method)
                details.append("| {} | {} | {} | {} | {} | {} | {} |".format(coin, SHORT[method], row["n"],
                    pct(row["accuracy"]), pct(row["macro_f1"]), " / ".join(str(n) for n in row["actual_down_neutral_up"]),
                    " / ".join(pct(v) for v in row["recall_down_neutral_up"])))
    with (run.OUT / "DETAILS_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(details)+"\n")
    files = [p for p in run.OUT.rglob("*") if p.is_file()]
    files.extend([*run.CODE, run.REGISTRY, Path(__file__), run.ROOT / "tools/check_genetic_primus_direction_ternary_fixed_v1.py"])
    artifact_pins = {str(p.relative_to(run.ROOT)).replace("\\", "/"): run.parent.digest(p) for p in files}
    run.parent.write(run.OUT / "delivery_manifest.json", {"artifact_pins": artifact_pins, "summary": summary,
        "independent_validation": False, **{k: False for k in run.core.SAFETY}})
    print(run.parent.json.dumps({"report": "REPORT_RU.md", "charts": 3, "rows": len(rows),
        "joint_positive_tasks": summary["joint_positive_tasks"], "pinned_artifacts": len(artifact_pins)}))


if __name__ == "__main__": main()
