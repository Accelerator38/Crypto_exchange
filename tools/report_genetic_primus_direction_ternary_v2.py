"""Russian three-state accuracy report; descriptive fixed bands are not ranked."""
from pathlib import Path
import sys
import os
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_genetic_primus_direction_ternary_v2 as audit
import numpy as np
import matplotlib.pyplot as plt

ROOT, OUT, SUP, parent = audit.ROOT, audit.BASE, audit.SUP, audit.parent
METHOD_NAMES = {"ternary_logit": "Трёхклассовая logit / prior fallback", "always_neutral": "Всегда нейтрально",
    "majority_train": "Majority по train", "random_uniform_3": "Random: 1/3 на каждый класс",
    "random_train_prior": "Random: частоты train", "momentum_deadband": "Momentum с той же полосой",
    "legacy_binary_logistic": "Старая бинарная logit", "legacy_binary_boosting": "Старый бинарный boosting"}
FIXED = ["fixed_0.1pct", "fixed_0.25pct", "fixed_0.5pct", "fixed_1pct", "fixed_2pct", "fixed_3pct", "fixed_4pct", "fixed_5pct"]
SCHEMES = ["exact_zero", *FIXED, "train_q50"]
SYMBOLS = ["ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT", "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT"]
HORIZONS = [1, 10, 30, 240]


def pct(v):
    return "—" if v is None else ("{:.2f}%".format(v*100)).replace(".", ",")


def decimal(v, n=4):
    return "—" if v is None else ("{:0."+str(n)+"f}").format(v).replace(".", ",")


def scheme_name(s):
    if s == "exact_zero": return "Точное отсутствие изменения"
    if s == "train_q50": return "Медиана |return| на train"
    return "±"+s.replace("fixed_", "").replace("pct", "").replace(".", ",")+"%"


def main():
    result = audit.read(OUT / "results.json"); supplement = audit.read(SUP / "results.json")
    for h in HORIZONS:
        check = audit.read(OUT / "audit_tf_{}m.json".format(h))
        assert check["status"] == "PASS" and check["metric_cells_recomputed"] == 640
    rows = [*result["rows"], *supplement["rows"]]
    lookup = {(r["minutes"], r["scheme"], r["symbol"], r["method"]): r for r in rows}
    def get(h, s, coin, method="ternary_logit"):
        return lookup[(h, s, coin, method)]
    conditioned = []
    for row in rows:
        cm = np.asarray(row["confusion_down_neutral_up"])
        directional_cm = cm[np.ix_([0, 2], [0, 2])]
        total = int(directional_cm.sum())
        conditioned.append({"minutes": row["minutes"], "scheme": row["scheme"], "symbol": row["symbol"],
            "method": row["method"], "true_and_predicted_move_n": total,
            "conditional_direction_accuracy": float(np.trace(directional_cm)/total) if total else None,
            "note": "Post-hoc diagnostic conditional on both actual and predicted non-neutral; not deployable filtering and not full-coverage accuracy"})
    parent.write(OUT / "direction_conditioning.json", {"rows": conditioned, "independent_validation": False})
    conditional_lookup = {(r["minutes"], r["scheme"], r["symbol"], r["method"]): r for r in conditioned}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})

    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    for h, ax in zip(HORIZONS, axes.ravel()):
        values = np.array([[100*get(h, s, coin)["actual_neutral_fraction"] for s in FIXED] for coin in SYMBOLS])
        image = ax.imshow(values, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        ax.set_xticks(range(8), [scheme_name(s)[1:-1] for s in FIXED])
        ax.set_yticks(range(8), [c.split("/")[0] for c in SYMBOLS])
        ax.set_title("{} мин: доля фактически нейтральных исходов".format(h))
        ax.set_xlabel("Полоса ±, проценты изменения цены")
        for i in range(8):
            for j in range(8):
                ax.text(j, i, "{:.1f}".format(values[i, j]), ha="center", va="center",
                        fontsize=8, color="black" if values[i, j] >= 65 else "white")
    fig.suptitle("Широкая полоса создаёт почти постоянный класс, а не точный прогноз направления", fontsize=13)
    fig.subplots_adjust(left=.07, right=.89, top=.91, bottom=.08, hspace=.38, wspace=.25)
    color_ax = fig.add_axes([.92, .15, .015, .65]); fig.colorbar(image, cax=color_ax, label="Нейтральные исходы, %")
    fig.savefig(OUT / "neutral_fraction.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8), sharey=True)
    for h, ax in zip(HORIZONS, axes.ravel()):
        x = np.arange(8)
        for shift, method, color, title in [(-.25, "ternary_logit", "#276c9e", "Трёхклассовая модель"),
                (0, "majority_train", "#999999", "Majority train"),
                (.25, "random_uniform_3", "#d6ad62", "Random, 3 класса")]:
            values = [get(h, "train_q50", c, method)["accuracy"]*100 for c in SYMBOLS]
            ax.bar(x+shift, values, width=.24, label=title, color=color)
        model_rows = [get(h, "train_q50", c) for c in SYMBOLS]
        values = np.array([r["accuracy"]*100 for r in model_rows])
        low = np.array([r["calendar_ci95_by_bank"]["accuracy"][0][0]*100 for r in model_rows])
        high = np.array([r["calendar_ci95_by_bank"]["accuracy"][0][1]*100 for r in model_rows])
        ax.errorbar(x-.25, values, yerr=np.stack([np.maximum(0, values-low), np.maximum(0, high-values)]),
                    fmt="none", color="#12344a", capsize=2, linewidth=.8)
        ax.set_xticks(x, [c.split("/")[0] for c in SYMBOLS]); ax.set_ylim(0, 100)
        ax.set_title("{} мин; n={} на валюту".format(h, model_rows[0]["n"]))
        ax.set_ylabel("Общая трёхклассовая точность, %"); ax.grid(axis="y", alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle("Адаптивная полоса: модель против тривиальных контролей, а не против условных 50%", fontsize=12)
    fig.tight_layout(rect=(0, .07, 1, .94))
    fig.savefig(OUT / "primary_accuracy.png", dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5))
    for h, ax in zip(HORIZONS, axes.ravel()):
        cm = sum(np.asarray(get(h, "train_q50", c)["confusion_down_neutral_up"]) for c in SYMBOLS)
        norm = cm/cm.sum(axis=1, keepdims=True)*100
        image = ax.imshow(norm, vmin=0, vmax=100, cmap="Blues")
        ax.set_xticks(range(3), ["Вниз", "Нейтрально", "Вверх"])
        ax.set_yticks(range(3), ["Вниз", "Нейтрально", "Вверх"])
        ax.set_xlabel("Прогноз"); ax.set_ylabel("Факт"); ax.set_title("{} мин".format(h))
        for i in range(3):
            for j in range(3):
                ax.text(j, i, "{:.1f}%".format(norm[i, j]), ha="center", va="center",
                        fontsize=11, color="white" if norm[i, j] > 60 else "black")
    fig.suptitle("Распознавание каждого состояния: нормировка по строкам, все 8 валют", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / "primary_confusion.png", dpi=150); plt.close(fig)

    summary = {"historical_fit_attempts": result["fit_attempts"]+supplement["fit_attempts"],
        "fitted_estimators": result["fitted_estimators"]+supplement["fitted_estimators"],
        "prior_only": result["prior_only"]+supplement["prior_only"],
        "estimator_warning_count": result["estimator_warning_count"]+supplement["estimator_warning_count"],
        "metric_cells": len(rows), "primary_family_cells": 448, "synthetic_fits_including_stopped_v1": 6,
        "no_threshold_selection_on_test": True, "independent_validation": False, "by_horizon": []}
    for h in HORIZONS:
        primary = [get(h, "train_q50", c) for c in SYMBOLS]
        primary_pairs = [p for p in result["comparisons"] if p["minutes"] == h]
        records = {"minutes": h, "primary_median_accuracy": float(np.median([r["accuracy"] for r in primary])),
            "primary_median_macro_f1": float(np.median([r["macro_f1"] for r in primary])),
            "primary_median_always_neutral": float(np.median([get(h, "train_q50", c, "always_neutral")["accuracy"] for c in SYMBOLS])),
            "primary_median_majority": float(np.median([get(h, "train_q50", c, "majority_train")["accuracy"] for c in SYMBOLS])),
            "primary_accuracy_positive_vs_majority_both_banks": sum(p["control"] == "majority_train" and p["metric"] == "accuracy"
                and all(v > 0 for v in p["simultaneous_lcb95_by_bank"]) for p in primary_pairs),
            "fixed": []}
        for scheme in FIXED:
            selected = [get(h, scheme, c) for c in SYMBOLS]
            records["fixed"].append({"scheme": scheme, "median_neutral_fraction": float(np.median([r["actual_neutral_fraction"] for r in selected])),
                "median_model_accuracy": float(np.median([r["accuracy"] for r in selected]))})
        summary["by_horizon"].append(records)
    parent.write(OUT / "summary.json", summary)

    lines = ["# Три состояния цены: пересчёт точности и проверка нейтральной полосы", "",
        "13 октября 2026. Период: **1 августа — 30 сентября 2026 UTC**, 61 день. "
        "Это уже раскрытые исследовательские данные, не новая независимая валидация.", "",
        "## Итог", "",
        "Проверены точное отсутствие изменения, фиксированные полосы ±0,1/0,25/0,5/1/2/3/4/5% и "
        "предварительно заданная адаптивная полоса: медиана абсолютного будущего изменения цены только на train, "
        "отдельно для валюты и горизонта. Модель действительно переобучена на трёх классах; старые бинарные "
        "прогнозы оставлены отдельными контролями, не переименованы в трёхклассовые.", "",
        "Широкие полосы дают 80–100% общей точности даже постоянному ответу «нейтрально». "
        "Это изменение определения цели, а не подтверждение 80–90% точности направления. "
        "Одного универсального порога для 1 минуты и 4 часов нет. ±0,1% меньше ±0,25%, но всё ещё широко "
        "для минутной задачи и узко для четырёхчасовой.", "",
        "| Горизонт | Нейтральных при ±0,1% | Модель при ±0,1% | Нейтральных при ±0,25% | Модель при ±0,25% |",
        "|---|---:|---:|---:|---:|"]
    for h in HORIZONS:
        a, b = [r for r in summary["by_horizon"] if r["minutes"] == h][0]["fixed"][:2]
        lines.append("| {} мин | {} | {} | {} | {} |".format(h, pct(a["median_neutral_fraction"]), pct(a["median_model_accuracy"]),
                     pct(b["median_neutral_fraction"]), pct(b["median_model_accuracy"])))
    lines.extend(["", "Все значения в этой сводке — медианы по восьми валютам. Точность «всегда нейтрально» "
        "равна доле нейтральных исходов. Модель — class-balanced logit; её loss не оптимизирует только overall accuracy "
        "и специально не игнорирует редкие направления.", "", "![Доли нейтральных исходов](neutral_fraction.png)", "",
        "Рисунок 1. Нейтральная полоса меняет class balance по-разному на разных горизонтах. "
        "Цвет и число — доля исходов внутри полосы, не качество обученной модели.", "",
        "## Правило класса и единицы", "",
        "Изменение цены измеряется в процентах: r = 100 × (P после горизонта / P в момент решения − 1). "
        "Если r < −δ — вниз; если |r| ≤ δ — нейтрально; если r > δ — вверх. Границы включены в нейтральную полосу.", "",
        "δ=0,1 означает **0,1% = 10 bps**, δ=0,25 — **0,25% = 25 bps**, δ=5 — **5% = 500 bps**. "
        "«Нейтрально» при δ>0 означает небольшое движение, а не буквально неизменившуюся цену.", "",
        "## Модель, данные и причинность", "",
        "- Источник: замороженные `direction_multiscale_v1_20261012/tf_{1,10,30,240}m/{times,closes,features}.npy`.",
        "- ADA, BNB, BTC, DOGE, ETH, LINK, SOL, XRP. Для каждого горизонта прогнозируется следующая свеча; "
        "горизонт, шаг примеров и решений равны 1/10/30/240 минут. Это отдельный OHLCV-блок, не стаканный однодневный эксперимент.",
        "- Причинное переобучение раз в 3 дня: 21 origin. Rolling30d содержит 7 дней calibration; "
        "примерно 23 дня остаётся для fit. Сохранено до 4000 одинаково выбранных времён для всех схем, по 8 валют на время.",
        "- Purge7 баров на обеих границах; label должен завершиться строго раньше cutoff. "
        "Нейтральные labels не удаляются. Scaler обучается только на fit-строках, без class/target-dependent удаления строк.",
        "- Модель: multinomial logistic regression, 12 прежних признаков, 39 номинальных коэффициентов "
        "(3×(12+1)), C=0,1, class_weight=balanced, max_iter=120. Нет MLP, GA, tuning или торговли.",
        "- Если в pooled train меньше 32 исходов хотя бы одного класса, estimator не обучается: используется "
        "помеченный train-prior fallback. Наличие `.joblib` в таком случае не означает обучение полноценного классификатора.",
        "- Семь контролей: всегда нейтрально, majority train, uniform random по 3 классам, random по train priors, "
        "momentum с той же полосой, два прежних бинарных прогноза. Общая временная сетка, полное покрытие.",
        "- Порог train_q50 вычисляется отдельно для каждой валюты на каждом origin. Он не калибруется по test "
        "или calibration. Пересчёт каждые 3 дня — зафиксированное правило, не адаптивный поиск победителя.", "",
        "## Порог, учитывающий таймфрейм", "",
        "Для рабочей исследовательской постановки оставлено train_q50: нижняя половина абсолютных движений "
        "в обучении считается нейтральной. Это осмысленное заранее заданное определение масштаба, **не доказанный "
        "оптимальный торговый порог**. Отдельная финансовая постановка должна учитывать costs и полезность действий.", "",
        "| Валюта | 1 мин, δ% | 10 мин, δ% | 30 мин, δ% | 240 мин, δ% |", "|---|---:|---:|---:|---:|"])
    for coin in SYMBOLS:
        lines.append("| {} | {} | {} | {} | {} |".format(coin, *(decimal(get(h, "train_q50", coin)["band_pct_median"]) for h in HORIZONS)))
    lines.extend(["", "Это медианы 21 train-порога, не дополнительные подобранные константы. "
        "Минимум и максимум по origins приведены в полной статистике. Одни и те же проценты для всех валют "
        "также могут быть несопоставимы из-за разной волатильности.", "",
        "## Результат адаптивной трёхклассовой задачи", "",
        "![Сравнение модели и контролей](primary_accuracy.png)", "",
        "Рисунок 2. Accuracy модели сопоставляется с majority train и random3 на одинаковых исходах. "
        "Интервалы — точечные 95% calendar-block CI первого seed bank; для выводов по сравнениям ниже "
        "использованы парные simultaneous LCB обоих банков, а не пересечение этих столбиков."])
    for h in HORIZONS:
        lines.extend(["", "### {} минут".format(h), "",
            "| Валюта | n | Модель | Majority train | Macro-F1 | Recall вниз / нейтр. / вверх | Условный знак¹ | n¹ |",
            "|---|---:|---:|---:|---:|---|---:|---:|"])
        for coin in SYMBOLS:
            r, control = get(h, "train_q50", coin), get(h, "train_q50", coin, "majority_train")
            cond = conditional_lookup[(h, "train_q50", coin, "ternary_logit")]
            lines.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(coin, r["n"], pct(r["accuracy"]),
                pct(control["accuracy"]), pct(r["macro_f1"]), " / ".join(pct(v) for v in r["recall_down_neutral_up"]),
                pct(cond["conditional_direction_accuracy"]), cond["true_and_predicted_move_n"]))
    lines.extend(["", "¹ Условная точность знака вычислена только там, где **и факт, и прогноз** ненейтральны. "
        "Это post-hoc диагностическая подгруппа с известным будущим, не реализуемый фильтр, не полное покрытие "
        "и не обещанная точность в торговле. Полные directional precision/recall и confusion есть в JSON.", "",
        "![Матрицы ошибок](primary_confusion.png)", "",
        "Рисунок 3. Строки нормированы по фактическому классу. Так видно, не достигается ли overall accuracy "
        "только распознаванием нейтральных исходов. Валюты суммированы для наглядности, не объявлены независимыми рынками.", "",
        "## Статистика и контроль перебора", "",
        "Фиксированная сетка порогов — описательная чувствительность, без выбора лучшего по тестовой accuracy. "
        "Основная схема train_q50 задана до обучения. Общие calendar block-bootstrap banks: 61 день, блок 3 дня, "
        "1000 повторов, два seed. Simultaneous max-t family: 448 сравнений (4 горизонта × 8 валют × 7 контролей × "
        "2 метрики accuracy/macro-F1). 8 валют не создают восемь независимых временных периодов.", "",
        "| Горизонт | Медиана accuracy | Медиана majority | Медиана macro-F1 | Положительный simultaneous accuracy-LCB против majority в обоих банках, валют |",
        "|---|---:|---:|---:|---:|"])
    for r in summary["by_horizon"]:
        lines.append("| {} мин | {} | {} | {} | {} / 8 |".format(r["minutes"], pct(r["primary_median_accuracy"]),
            pct(r["primary_median_majority"]), pct(r["primary_median_macro_f1"]), r["primary_accuracy_positive_vs_majority_both_banks"]))
    lines.extend(["", "LCB внутри уже раскрытого периода — проверка устойчивости при текущем протоколе, "
        "а не новая независимая валидация. Победа над старым binary на трёхклассовых labels ожидаема, если он "
        "никогда не выдаёт нейтральный ответ; сама по себе она не доказывает более точный знак движения.", "",
        "## Воспроизводимость и ограничения", "",
        "- {} попыток обучения по конфигурациям; {} действительно обученных estimator, {} prior-only fallback. "
        "Дополнительно 6 синтетических fit, включая остановленный v1. Повторного обучения при аудите не было.".format(
            summary["historical_fit_attempts"], summary["fitted_estimators"], summary["prior_only"]),
        "- {} estimator warnings о сходимости/лимитах сохранены. Никакого увеличения итераций после просмотра "
        "теста не делалось. Их локализация — receipts моделей и `results_*m.json`.".format(summary["estimator_warning_count"]),
        "- Аудит подтвердил 840 model/prior SHA, все causal scalers, bands, labels, legacy label parity, "
        "2560 confusion/accuracy/macro-F1 cells и все 448 paired bootstrap columns.",
        "- v1 остановлен до первого исторического fit: parent split допускает endpoint ровно на cutoff, "
        "v1 assertion требовал строго раньше. В v2 явно исключён endpoint равенства (не более одной строки "
        "на границе); старые helper/результаты не изменены. Это не свидетельство leakage в прежней inclusive "
        "границе: семибарный gap сохранялся. Сравнение со старым binary поэтому имеет небольшое различие train-grid на границе.",
        "- ±0,25% включён до v1 registration. ±0,1% добавлен по прямому запросу пользователя отдельным "
        "registry v21 после начала основной работы, но до дополнительных fit и до просмотра итоговых accuracy. "
        "Главный v2 не переписывался и не переобучался из-за дополнения.",
        "- Registry v19/v20/v21 сохранены. Это новая постановка на давно раскрытых данных; полная историческая "
        "адаптивность не устранена одним seed bank. Новый prospective-контракт не изменён.",
        "- Выполнены py_compile отдельных файлов, 10 детерминированных проверок, strict-boundary regression, "
        "четыре независимых аудит-прогона без pytest. Все commands — foreground, один локальный шаг за раз.",
        "- Ни сети, collector, WebSocket, торгового runtime, paper/live, credentials, ордеров или promotion. "
        "Новых OHLCV, trades/L2 не скачивалось. Трёхклассовые book-модели на декабрьском однодневном примере "
        "в этот OHLCV-эксперимент не включались.", "",
        "## Практический вывод", "",
        "Идея нейтрального состояния полезна: она отделяет небольшое движение от направленного значимого. "
        "Но размер полосы следует задавать по задаче и причинному масштабу каждой валюты/горизонта. "
        "Для продолжения зафиксировать train_q50 как рабочий benchmark; не выбирать ±5% ради красивых 99–100%. "
        "Проверять macro-F1, распознавание обоих направлений и добавку к majority/momentum на новом периоде. "
        "80–90% точности направления этим пересчётом не подтверждены.", "",
        "## Файлы", "",
        "- [Полные таблицы по всем валютам/горизонтам/порогам и контролям](DETAILS_RU.md).",
        "- `results.json` + соседний `direction_ternary_01pct_v1_20261013/results.json`: 2560 metric cells.",
        "- `direction_conditioning.json`: явно условная точность знака и размер подгруппы.",
        "- `origin_*/tf_*m.json/.npz/.joblib`: folds, cutoff, SHA моделей, train bands и каждый прогноз.",
        "- `audit_tf_*m.json`, `checks.json`, `strict_boundary_check.json`: воспроизводимость.",
        "- Отчёт Markdown + PNG; DOCX не запрашивался и не создавался.", ""])
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(lines))
    details = ["# Полные таблицы трёхклассового эксперимента", "",
        "Период 01.08–30.09.2026 UTC, раскрытый research. Проценты — не результат торгов. "
        "Доли классов, n и условность показателей не следует игнорировать."]
    for h in HORIZONS:
        for scheme in SCHEMES:
            details.extend(["", "## {} мин — {}".format(h, scheme_name(scheme)), "",
                "| Валюта | n | δ%, min / med / max | Факт вниз / нейтр. / вверх, n | Accuracy модели | Всегда нейтрально | Macro-F1 | MCC |",
                "|---|---:|---|---|---:|---:|---:|---:|"])
            for coin in SYMBOLS:
                r = get(h, scheme, coin)
                details.append("| {} | {} | {} | {} | {} | {} | {} | {} |".format(coin, r["n"],
                    " / ".join(decimal(r[k]) for k in ["band_pct_min", "band_pct_median", "band_pct_max"]),
                    " / ".join(str(v) for v in r["actual_down_neutral_up"]), pct(r["accuracy"]),
                    pct(get(h, scheme, coin, "always_neutral")["accuracy"]), pct(r["macro_f1"]), decimal(r["mcc"], 3)))
        details.extend(["", "## {} мин — все контроли, train_q50".format(h), "",
            "| Валюта | Метод | Accuracy | Macro-F1 | Recall вниз / нейтр. / вверх | Доля направленных прогнозов |",
            "|---|---|---:|---:|---|---:|"])
        for coin in SYMBOLS:
            for method in audit.METHODS:
                r = get(h, "train_q50", coin, method)
                details.append("| {} | {} | {} | {} | {} | {} |".format(coin, METHOD_NAMES[method],
                    pct(r["accuracy"]), pct(r["macro_f1"]), " / ".join(pct(v) for v in r["recall_down_neutral_up"]),
                    pct(r["directional_call_fraction"])))
    with (OUT / "DETAILS_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(details)+"\n")
    files = [p for directory in (OUT, SUP, ROOT/"Reports/Exia/Genetic_Primus/direction_ternary_v1_20261013")
             for p in directory.rglob("*") if p.is_file()]
    files.extend(ROOT/"configs"/("genetic_primus_objective_registry_v{}.json".format(v)) for v in [19, 20, 21])
    files.extend(ROOT/p for directory in (OUT, SUP) for p in audit.read(directory/"registration.json")["code_pins"])
    files.extend([Path(__file__), Path(audit.__file__)])
    artifact_pins = {str(p.relative_to(ROOT)).replace("\\", "/"): parent.digest(p) for p in files}
    parent.write(OUT / "delivery_manifest.json", {"artifact_pins": artifact_pins, "summary": summary,
        "independent_validation": False, "credentials_allowed": False, "orders_enabled": False,
        "paper": False, "live": False, "runtime_authority": False, "network_allowed": False, "promotion_authority": False})
    print(parent.json.dumps({"report": "REPORT_RU.md", "charts": 3, "rows": len(rows),
        "fitted_estimators": summary["fitted_estimators"], "warnings": summary["estimator_warning_count"], "pinned_artifacts": len(artifact_pins)}))


if __name__ == "__main__": main()
