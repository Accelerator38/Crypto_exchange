"""Derived forecast statistics, Russian report, TSV tables and plots; no fits."""
from __future__ import annotations
import run_genetic_primus_direction_multiscale_v1 as b
import numpy as np
from pathlib import Path
import csv
import re
import argparse

NAMES = {"ridge6": "Ridge6", "logistic6": "Logistic6", "logistic12": "Logistic12", "gaussian_nb12": "GaussianNB",
         "lda12": "LDA", "qda12": "QDA", "tree12": "Tree", "forest12": "RandomForest", "extra_trees12": "ExtraTrees",
         "hist_boost12": "Boosting", "sgd_logistic12": "SGD Logistic", "mlp12": "MLP 12→8→1",
         "random_coin": "Random: монета", "train_majority": "Train-majority", "asset_train_majority": "Asset-majority",
         "persistence": "Последний знак", "reverse_last_bar": "Против последнего знака", "ema_direction": "Знак EMA"}
METHODS = b.d.models.MODELS + b.CONTROLS


def stats():
    reg = b.registration()
    rows = []
    for tf in b.TIMEFRAMES:
        b.registration(tf)
        b.sealed(b.tf_path(tf) / "independent_checks.json")
        rows += b.sealed(b.tf_path(tf) / "scores.json")["rows"]
    trained = [r for r in rows if r["model"] in b.d.models.MODELS]
    assert len(rows) == 576 and len(trained) == 384
    families = {}
    for key in ("primary", "audit"):
        draws = []
        for tf in b.TIMEFRAMES:
            path = b.tf_path(tf) / "paired_bootstrap_draws.npz"
            assert b.pin(path) == b.sealed(b.tf_path(tf) / "scores.json")["bootstrap_draws_pin"]
            with np.load(path, allow_pickle=False) as stream:
                for row in trained:
                    if row["timeframe_minutes"] == tf:
                        draws.append(stream[f"{row['model']}|{row['symbol']}|{key}"])
        draws = np.array(draws)
        point = np.array([r[key]["paired_random_delta"] for r in trained])
        se = draws.std(axis=1, ddof=1)
        if np.any(se <= 0): raise ValueError("nondegenerate paired bootstrap required")
        max_t = ((draws - point[:, None]) / se[:, None]).max(axis=0)
        critical = float(np.quantile(max_t, 0.95))
        for i, row in enumerate(trained):
            row.setdefault("simultaneous", {})[key] = {"paired_random_lcb": float(point[i] - critical * se[i]),
                "critical_max_t": critical, "comparisons": 384, "family_alpha": 0.05,
                "status": "approximate_block_bootstrap_family_adjustment_not_independent_validation"}
        families[key] = {"critical_max_t": critical, "comparisons": 384}
    best = []
    for tf in b.TIMEFRAMES:
        for symbol in b.SYMBOLS:
            subset = [r for r in trained if r["timeframe_minutes"] == tf and r["symbol"] == symbol]
            row = max(subset, key=lambda r: r["accuracy_nonflat"])
            best.append({"timeframe_minutes": tf, "symbol": symbol, "model": row["model"],
                         "accuracy_nonflat": row["accuracy_nonflat"], "accuracy_all": row["accuracy_all"],
                         "selection": "descriptive_maximum_of_12_not_independent_test"})
    robust = [r for r in trained if min(r["simultaneous"][k]["paired_random_lcb"] for k in ("primary", "audit")) > 0]
    result = b.write(b.OUT / "comparison.json", {"registration_sha256": reg["sha256"], "rows": rows,
        "family_adjustment": families, "descriptive_best_by_asset": best,
        "paired_random_positive_in_both_banks": [{k: row[k] for k in ("timeframe_minutes", "model", "symbol", "accuracy_nonflat", "simultaneous")} for row in robust],
        "any_full_coverage_accuracy_80": any(r["accuracy_all"] >= 0.8 for r in trained),
        "any_nonflat_accuracy_80": any(r["accuracy_nonflat"] >= 0.8 for r in trained),
        "historical_fits": 1008, "financial_replays": 0, "safety": reg["safety"]})
    print({"model_asset_timeframe_rows": len(rows), "paired_comparisons": len(trained),
           "positive_vs_random_both_banks": len(robust), "any_80_percent": result["any_nonflat_accuracy_80"], "new_fits": 0})


def pct(value, digits=2):
    return "—" if value is None else f"{value * 100:.{digits}f}".replace(".", ",")


def link(name): return (b.OUT / name).as_posix()


def export_tables():
    result = b.sealed(b.OUT / "comparison.json")
    header = ["Таймфрейм_мин", "Метод", "Валюта", "Период", "Интервалов_всего", "Ненулевых", "Рост", "Падение", "Нулей", "Верных",
        "Точность_все_проц", "Точность_ненулевые_проц", "Баланс_точность_проц", "Recall_рост_проц", "Recall_падение_проц",
        "Прогнозов_вверх_проц", "Нулей_проц", "CI95_primary_низ_ненулевые_проц", "CI95_primary_верх_ненулевые_проц",
        "CI95_audit_низ_ненулевые_проц", "CI95_audit_верх_ненулевые_проц", "Добавка_к_random_пп",
        "Совместная_LCB_primary_пп", "Совместная_LCB_audit_пп", "Рост_угадан", "Рост_прогноз_падение", "Падение_угадано",
        "Падение_прогноз_рост", "Ноль_прогноз_рост", "Ноль_прогноз_падение"]
    output = []
    for row in result["rows"]:
        for period, v in [("2026-08_2026-09", row)] + list(row["by_month"].items()):
            c = v["confusion"]
            record = [row["timeframe_minutes"], NAMES[row["model"]], row["symbol"], period,
                *[v[k] for k in ("n_all", "n_nonflat", "n_up", "n_down", "n_flat", "n_correct")],
                *[None if v[k] is None else v[k] * 100 for k in ("accuracy_all", "accuracy_nonflat", "balanced_accuracy", "up_recall", "down_recall", "predicted_up_fraction", "flat_fraction")]]
            if v is row:
                record += [*[a * 100 for a in row["primary"]["accuracy_nonflat_ci95"]],
                           *[a * 100 for a in row["audit"]["accuracy_nonflat_ci95"]], row["primary"]["paired_random_delta"] * 100,
                           *[row.get("simultaneous", {}).get(k, {}).get("paired_random_lcb", 0) * 100 if "simultaneous" in row else None for k in ("primary", "audit")]]
            else: record += [None] * 7
            record += [c[k] for k in ("up_up", "up_down", "down_down", "down_up", "flat_up", "flat_down")]
            assert len(record) == len(header)
            output.append(record)
    path = b.OUT / "statistics_excel_ru.tsv"
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(header)
        for record in output:
            writer.writerow(["" if v is None else (f"{v:.10f}".replace(".", ",") if isinstance(v, float) else v) for v in record])
    with path.open(encoding="utf-8-sig", newline="") as stream:
        parsed = list(csv.reader(stream, delimiter="\t"))
    assert parsed[0] == header and len(parsed) == 1729
    assert all(len(row) == len(header) for row in parsed)
    assert float(parsed[1][10].replace(",", ".")) == round(output[0][10], 10)
    b.write(b.OUT / "table_verification.json", {"file_pin": b.pin(path), "rows_without_header": len(output), "columns": len(header),
        "encoding": "UTF-8 BOM", "delimiter": "tab", "decimal_separator": "comma_Russian_locale", "readback_verified": True,
        "all_metrics_cells_separate": True, "monthly_confidence_intervals_not_computed": True})
    print({"table_rows": len(output), "columns": len(header), "separate_cells_verified": True})


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    result = b.sealed(b.OUT / "comparison.json")
    rows = {(r["timeframe_minutes"], r["model"], r["symbol"]): r for r in result["rows"]}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    for tf in b.TIMEFRAMES:
        matrix = np.array([[rows[tf, model, s]["accuracy_nonflat"] * 100 for s in b.SYMBOLS] for model in METHODS])
        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(8), [s.split("/")[0] for s in b.SYMBOLS])
        ax.set_yticks(range(len(METHODS)), [NAMES[m] for m in METHODS])
        for i in range(len(METHODS)):
            for j in range(8): ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=9)
        ax.set(title=f"Горизонт {tf} мин: верное направление по каждой валюте, %\nНенулевые исходы; все методы без abstention; август–сентябрь 2026",
               xlabel="Валюта", ylabel="Метод")
        fig.colorbar(im, ax=ax, label="Accuracy, %; одна шкала 0–100 для всех горизонтов")
        fig.tight_layout(); fig.savefig(b.OUT / f"accuracy_{tf}m.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for ax, tf in zip(axes.flat, b.TIMEFRAMES):
        best = [next(r for r in result["descriptive_best_by_asset"] if r["timeframe_minutes"] == tf and r["symbol"] == s) for s in b.SYMBOLS]
        x = np.arange(8)
        ax.bar(x - 0.18, [r["accuracy_nonflat"] * 100 for r in best], 0.36, label="Максимум из 12 моделей", color="#4682b4")
        null = [rows[tf, "random_coin", s] for s in b.SYMBOLS]
        ax.bar(x + 0.18, [r["accuracy_nonflat"] * 100 for r in null], 0.36, label="Фиксированная random-монета", color="#aaaaaa")
        ax.axhline(50, color="gray", linestyle=":"); ax.axhline(80, color="#b52e31", linestyle="--")
        ax.set_xticks(x, [s.split("/")[0] for s in b.SYMBOLS]); ax.set(title=f"{tf} минут", ylim=(35, 90), ylabel="Accuracy ненулевых, %")
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Уменьшение горизонта не дало 80%\nМаксимумы выбраны описательно: это не новая независимая валидация", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(b.OUT / "random_and_best_by_asset.png", dpi=140); plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(8)
    for i, tf in enumerate(b.TIMEFRAMES):
        ax.bar(x + (i - 1.5) * 0.2, [rows[tf, "random_coin", s]["flat_fraction"] * 100 for s in b.SYMBOLS], 0.2, label=f"{tf} мин")
    ax.set_xticks(x, [s.split("/")[0] for s in b.SYMBOLS]); ax.set(ylabel="Интервалов без изменения close, %", xlabel="Валюта",
        title="Нулевые движения ограничивают accuracy бинарного ответа по всем интервалам")
    ax.legend(); ax.grid(axis="y", alpha=0.2); fig.tight_layout(); fig.savefig(b.OUT / "flat_share_by_asset.png", dpi=140); plt.close(fig)
    print({"charts": 6, "financial_charts": 0})


def report():
    result = b.sealed(b.OUT / "comparison.json")
    rows = {(r["timeframe_minutes"], r["model"], r["symbol"]): r for r in result["rows"]}
    reg = b.registration()
    warnings = {}
    prediction_checks = 0
    for tf in b.TIMEFRAMES:
        check = b.sealed(b.tf_path(tf) / "independent_checks.json")
        prediction_checks += check["checks"]["prediction_cells"]
        for key, value in check["warnings"].items(): warnings[key] = warnings.get(key, 0) + value
    lines = ["# Прогноз направления на 1 / 10 / 30 минут и 4 часа", "", "Отчёт от 12 октября 2026. Только прогнозы: прибыль, costs, funding и торговое исполнение не рассчитывались.", "",
        "## 1. Итог исследования", "",
        "Проверены 12 обучаемых вариантов и 6 контрольных правил на четырёх таймфреймах, отдельно по восьми валютам. Выполнены 1 008 исторических fit; финансовых replay — 0. В основной статистике 576 сочетаний «метод × таймфрейм × валюта»; вместе с августом и сентябрём отдельно — 1 728 строк.", "",
        "Ни одна модель не достигла 80% ни среди всех интервалов, ни среди ненулевых. Уменьшение горизонта до 1/10/30 минут не превратило текущие OHLCV-признаки в сильный классификатор. Это вывод об этих фиксированных моделях и данных, а не доказательство невозможности прогнозирования вообще.", "",
        "Независимая проверка прежнего 4h-блока точно воспроизвела accuracy всех 96 пар «12 моделей × 8 валют» на ценах, заново агрегированных из минутных OHLCV. Ошибка смещения целевого close не обнаружена. Слабая accuracy не возникла из-за усреднения валют или замены random на постоянное направление.", "",
        "### Описательные максимумы среди 12 моделей", "", "Для каждой ячейки выбран лучший метод **после** сравнения. Это не единая заранее выбранная модель и не независимая оценка выбранного победителя. Таблица показывает accuracy среди ненулевых движений; ниже есть все методы и строгая версия по всем интервалам.", "",
        "| Валюта | 1 мин, % | 10 мин, % | 30 мин, % | 240 мин, % |", "|---|---:|---:|---:|---:|"]
    for s in b.SYMBOLS:
        values = [next(r for r in result["descriptive_best_by_asset"] if r["timeframe_minutes"] == tf and r["symbol"] == s) for tf in b.TIMEFRAMES]
        lines.append("| " + s + " | " + " | ".join(pct(v["accuracy_nonflat"]) for v in values) + " |")
    lines += ["", f"![Random и описательные максимумы]({link('random_and_best_by_asset.png')})", "",
        "### Проверка против random с учётом перебора", "",
        f"Для 384 заранее перечисленных обучаемых model/timeframe/asset сравнений рассчитана совместная односторонняя centered block-bootstrap max-t поправка. Primary critical={result['family_adjustment']['primary']['critical_max_t']:.4f}, audit={result['family_adjustment']['audit']['critical_max_t']:.4f}. Положительная нижняя граница добавки к фиксированной random-монете сохраняется в обоих banks для **{len(result['paired_random_positive_in_both_banks'])}** сочетаний. Это проверка слабого преимущества над данным random-контролем, а не над лучшим простым правилом, не 80%-точность и не независимая валидация.", ""]
    if result["paired_random_positive_in_both_banks"]:
        lines += ["| Минут | Метод | Валюта | Accuracy ненулевых, % | Joint LCB primary, п.п. | Joint LCB audit, п.п. |", "|---:|---|---|---:|---:|---:|"]
        for v in result["paired_random_positive_in_both_banks"]:
            lines.append(f"| {v['timeframe_minutes']} | {NAMES[v['model']]} | {v['symbol']} | {pct(v['accuracy_nonflat'])} | {pct(v['simultaneous']['primary']['paired_random_lcb'])} | {pct(v['simultaneous']['audit']['paired_random_lcb'])} |")
    lines += ["", "## 2. Что именно считается", "",
        "В момент t известна завершённая свеча и её close C(t). Модель предсказывает цену в конце **следующей** свечи C(t+Δ), а не направление уже наблюдаемой свечи. Предсказание вверх, если p_up ≥ 0,5, иначе вниз. Для random используется отдельный Bernoulli(0,5) бросок на каждом времени и активе.", "",
        "Верный ответ: (прогноз вверх и C(t+Δ)>C(t)) ИЛИ (прогноз вниз и C(t+Δ)<C(t)). Равенство цен не относится ни к росту, ни к падению.", "",
        "1. **Accuracy по всем интервалам = верных / всех интервалов.** Нулевое изменение считается неугаданным при обоих бинарных ответах. Это строгая трактовка запрошенной доли таймфреймов.",
        "2. **Accuracy ненулевых = верных / (ростов + падений).** Это ответ на вопрос, насколько верно определяется знак, когда цена действительно изменилась. Исключение нулей явно указано и не зависит от модели.",
        "3. **Balanced accuracy = (recall роста + recall падения)/2.** Она помогает заметить простое предпочтение одного класса.", "",
        "Все прогнозы выдаются на каждой точке, coverage=100%; confidence-отбор и abstention не используются. Денежный эффект не входит в ranking. ROC/доходность не подменяют процент угаданного направления.", "",
        "### Почему random не всегда ровно 50%", "",
        "У настоящего случайного броска 50% — математическое ожидание accuracy среди ненулевых исходов, а не обязательное эмпирическое значение. На коротком 4h-ряду отклонения сильнее. Среди **всех** интервалов ожидание равно 0,5 × доля ненулевых, поскольку бинарный ответ не угадывает flat. Сырые coin-ответы сохранены, seed=50521; SeedSequence отдельно зависит от таймфрейма и номера валюты. Лучший random seed не выбирался.", "",
        "## 3. Данные и честность временного расчёта", "",
        f"Фиксированный протокол, SHA исходников, данных и folds: [registration.json]({(b.OUT / 'registration.json').as_posix()}). Проверка исходных файлов и минутного ряда: [minute_data_audit.json]({(b.OUT / 'minute_data_audit.json').as_posix()}).", "",
        "Период оценки: 2026-08-01 00:00 UTC — 2026-10-01 00:00 UTC, конец исключён. Источники: 15 локальных минутных Parquet, 148 320 полных минутных свечей на актив с 20 июня до конца сентября; 8 активов. Пропусков и дублей нет, OHLCV проверки пройдены. Новые скачивания, сеть, торговый runtime и collector не запускались.", "",
        "Минутные open timestamp преобразуются в время доступности как open+1 минута. Свечи 10/30/240 минут собираются из полных минутных блоков с UTC-якорем: open первой, max high, min low, close последней, сумма volume. 4 944 исторических четырёхчасовых close точно совпали с прежним frozen bank. Не выполнялась новая независимая биржевая сверка, поэтому согласованность источников не доказывает истинность всех исходных цен.", "",
        "Обучение: rolling 30 календарных дней; каждые 3 дня, всего 21 origin на таймфрейм; только созревшие next-bar labels; purge 7 **native** баров. Максимум 4 000 времён обучения, выбранных равномерно без возвращения и без использования labels, все 8 активов сохранены для каждого времени. Если кандидатов меньше, используются все. Обе стороны исходов имеют uniform вес; точные нули исключены только из бинарного обучения.", "",
        "Scaler обучается заново только на выбранных ненулевых train-строках. Model SHA, training-data SHA, индексы folds, cutoff и sampling banks зафиксированы. Никакого random train/test split, перемешивания будущего в scaler или подбора параметров по августу–сентябрю нет.", "",
        "**Ограничение сравнения:** это не чистая абляция одного горизонта. На разных таймфреймах различается календарная длина native lookback-признаков, число доступных train-примеров и purge в минутах. Для скорости применён train sample cap; не использованы все минутные train-строки. Новый 4h-контроль переобучен на общем 30d-протоколе, поэтому его числа не следует выдавать за улучшение прежнего expanding 4h-experiment.", "",
        "Модели pooled по восьми активам, но **все метрики считаются отдельно для каждой валюты**. Обучение отдельной модели на каждой валюте — другой эксперимент, не выполненный здесь. Август–сентябрь уже раскрыты; текущий replay не является новым sealed interval.", "",
        f"SHA регистрации: `{reg['sha256']}`. Append-only objective registry v12 сохраняет v11; полная адаптивная история предыдущих исследований не объявляется восстановленной.", "",
        "## 4. Методы", "",
        "Использованы прежние фиксированные гиперпараметры, без нового tuning: Ridge6 (7 параметров, L2=0,01); Logistic6 (7 параметров, C=0,1); Logistic12 (13 параметров); GaussianNB; shrinkage LDA; regularized QDA; Tree depth≤3; RandomForest и ExtraTrees по 24 дерева depth≤4; HistBoost 32 итерации и ≤7 листьев; SGD Logistic alpha=0,001; MLP 12→8→1, 113 параметров, максимум 40 эпох. Деревья: минимум 128 строк в листе.", "",
        "Первые 6 признаков: distance EMA48, return18, efficiency18, relative return18 к группе активов, volatility48, breadth EMA48. Дополнительные 6: return1, тело, диапазон, верхняя и нижняя тени, log-volume относительно предыдущих 42 баров. Окна означают число завершённых **native** свечей. Текущие high/low/volume не используются до закрытия соответствующей свечи.", "",
        "Ridge обучается численной next-bar доходности, направление выводится по знаку; остальные trainable модели обучают бинарный знак. Controls: настоящая random-монета; train-majority; asset-train-majority; продолжение последнего знака; его инверсия; EMA-direction. Последние три — простые причинные правила, а не обучаемые нейросети. При ровном предыдущем баре persistence выбирает вверх, reverse — вниз; это заранее фиксированное tie-rule.", "",
        f"Предупреждения fit сохранены: `{warnings}`. Лимиты не расширялись после результатов. MLP с ограниченным числом эпох не доказывает пределы всех нейросетей; гиперпараметры также не являются оптимальными для каждого timeframe.", "",
        "## 5. Полные таблицы по всем валютам и таймфреймам", "",
        "Первой идёт запрошенная строгая accuracy по всем интервалам, затем accuracy ненулевых. В отдельных TSV/JSON дополнительно есть balanced accuracy, recall обоих направлений, матрицы ошибок, доверительные интервалы и месяцы. Никакое среднее по активам не заменяет эти таблицы.", ""]
    for tf in b.TIMEFRAMES:
        lines += [f"### {tf} минут: число исходов", "", "| Валюта | Всего | Рост | Падение | Ноль | Нулей, % | Random, все % | Random, ненулевые % |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for s in b.SYMBOLS:
            r = rows[tf, "random_coin", s]
            lines.append(f"| {s} | {r['n_all']} | {r['n_up']} | {r['n_down']} | {r['n_flat']} | {pct(r['flat_fraction'])} | {pct(r['accuracy_all'])} | {pct(r['accuracy_nonflat'])} |")
        for field, heading in (("accuracy_all", "Строгая accuracy по всем интервалам, %"), ("accuracy_nonflat", "Accuracy среди ненулевых движений, %")):
            lines += ["", "#### " + heading, "", "| Метод | " + " | ".join(s.split("/")[0] for s in b.SYMBOLS) + " |", "|---|" + "---:|" * 8]
            for model in METHODS:
                lines.append("| " + NAMES[model] + " | " + " | ".join(pct(rows[tf, model, s][field]) for s in b.SYMBOLS) + " |")
        lines += ["", f"![Все методы и валюты на {tf} мин]({link(f'accuracy_{tf}m.png')})", ""]
    lines += ["## 6. Нулевые исходы, неопределённость и ограничения", "",
        "На минутном ADA 25,62% исходов — точный flat; для обязательного бинарного ответа потолок accuracy **по всем интервалам** равен 74,38%, даже у оракула. Для цели 80% нужно либо оценивать только ненулевые исходы, либо явно изменить задачу на три класса вверх/flat/вниз. Искусственно относить flat к падению ради повышения accuracy некорректно.", "",
        f"![Нулевые исходы по валютам]({link('flat_share_by_asset.png')})", "",
        "Confidence intervals: одинаковый circular block bank из 5 000 выборок, блок 3 UTC-дня, переносится весь день со всеми активами. Primary seed=48101, audit=48103. Каждое per-asset значение имеет свой знаменатель ненулевых; bootstrap повторяет отношение сумм correct/nonflat, а не среднее процентов с разными знаменателями. Audit seed не создаёт новые данные. Совместная поправка max-t учитывает текущие 384 model/timeframe/asset сравнения; это приближённый bootstrap-контроль, не гарантия при любых формах нестационарности и не коррекция всей истории исследования.", "",
        "Exact binomial null bands в JSON/метриках относятся только к действительно независимым random-coin броскам условно на фиксированные фактические исходы. Они не используются как iid-интервалы для обученных моделей. Матрицы ошибок учитывают нули отдельной колонкой; balanced accuracy рассчитывается по up/down, а не по flat.", "",
        "## 7. Почему алгоритмы остаются около 50%", "",
        "Проверки индексов, цен, классов и знаменателей не обнаружили вычислительной причины, которая принудительно сводит accuracy к 50%. У методов разные ответы, recall и bias; они не заменены одним и тем же массивом. Random использует отдельные iid-подобные PRNG броски и воспроизводим.", "",
        "Вероятные исследовательские ограничения: (а) текущие свечные признаки слабо связаны с следующим знаком; (б) короткие изменения содержат много микроструктурного шума и flat; (в) train sample cap и pooled обучение ограничивают разнообразие и asset-specific структуру; (г) 30d window и переобучение раз в 3 дня не гарантируют адаптацию к каждому режиму; (д) нейросеть и гиперпараметры ограничены. Это гипотезы, не установленные этим экспериментом причинные объяснения. Приписывать весь эффект одному фактору нельзя без контролируемой абляции.", "",
        "Слабое преимущество над random не означает полную бесполезность, но и не подтверждает практически сильный ценовой вектор. Для выбора модели нужен прирост над причинными majority/persistence/reversal controls, устойчивость по валютам/месяцам и новый unseen период. Одни сырые проценты на многократно раскрытых данных недостаточны.", "",
        "## 8. Следующие исследовательские шаги — только прогнозирование", "",
        "1. Согласовать primary: бинарный знак среди ненулевых или трёхклассовый up/flat/down. Сохранить строгий all-interval показатель рядом; без подмены знаменателя.",
        "2. Один ограниченный feature ablation на наиболее интересном коротком горизонте: одинаковые wall-clock lookbacks против native-bar окон. Отделить смену target от смены входов.",
        "3. Проверить pooled против per-asset обучения при одинаковых training dates и бюджете; сначала logistic и один nonlinear метод. Не запускать сразу ещё десятки архитектур.",
        "4. Оценить влияние train cap: зафиксированный парный sample-cap контроль, без подбора seed по accuracy. Проверить, улучшается ли результат от данных, а не только от числа параметров.",
        "5. Для новых типов информации отдельно проверить наличие исторических trades/order-flow/OI; свечи не содержат данных о конкретных игроках. Загрузка и сеть здесь не выполнялись и не разрешаются результатами отчёта.",
        "6. После фиксации target/features/training — независимый новый период с единственным final scoring по валютам и preregistered comparisons. Не добиваться 80% многократной подгонкой августа–сентября.", "",
        "## 9. Проверка и доказательные артефакты", "",
        f"Пройдены 14 чистых синтетических assertions: UTC-агрегация, causal future perturbation, rolling volatility против прямого окна, идеальный/инвертированный прогноз и отдельный flat, random coin. Независимо проверены все 1 008 сохранённых моделей и **{prediction_checks:,}** прогнозных значений; scaler и случайная train-подвыборка воспроизведены. Для linear/logistic вероятность также пересчитана отдельно через coefficients/intercept. 576 metric-строк и их месячные значения подтверждены независимым сопоставлением signed price difference с бинарным ответом; матрицы 2×3 совпали.", "",
        "Старый 4h-блок: 96 exact per-asset metric matches, 4 944 exact minute→4h close matches. Все данные и код заморожены hashes. Чужие изменения не удалялись. Использован только workspace Python, по одной короткой foreground-команде; pytest, collector, фоновые процессы и торговый runtime не запускались. Разрешения credentials/orders/paper/live/runtime/network/promotion остаются false.", "",
        f"[Полная таблица для Excel, UTF-8 BOM TSV, 1 728 строк × 30 колонок]({link('statistics_excel_ru.tsv')})", "",
        f"[Машинная статистика, CI, confusion и joint LCB]({link('comparison.json')}) · [Независимый аудит прежнего 4h]({link('previous_4h_independent_audit.json')})", "",
        "TSV использует отдельные ячейки, tab delimiter и десятичную запятую для русской локали Excel. Native XLSX не создавался: стандартный XLSX authoring workflow требует JS, а задано ограничение только на workspace Python. Структура TSV проверена повторным чтением, включая число строк/колонок и числовые значения. Месячные CI не вычислялись и оставлены пустыми; confidence intervals в файле относятся к суммарному периоду.", ""]
    with (b.OUT / "REPORT_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(lines))
    best_table = ["# Краткий итог: направление без торгового расчёта", "", "1 008 fit; 12 моделей + 6 controls; 1/10/30/240 минут; 8 валют; август–сентябрь 2026. Финансовых replay и сетевых запросов — 0.", "",
        "Расчёт направления проверен независимо. Прежние 4h-метрики совпали для всех 96 model/asset пар; ошибки сдвига target не обнаружено. Новые 576 per-asset результатов и 9 592 128 прогнозных значений проверены. Ни один вариант не дал 80%.", "",
        "Максимумы accuracy ненулевых по каждой валюте (описательный выбор из 12, не независимая оценка):", "",
        "| Валюта | 1 мин | 10 мин | 30 мин | 240 мин |", "|---|---:|---:|---:|---:|"]
    for s in b.SYMBOLS:
        best_table.append("| " + s + " | " + " | ".join(pct(next(r for r in result["descriptive_best_by_asset"] if r["symbol"] == s and r["timeframe_minutes"] == tf)["accuracy_nonflat"]) for tf in b.TIMEFRAMES) + " |")
    best_table += ["", "Random — воспроизводимая монета 50/50 на каждой валюте и времени. Среди всех интервалов её ожидаемая accuracy ниже 50%, если есть flat. У минутной ADA 25,62% flat: бинарный all-interval потолок 74,38%. Поэтому полный отчёт показывает обе версии accuracy, все методы, месяцы, counts, recall, CI и матрицы ошибок.", "",
        f"[Полный отчёт]({link('REPORT_RU.md')}) · [Полная таблица для Excel]({link('statistics_excel_ru.tsv')})", "",
        f"![Сравнение с random]({link('random_and_best_by_asset.png')})", ""]
    with (b.OUT / "SUMMARY_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(best_table))
    print({"reports": 2, "full_method_asset_tables": 8, "statistics_rows": 1728, "profit_calculated": False})


def verify_delivery():
    reg = b.registration()
    for tf in b.TIMEFRAMES:
        b.registration(tf)
        checks = b.sealed(b.tf_path(tf) / "independent_checks.json")
        assert checks["checker_pin"] == b.pin(b.ROOT / "tools/check_genetic_primus_direction_multiscale_v1.py")
    registry, previous = b.sealed(b.NEXT_REGISTRY), b.sealed(b.PREVIOUS_REGISTRY)
    assert registry["objectives"][:-1] == previous["objectives"]
    assert registry["objectives"][-1]["registration_sha256"] == reg["sha256"]
    table = b.sealed(b.OUT / "table_verification.json")
    assert table["file_pin"] == b.pin(b.OUT / "statistics_excel_ru.tsv")
    result = b.sealed(b.OUT / "comparison.json")
    rows = {(r["timeframe_minutes"], r["model"], r["symbol"]): r for r in result["rows"]}
    report_text = (b.OUT / "REPORT_RU.md").read_text(encoding="utf-8")
    # Check every displayed per-method metric cell against authoritative JSON.
    tf, field, cell_checks = None, None, 0
    inverse = {v: k for k, v in NAMES.items()}
    for line in report_text.splitlines():
        found = re.match(r"### (\d+) минут: число исходов", line)
        if found: tf = int(found.group(1)); field = None
        if line == "#### Строгая accuracy по всем интервалам, %": field = "accuracy_all"
        if line == "#### Accuracy среди ненулевых движений, %": field = "accuracy_nonflat"
        if field and line.startswith("| "):
            cells = [v.strip() for v in line.strip("|").split("|")]
            if len(cells) == 9 and cells[0] in inverse:
                for s, text in zip(b.SYMBOLS, cells[1:]):
                    assert text == pct(rows[tf, inverse[cells[0]], s][field])
                    cell_checks += 1
    assert cell_checks == 1152
    links = 0
    for name in ("REPORT_RU.md", "SUMMARY_RU.md"):
        for path in re.findall(r"\]\((E:/[^)]+)\)", (b.OUT / name).read_text(encoding="utf-8")):
            assert Path(path).is_file(); links += 1
    files = {p.relative_to(b.OUT).as_posix(): b.pin(p) for p in sorted(b.OUT.rglob("*")) if p.is_file()}
    result = b.write(b.OUT / "delivery_receipt.json", {"registration_sha256": reg["sha256"], "artifact_pins": files,
        "report_source_pin": b.pin(Path(__file__)), "report_metric_cells_checked": cell_checks, "local_links_checked": links,
        "table_rows": 1728, "table_columns": 30, "historical_fits": 1008, "financial_replays": 0,
        "random_primary_seed": 50521, "previous_4h_exact_metric_matches": 96,
        "registry_immutable_predecessor_preserved": True, "safety": reg["safety"], "independent_validation": False})
    print({"artifact_pins": len(files), "report_cells_checked": cell_checks, "table_verified": True, "financial_replays": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["stats", "tables", "figures", "report", "verify"])
    stage = parser.parse_args().stage
    if stage == "stats": stats()
    elif stage == "tables": export_tables()
    elif stage == "figures": figures()
    elif stage == "report": report()
    else: verify_delivery()
