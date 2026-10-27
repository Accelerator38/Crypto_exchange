"""Derived paired accuracy analysis and reporting; never trains a model."""
from __future__ import annotations
import run_genetic_primus_direction_specialization_v1 as s
import argparse
import csv
import re
from pathlib import Path
import numpy as np

b, OUT = s.b, s.OUT
NAMES = {"logistic12": "Logistic12", "forest12": "RandomForest"}
CONTROL_NAMES = {"random_coin": "Random 50/50", "asset_train_majority": "Train-majority валюты",
                 "reverse_last_bar": "Против последнего знака"}
PERIODS = ("full", "2026-08", "2026-09")


def complete_registration():
    reg = s.registration()
    for name, expected in reg["parent_pins"].items():
        if b.pin(b.ROOT / name) != expected: raise ValueError("parent artifact changed: " + name)
    return reg


def collect(name, reg):
    parts, times = [], []
    for i in range(21):
        receipt = b.sealed(OUT / f"origin_{i:02d}_receipt.json")
        audit = b.sealed(OUT / f"origin_{i:02d}_audit.json")
        columns, clock = [], None
        for a in range(8):
            base = s.prefix(i, a, name)
            meta = b.sealed(base.with_suffix(".json"))
            if receipt["fits"][f"{a}:{name}"]["sha256"] != meta["sha256"]:
                raise ValueError("receipt/metadata mismatch")
            if audit["entries"][f"{a}:{name}"]["metadata_sha256"] != meta["sha256"]:
                raise ValueError("audit/metadata mismatch")
            if meta["registration_sha256"] != reg["sha256"]: raise ValueError("wrong fit registration")
            if b.pin(base.with_suffix(".npz")) != meta["prediction_pin"] or b.pin(base.with_suffix(".joblib")) != meta["model_pin"]:
                raise ValueError("audited model/prediction changed")
            with np.load(base.with_suffix(".npz"), allow_pickle=False) as data:
                if clock is None: clock = data["times"]
                elif not np.array_equal(clock, data["times"]): raise ValueError("asset clocks differ")
                columns.append(data["probability"])
        times.append(clock); parts.append(np.column_stack(columns))
    clocks, values = np.concatenate(times), np.concatenate(parts)
    if not np.array_equal(clocks, np.arange(b.START, b.END, s.TF * b.d.MINUTE)):
        raise ValueError("complete common grid required")
    return clocks, values


def independently_checked_metric(pred, current, future):
    result = b.d.metrics(pred, current, future)
    actual = np.sign(future - current).astype(np.int8)
    guessed = np.where(pred, 1, -1)
    correct = guessed == actual
    count = np.bincount(np.where(actual == 1, 0, np.where(actual == -1, 1, 2)) * 2 + (~pred).astype(int), minlength=6)
    expected_confusion = dict(zip(("up_up", "up_down", "down_up", "down_down", "flat_up", "flat_down"), count.tolist()))
    if result["confusion"] != expected_confusion or result["n_correct"] != int(correct.sum()):
        raise ValueError("independent confusion mismatch")
    if result["accuracy_all"] != float(correct.mean()): raise ValueError("independent denominator mismatch")
    if result["accuracy_nonflat"] != float(correct.sum() / np.count_nonzero(actual)):
        raise ValueError("independent nonflat denominator mismatch")
    return result, correct


def score():
    reg = complete_registration()
    bank = b.load_bank(s.TF)
    ix = np.flatnonzero((bank["times"][:-1] >= b.START) & (bank["times"][:-1] < b.END))
    times = np.asarray(bank["times"][ix])
    current, future = np.asarray(bank["closes"][ix]), np.asarray(bank["closes"][ix + 1])
    valid = future != current
    day = ((times - b.START) // 86_400_000).astype(int)
    with np.load(b.OUT / "resample_bank.npz", allow_pickle=False) as data:
        banks = {k: data[k] for k in data.files}
    masks = {"full": np.ones(len(times), bool), "2026-08": day < 31, "2026-09": day >= 31}
    rows, pairs, raw, fit_rows, warnings = [], [], {}, [], []
    for i in range(21):
        for a in range(8):
            for name in s.MODELS:
                meta = b.sealed(s.prefix(i, a, name).with_suffix(".json"))
                fit_rows.append({"origin": i, "symbol": b.SYMBOLS[a], "model": name,
                                 "training_rows": meta["info"]["rows"], "latest_label_ms": meta["fold"]["latest_label_ms"],
                                 "cutoff_ms": meta["fold"]["cutoff_ms"]})
                warnings.extend(meta["info"].get("warnings", []))
    for name in s.MODELS:
        clocks, new = collect(name, reg)
        parent_clocks, pooled = b.predictions(s.TF, name)
        if not np.array_equal(times, clocks) or not np.array_equal(times, parent_clocks): raise ValueError("target clock mismatch")
        for a, symbol in enumerate(b.SYMBOLS):
            correctness = {}
            for arm, probability in (("pooled", pooled[:, a]), ("asset", new[:, a])):
                pred = probability >= 0.5
                for period, mask in masks.items():
                    metric, correct = independently_checked_metric(pred[mask], current[mask, a], future[mask, a])
                    rows.append({"model": name, "arm": arm, "symbol": symbol, "period": period, **metric})
                    if period == "full": correctness[arm] = correct
            draws = s.paired_draws(correctness["asset"], correctness["pooled"], valid[:, a], day, banks)
            delta = correctness["asset"].astype(int) - correctness["pooled"].astype(int)
            pair = {"model": name, "symbol": symbol, "delta_all": float(delta.mean()),
                    "delta_nonflat": float(delta.sum() / valid[:, a].sum()),
                    "decisions_gained": int((delta > 0).sum()), "decisions_lost": int((delta < 0).sum()),
                    "prediction_disagreement": float(np.mean((new[:, a] >= 0.5) != (pooled[:, a] >= 0.5))),
                    "monthly_delta_all": {p: float(delta[mask].mean()) for p, mask in masks.items() if p != "full"},
                    "bootstrap": {k: {metric: {"ci95": np.quantile(values, [.025, .975]).tolist(),
                                              "lcb95": float(np.quantile(values, .05))}
                                       for metric, values in sample.items()} for k, sample in draws.items()}}
            pairs.append(pair)
            for key, sample in draws.items(): raw[f"{name}_{a}_{key}"] = sample["all"]
    for name in CONTROL_NAMES:
        clocks, values = b.predictions(s.TF, name)
        if not np.array_equal(clocks, times): raise ValueError("control clock mismatch")
        for a, symbol in enumerate(b.SYMBOLS):
            for period, mask in masks.items():
                metric, _ = independently_checked_metric(values[mask, a] >= .5, current[mask, a], future[mask, a])
                rows.append({"model": name, "arm": "control", "symbol": symbol, "period": period, **metric})
    family = {}
    points = np.array([p["delta_all"] for p in pairs])
    for key in banks:
        draws = np.stack([raw[f"{p['model']}_{b.SYMBOLS.index(p['symbol'])}_{key}"] for p in pairs])
        se = draws.std(axis=1, ddof=1)
        if np.any(se <= 0): raise ValueError("degenerate bootstrap contrast")
        critical = float(np.quantile(((draws - points[:, None]) / se[:, None]).max(axis=0), .95))
        family[key] = {"critical_max_t": critical, "family_size": 16, "method": "approximate centered max-t, one-sided alpha .05"}
        for i, pair in enumerate(pairs):
            pair.setdefault("simultaneous_lcb95_all", {})[key] = float(points[i] - critical * se[i])
    positive = [{"model": p["model"], "symbol": p["symbol"]} for p in pairs if min(p["simultaneous_lcb95_all"].values()) > 0]
    # Reused pooled rows must exactly reproduce their frozen previous measurement.
    previous = b.sealed(b.tf_path(s.TF) / "scores.json")["rows"]
    for row in rows:
        if row["arm"] == "pooled" and row["period"] == "full":
            old = next(x for x in previous if x["model"] == row["model"] and x["symbol"] == row["symbol"])
            if old["accuracy_all"] != row["accuracy_all"]: raise ValueError("old pooled metric changed")
    if len(rows) != 168 or len(pairs) != 16 or len(fit_rows) != 336: raise ValueError("bounded counts failed")
    with (OUT / "paired_draws.npz").open("xb") as stream: np.savez_compressed(stream, **raw)
    b.write(OUT / "comparison.json", {"registration_sha256": reg["sha256"], "rows": rows, "pairs": pairs,
        "family_adjustment": family, "positive_both_banks": positive, "training_rows": fit_rows,
        "fit_warning_count": len(warnings), "fit_warnings": sorted(set(warnings)), "new_fits": 336,
        "reused_pooled_fits": 42, "financial_replays": 0, "safety": reg["safety"]})
    print({"metric_rows": 168, "paired_comparisons": 16, "positive_both_banks": len(positive), "new_fits": 0})


def number(value, digits=10):
    if value is None: return ""
    if isinstance(value, (int, np.integer)): return str(value)
    return f"{value:.{digits}f}".replace(".", ",")


def tables():
    result = b.sealed(OUT / "comparison.json")
    pairs = {(p["model"], p["symbol"]): p for p in result["pairs"]}
    header = ["Период", "Валюта", "Метод", "Обучение", "Интервалы", "Ненулевые", "Рост", "Падение", "Flat", "Верно",
              "Точность_все_%", "Точность_ненулевые_%", "Balanced_accuracy_%", "Recall_рост_%", "Recall_падение_%", "Прогноз_вверх_%",
              "Добавка_к_общей_пп", "Парный_CI95_низ_пп", "Парный_CI95_верх_пп", "Семейный_LCB_primary_пп", "Семейный_LCB_audit_пп",
              "Рост_вверх", "Рост_вниз", "Падение_вверх", "Падение_вниз", "Flat_вверх", "Flat_вниз"]
    path = OUT / "statistics_excel_ru.tsv"
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t"); writer.writerow(header)
        for row in result["rows"]:
            pair = pairs.get((row["model"], row["symbol"]))
            extras = [None] * 5
            if row["arm"] == "asset" and row["period"] == "full":
                ci = pair["bootstrap"]["primary"]["all"]["ci95"]
                extras = [pair["delta_all"], *ci, pair["simultaneous_lcb95_all"]["primary"], pair["simultaneous_lcb95_all"]["audit"]]
            writer.writerow([row["period"], row["symbol"], NAMES.get(row["model"], CONTROL_NAMES.get(row["model"])), row["arm"],
                *[row[k] for k in ("n_all", "n_nonflat", "n_up", "n_down", "n_flat", "n_correct")],
                *[number(None if row[k] is None else 100 * row[k]) for k in
                  ("accuracy_all", "accuracy_nonflat", "balanced_accuracy", "up_recall", "down_recall", "predicted_up_fraction")],
                *[number(None if v is None else 100 * v) for v in extras],
                *[row["confusion"][k] for k in ("up_up", "up_down", "down_up", "down_down", "flat_up", "flat_down")]])
    with path.open(encoding="utf-8-sig", newline="") as stream: parsed = list(csv.reader(stream, delimiter="\t"))
    if len(parsed) != 169 or any(len(row) != 27 for row in parsed): raise ValueError("Excel cells/rows failed")
    for values, row in zip(parsed[1:], result["rows"]):
        if int(values[4]) != row["n_all"] or abs(float(values[10].replace(",", ".")) - row["accuracy_all"] * 100) > 1e-9:
            raise ValueError("Excel numeric value mismatch")
    b.write(OUT / "table_verification.json", {"rows": 168, "columns": 27, "table_pin": b.pin(path),
        "encoding": "utf-8-sig", "delimiter": "tab", "decimal": "comma", "separate_cells": True})
    print({"table_rows": 168, "columns": 27, "numeric_cells_verified": 336})


def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    result = b.sealed(OUT / "comparison.json")
    rows = {(r["model"], r["symbol"], r["arm"]): r for r in result["rows"] if r["period"] == "full"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    x = np.arange(8)
    for ax, model in zip(axes, s.MODELS):
        for shift, arm, label in ((-.19, "pooled", "Общая модель"), (.19, "asset", "Отдельная модель")):
            values = [100 * rows[(model, sym, arm)]["accuracy_all"] for sym in b.SYMBOLS]
            ax.bar(x + shift, values, width=.36, label=label)
        ax.scatter(x, [100 * rows[("random_coin", sym, "control")]["accuracy_all"] for sym in b.SYMBOLS],
                   color="black", marker="_", s=140, label="Random 50/50")
        ax.set(title=NAMES[model], xticks=x, xticklabels=[z.split('/')[0] for z in b.SYMBOLS], ylim=(0, 100), ylabel="Верное направление / все интервалы, %")
        ax.grid(axis="y", alpha=.2); ax.legend(fontsize=9)
    fig.suptitle("10 минут: общая модель против отдельных по валютам\nАвгуст–сентябрь 2026; flat считается неверным бинарным ответом")
    fig.tight_layout(); fig.savefig(OUT / "accuracy_by_asset.png", dpi=140); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, model in zip(axes, s.MODELS):
        pairs = [next(p for p in result["pairs"] if p["model"] == model and p["symbol"] == sym) for sym in b.SYMBOLS]
        point = np.array([100 * p["delta_all"] for p in pairs])
        ci = np.array([p["bootstrap"]["primary"]["all"]["ci95"] for p in pairs]) * 100
        ax.errorbar(point, x, xerr=np.array([point - ci[:, 0], ci[:, 1] - point]), fmt="o", capsize=4)
        ax.axvline(0, color="black", linestyle="--"); ax.grid(axis="x", alpha=.2)
        ax.set(title=NAMES[model], yticks=x, yticklabels=[z.split('/')[0] for z in b.SYMBOLS], xlabel="Отдельная − общая accuracy, процентные пункты")
    axes[0].invert_yaxis()
    fig.suptitle("Парная добавка: один и тот же прогнозный интервал\nПолосы: обычный 95% block-bootstrap CI; поправка на 16 сравнений — в отчёте")
    fig.tight_layout(); fig.savefig(OUT / "paired_accuracy_delta.png", dpi=140); plt.close(fig)
    pairs = result["pairs"]
    values = np.array([[100 * p["monthly_delta_all"][month] for month in ("2026-08", "2026-09")] for p in pairs])
    limit = max(1., float(np.max(np.abs(values))))
    fig, ax = plt.subplots(figsize=(8, 9))
    handle = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
    ax.set(xticks=[0, 1], xticklabels=["Август", "Сентябрь"], yticks=np.arange(16),
           yticklabels=[p["symbol"].split('/')[0] + " — " + NAMES[p["model"]] for p in pairs],
           title="Сохраняется ли добавка между месяцами?\nОтдельная − общая accuracy по всем интервалам, п.п.")
    for i in range(16):
        for j in range(2): ax.text(j, i, f"{values[i, j]:+.2f}", ha="center", va="center", fontsize=10)
    fig.colorbar(handle, ax=ax, label="Добавка, п.п.")
    fig.tight_layout(); fig.savefig(OUT / "monthly_accuracy_delta.png", dpi=140); plt.close(fig)
    print({"charts": 3, "financial_charts": 0})


def pct(value): return f"{100 * value:.2f}".replace(".", ",")


def comparison_lines(result, field):
    rows = {(r["model"], r["symbol"], r["arm"]): r for r in result["rows"] if r["period"] == "full"}
    delta = "delta_all" if field == "accuracy_all" else "delta_nonflat"
    lines = ["| Валюта | Метод | Общая, % | Отдельная, % | Добавка, п.п. |", "|---|---|---:|---:|---:|"]
    for p in result["pairs"]:
        pooled, asset = rows[(p["model"], p["symbol"], "pooled")], rows[(p["model"], p["symbol"], "asset")]
        lines.append(f"| {p['symbol']} | {NAMES[p['model']]} | {pct(pooled[field])} | {pct(asset[field])} | {pct(p[delta])} |")
    return lines


def report():
    result = b.sealed(OUT / "comparison.json")
    def link(name): return (OUT / name).as_posix()
    new_rows = [r for r in result["rows"] if r["period"] == "full" and r["arm"] == "asset"]
    wins = sum(p["delta_all"] > 0 for p in result["pairs"])
    stable = sum(min(p["monthly_delta_all"].values()) > 0 for p in result["pairs"])
    best = max(new_rows, key=lambda r: r["accuracy_nonflat"])
    positive = result["positive_both_banks"]
    lines = ["# Общая модель против отдельных моделей валют: прогноз на 10 минут", "",
        "Отчёт от 12 октября 2026. Только направление close → следующий close. Прибыль и исполнение не рассчитывались.", "",
        "## 1. Результат и область применимости", "",
        f"Выполнены 336 новых fit: 2 метода × 8 валют × 21 origin. Повторно использованы 42 прежних pooled fit, без нового обучения общих моделей. Отдельное обучение дало положительную описательную добавку в {wins} из 16 пар; в обоих календарных месяцах — в {stable} из 16. После приблизительной семейной поправки на 16 сравнений нижняя граница добавки положительна на обоих bootstrap banks у {len(positive)} пар.", "",
        f"Максимальная accuracy ненулевых среди отдельных моделей: {pct(best['accuracy_nonflat'])}% — {NAMES[best['model']]} {best['symbol']}; accuracy по всем интервалам у неё {pct(best['accuracy_all'])}%. 80% не достигнуто. Это описательный максимум, не независимая оценка выбранного победителя.", "",
        "Август–сентябрь уже раскрыты. Это адаптивное исследовательское продолжение предыдущего эксперимента, не новый outer/prospective. Audit seed bank меняет bootstrap-пересэмплирование, но не добавляет новых данных. Поправка на текущие 16 пар не компенсирует всю историю исследовательских попыток.", "",
        "## 2. Данные, контракт и вычислительный путь", "",
        f"[Контракт]({s.CONFIG.as_posix()}) · [Регистрация и hashes]({link('registration.json')}) · [Предыдущие данные и протокол]({(b.OUT / 'registration.json').as_posix()}) · [Реестр objectives v13]({s.REGISTRY.as_posix()}).", "",
        "Используется неизменённый 10-минутный банк из 15 локальных 1m OHLCV Parquet: ADA, BNB, BTC, DOGE, ETH, LINK, SOL, XRP против USDT. Оценка 2026-08-01 00:00 UTC — 2026-10-01 00:00 UTC, конец исключён. 8 784 интервала на валюту; август 4 464, сентябрь 4 320. Новых скачиваний и сетевых запросов нет.", "",
        "Сохранены 12 OHLCV-признаков, горизонт 10 минут, порог 0,5, окно train 30 дней, retrain каждые 3 дня, purge 7 native bars (70 минут), прежние train/test time indices и зрелые labels. На каждом origin scaler обучается только на ненулевых train labels; у pooled scaler общий, у asset — отдельный. Нулевая train-доходность исключается одинаково, но все flat тестового периода остаются в основном знаменателе.", "",
        "Logistic12: 13 параметров, C=0,1, max_iter=120, tol=1e-6. RandomForest: 24 дерева, max_depth=4, min_samples_leaf=128, max_features=0,75, n_jobs=1. Seed=20261012. Параметры точно совпадают с прежними общими моделями; поиска и подбора seed нет.", "",
        "Одинаковые даты не означают одинаковое число строк: pooled получает данные всех восьми валют, asset — одной. Восемь asset fit также дороже одного pooled fit. Фиксированный абсолютный min_samples_leaf=128 задаёт более сильную относительную регуляризацию на меньшей выборке. Эксперимент измеряет весь эффект такого разделения, включая отдельный scaler и размер выборки; он не изолирует чистую специализацию от этих составляющих.", "",
        "### Объём обучения по валютам", "", "| Валюта | Минимум nonflat train строк | Максимум |", "|---|---:|---:|"]
    for symbol in b.SYMBOLS:
        sizes = [r["training_rows"] for r in result["training_rows"] if r["symbol"] == symbol]
        lines.append(f"| {symbol} | {min(sizes)} | {max(sizes)} |")
    lines += ["", "## 3. Определение метрик", "",
        "Верно: прогноз вверх при следующем close > текущего, либо вниз при следующем close < текущего. Flat не угадан ни одним бинарным ответом. Accuracy all = correct / all; accuracy nonflat = correct / (up + down). Balanced accuracy = (recall up + recall down) / 2. Coverage=100%, abstention отсутствует. Random — прежняя фиксированная монета 50/50, без выбора удачного seed.", "",
        "Парная добавка = accuracy asset − accuracy pooled на одних и тех же датах одной валюты. Bootstrap: 5 000 общих круговых календарных 3-day block samples из 61 дня, плюс отдельный audit bank. Для семейного one-sided LCB используется point − q95(max centered standardized bootstrap errors) × bootstrap SE, общая семья 16 пар. Это приблизительный block-bootstrap контроль текущей семьи, не точная конечновыборочная гарантия.", "",
        "## 4. Все результаты сравнения", "", "### Accuracy по всем интервалам — основной показатель", ""]
    lines += comparison_lines(result, "accuracy_all")
    lines += ["", "### Accuracy среди ненулевых движений — дополнительный показатель", ""]
    lines += comparison_lines(result, "accuracy_nonflat")
    lines += ["", "### Парная устойчивость по основному показателю", "",
        "| Валюта | Метод | CI95 добавки, п.п. | Family LCB primary, п.п. | Family LCB audit, п.п. | Исправлено / испорчено прогнозов |", "|---|---|---:|---:|---:|---:|"]
    for p in result["pairs"]:
        lo, hi = p["bootstrap"]["primary"]["all"]["ci95"]
        lines.append(f"| {p['symbol']} | {NAMES[p['model']]} | [{pct(lo)}; {pct(hi)}] | {pct(p['simultaneous_lcb95_all']['primary'])} | {pct(p['simultaneous_lcb95_all']['audit'])} | {p['decisions_gained']} / {p['decisions_lost']} |")
    lines += ["", "### Добавка по календарным месяцам", "", "| Валюта | Метод | Август, п.п. | Сентябрь, п.п. |", "|---|---|---:|---:|"]
    for p in result["pairs"]:
        lines.append(f"| {p['symbol']} | {NAMES[p['model']]} | {pct(p['monthly_delta_all']['2026-08'])} | {pct(p['monthly_delta_all']['2026-09'])} |")
    lines += ["", "### Простые контроли: accuracy по всем интервалам, %", "", "| Валюта | Random 50/50 | Train-majority валюты | Против последнего знака |", "|---|---:|---:|---:|"]
    for symbol in b.SYMBOLS:
        values = [next(r["accuracy_all"] for r in result["rows"] if r["symbol"] == symbol and r["model"] == model and r["period"] == "full") for model in CONTROL_NAMES]
        lines.append("| " + symbol + " | " + " | ".join(pct(v) for v in values) + " |")
    lines += ["", "## 5. Графики", "",
        "Сопоставление на единой шкале: рост сложности сам по себе не означает рост точности.", "",
        f"![Accuracy по валютам]({link('accuracy_by_asset.png')})", "",
        "Парная разница показывает размер эффекта и обычный CI; семейные LCB приведены выше.", "",
        f"![Парная добавка]({link('paired_accuracy_delta.png')})", "",
        "Разные знаки между месяцами означают отсутствие стабильной положительной описательной добавки на обоих блоках.", "",
        f"![Добавка по месяцам]({link('monthly_accuracy_delta.png')})", "",
        "## 6. Проверки, ограничения и дальнейший шаг", "",
        "Компиляция и 6 прямых synthetic checks выполнены до обучения; pytest/collector не запускались. Для всех 336 новых моделей проверены SHA модели и прогнозов, train-only scaler, train hashes, зрелость labels/purge, восстановление прогнозов из joblib и совпадение ценовых labels. Logistic probabilities дополнительно вычислены прямо из coefficients/intercept. Для pooled восстановление допускает численную погрешность ≤1e-14 из-за размера BLAS batch; бинарные решения должны совпасть точно. Проверены 281 088 new+pooled probability cells.", "",
        f"Метрики всех 168 строк проверены вторым расчётом через знак изменения цены и матрицу 3 фактических × 2 прогнозных классов. Сохранённые pooled full-period accuracy точно совпали с предыдущим расчётом. Warnings обучения: {result['fit_warning_count']}. Таблица TSV проверяется повторным чтением: 168 строк, 27 колонок, UTF-8 BOM, decimal comma, отдельные поля.", "",
        "Если семейные LCB не положительны, нет основания объявлять отдельное обучение улучшением. Даже положительный текущий LCB — лишь исследовательская зацепка: следующая проверка должна удерживать выборку и даты одинаковыми и проверять эффект ограничения числа train строк, а не сразу расширять архитектуры. После этого — одинаковые wall-clock окна признаков. При появлении устойчивого сигнала его потребуется проверить на действительно новых данных.", "",
        "80–90% не обещается. Этот опыт не доказывает ни универсальную бесполезность векторов, ни возможность добиться такого уровня на OHLCV. Данные о сделках, стакане и конкретных игроках сюда не включались. Никакие торговые разрешения не изменены: credentials/orders/paper/live/runtime/network/promotion=false.", "",
        f"[Полная статистика для Excel]({link('statistics_excel_ru.tsv')}) · [Машинная статистика и bootstrap]({link('comparison.json')}) · [Проверка таблицы]({link('table_verification.json')})."]
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(lines) + "\n")
    summary = ["# Итог сравнения: общая модель против отдельных", "",
        f"10 минут, август–сентябрь 2026. 336 новых fit; 42 pooled fit повторно использованы. Положительная описательная добавка: {wins}/16; положительна в обоих месяцах: {stable}/16. Family LCB > 0 на обоих banks: {len(positive)}/16. Независимой валидацией этот раскрытый период не является.", ""]
    summary += comparison_lines(result, "accuracy_all")
    summary += ["", "Accuracy по всем интервалам; flat считается неверным. Полные nonflat/BA/recall/confusion/monthly/CI — в отчёте и TSV.", "",
                f"[Отчёт с графиками]({link('REPORT_RU.md')}) · [Таблица для Excel]({link('statistics_excel_ru.tsv')})", ""]
    with (OUT / "SUMMARY_RU.md").open("x", encoding="utf-8") as stream: stream.write("\n".join(summary))
    print({"reports": 2, "comparison_pairs": 16, "plots": 3, "fits": 0})


def verify():
    reg = complete_registration()
    result = b.sealed(OUT / "comparison.json")
    table = b.sealed(OUT / "table_verification.json")
    if b.pin(OUT / "statistics_excel_ru.tsv") != table["table_pin"]: raise ValueError("table changed")
    with (OUT / "statistics_excel_ru.tsv").open(encoding="utf-8-sig", newline="") as stream:
        parsed = list(csv.reader(stream, delimiter="\t"))
    if len(parsed) != 169 or any(len(r) != 27 for r in parsed): raise ValueError("table dimensions failed")
    full = (OUT / "REPORT_RU.md").read_text(encoding="utf-8")
    for field in ("accuracy_all", "accuracy_nonflat"):
        for line in comparison_lines(result, field)[2:]:
            if line not in full: raise ValueError("report numerical row mismatch")
    for p in result["pairs"]:
        line = f"| {p['symbol']} | {NAMES[p['model']]} | {pct(p['monthly_delta_all']['2026-08'])} | {pct(p['monthly_delta_all']['2026-09'])} |"
        if line not in full: raise ValueError("monthly report row mismatch")
    for name in ("REPORT_RU.md", "SUMMARY_RU.md"):
        text = (OUT / name).read_text(encoding="utf-8")
        for target in re.findall(r"\]\((E:/[^)]+)\)", text):
            if not Path(target).is_file(): raise ValueError("broken local link: " + target)
    registry = b.sealed(s.REGISTRY)
    previous = b.sealed(s.PREVIOUS_REGISTRY)
    if registry["objectives"][:-1] != previous["objectives"]: raise ValueError("registry history changed")
    audit_cells = sum(b.sealed(OUT / f"origin_{i:02d}_audit.json")["new_and_pooled_probability_cells"] for i in range(21))
    if audit_cells != 281088: raise ValueError("audit coverage failed")
    artifacts = {p.relative_to(OUT).as_posix(): b.pin(p) for p in sorted(OUT.rglob("*")) if p.is_file()}
    b.write(OUT / "delivery_receipt.json", {"registration_sha256": reg["sha256"], "artifact_pins": artifacts,
        "report_builder_pin": b.pin(Path(__file__)), "registry_pin": b.pin(s.REGISTRY),
        "new_fits": 336, "reused_pooled_fits": 42, "probability_cells_audited": audit_cells,
        "report_numeric_cells_checked": 128, "table_rows": 168, "table_columns": 27,
        "financial_replays": 0, "safety": reg["safety"]})
    print({"artifact_pins": len(artifacts), "report_numeric_cells_checked": 128, "table_verified": True, "financial_replays": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["score", "tables", "figures", "report", "verify"])
    args = parser.parse_args()
    {"score": score, "tables": tables, "figures": figures, "report": report, "verify": verify}[args.stage]()
