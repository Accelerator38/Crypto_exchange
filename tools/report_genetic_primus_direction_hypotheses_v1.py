"""Evidence tables, figures and inventory; no fitting and no model selection writes."""
from __future__ import annotations
import os
os.environ["MPLBACKEND"] = "Agg"
import csv
import importlib.metadata
import json
from pathlib import Path
import sys
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from run_genetic_primus_direction_selective_v1 import read, write, digest, identity
from run_genetic_primus_direction_hypotheses_v1 import registration

BASE = ROOT / "Reports/Exia/Genetic_Primus"
OUT = BASE / "direction_hypotheses_v1_20261013"
MICRO = BASE / "direction_micro_hypotheses_v1_20261013"
PARENT = BASE / "direction_selective_v1_20261013"
DESCRIPTIONS = {
    "lag_logit": "Лаги return/body/volume 1/2/3/6 баров, logistic (25 коэффициентов)",
    "lag_boost": "Те же 24 признака с ограниченным boosting",
    "context_logit": "Час/день/неделя, волатильность дня, рыночный контекст и время UTC (21 коэффициент)",
    "context_boost": "Нелинейные зависимости тех же контекстных признаков",
    "asset_intercepts": "Общие slopes и отдельные intercepts валют (20 коэффициентов)",
    "robust_logit": "Train-only winsorization 0.5/99.5 percentiles (13 коэффициентов)",
    "recency90": "Half-life 90d; веса и положительным, и отрицательным исходам (13 коэффициентов)",
    "mlp65": "MLP 6→8→1, 65 параметров, alpha=10, лимит 80 итераций",
    "temporal_ensemble3": "Среднее текущей и двух предыдущих causal logistic моделей; новых fit нет",
    "large_train_logit": "16000 времён обучения вместо 4000; исходная подвыборка сохранена (13 коэффициентов)",
    "large_train_boost": "Тот же расширенный train bank с boosting",
    "flat_aware_gate": "Отдельный move/flat classifier меняет только отказы; 26 коэффициентов с direction-parent"
}


def export_tsv(path, fields, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    with path.open(encoding="utf-8-sig", newline="") as f:
        restored = list(csv.DictReader(f, delimiter="\t"))
    if len(restored) != len(rows) or any(len(r) != len(fields) for r in restored):
        raise ValueError("Excel table cell/row verification failed")


def figures(rows, micro_rows, reg):
    import matplotlib.pyplot as plt
    models = [v["id"] for v in reg["contract"]["variants"]]
    symbols = reg["symbols"]
    lookup = {(r["minutes"], r["model"], r["symbol"], r["coverage_target"]): r for r in rows}
    values = [100*r["accuracy_all"] for r in rows if r["accuracy_all"] is not None]
    vmin, vmax = min(values), max(values)
    output = []
    for coverage in [1, .5, .2, .1]:
        fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), sharey=True, layout="constrained")
        for ax, minutes in zip(axes, [1, 10, 30]):
            data = np.full((len(models), 8), np.nan)
            for i, model in enumerate(models):
                for j, symbol in enumerate(symbols):
                    r = lookup.get((minutes, model, symbol, coverage))
                    if r is not None and r["accuracy_all"] is not None: data[i, j] = 100*r["accuracy_all"]
            im = ax.imshow(np.ma.masked_invalid(data), vmin=vmin, vmax=vmax, cmap="viridis", aspect="auto")
            for i, j in np.argwhere(np.isfinite(data)):
                ax.text(j, i, f"{data[i,j]:.1f}", ha="center", va="center", fontsize=7, color="white" if data[i,j] < (vmin+vmax)/2 else "black")
            ax.set_xticks(range(8), [s.split("/")[0] for s in symbols], rotation=45)
            ax.set_yticks(range(len(models)), models, fontsize=8)
            ax.set_title(f"{minutes} min | target coverage {coverage:.0%}")
        fig.colorbar(im, ax=axes, shrink=.7, label="Accuracy including flat, %")
        fig.suptitle("Revealed Aug-Sep 2026 | actual coverage and n in TSV | blank = not registered")
        name = f"accuracy_target_{int(100*coverage)}.png"; fig.savefig(OUT / name, dpi=140); plt.close(fig); output.append(name)
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), sharey=True, layout="constrained")
    limit = max(abs(100*r["comparisons"]["parent_policy"]["bootstrap"]["delta"]) for r in rows if r["coverage_target"] == 1)
    for ax, minutes in zip(axes, [1, 10, 30]):
        a = np.full((len(models), 8), np.nan)
        for i, model in enumerate(models):
            for j, symbol in enumerate(symbols):
                r = lookup.get((minutes, model, symbol, 1))
                if r is not None: a[i, j] = 100*r["comparisons"]["parent_policy"]["bootstrap"]["delta"]
        im = ax.imshow(np.ma.masked_invalid(a), vmin=-limit, vmax=limit, cmap="RdBu", aspect="auto")
        for i, j in np.argwhere(np.isfinite(a)):
            ax.text(j, i, f"{a[i,j]:+.2f}", ha="center", va="center", fontsize=7)
        ax.set_xticks(range(8), [s.split("/")[0] for s in symbols], rotation=45)
        ax.set_yticks(range(len(models)), models, fontsize=8); ax.set_title(f"{minutes} min | full coverage")
    fig.colorbar(im, ax=axes, shrink=.7, label="Accuracy delta vs matched parent, percentage points")
    fig.suptitle("Point estimates | no parent comparison passes simultaneous LCB in both banks")
    name = "full_coverage_delta.png"; fig.savefig(OUT / name, dpi=140); plt.close(fig); output.append(name)
    micro_models = [v["id"] for v in read(MICRO / "registration.json")["contract"]["models"]]
    fig, axes = plt.subplots(1, 3, figsize=(17, 8), sharey=True, sharex=True, layout="constrained")
    for ax, horizon in zip(axes, [1, 10, 30]):
        selected = [next(r for r in micro_rows if (r["horizon_minutes"], r["model"], r["coverage_target"]) == (horizon, model, 1)) for model in micro_models]
        values = [100*r["accuracy_all"] for r in selected]
        ax.barh(range(len(micro_models)), values, color=["grey" if m.startswith("reference") else "steelblue" for m in micro_models])
        for i, value in enumerate(values): ax.text(value+.15, i, f"{value:.2f}", va="center", fontsize=7)
        controls = selected[0]["controls"]
        for control, style in [("random_coin", ":"), ("train_majority", "--"), ("momentum", "-.")]:
            ax.axvline(100*controls[control]["accuracy_all"], ls=style, color="black", alpha=.5, label=control)
        ax.set_yticks(range(len(micro_models)), micro_models, fontsize=8)
        ax.set_title(f"h={horizon} min | n={selected[0]['n_emitted']}"); ax.set_xlim(30, 68)
    axes[0].invert_yaxis(); axes[2].legend(fontsize=7, loc="lower right")
    fig.supxlabel("Primary trade-price direction accuracy, %; including flat")
    fig.suptitle("BTC single-day engineering | full coverage | no independent significance claim")
    fig.savefig(MICRO / "full_coverage_comparison.png", dpi=140); plt.close(fig)
    return output


def main():
    reg = registration()
    result = read(OUT / "results.json"); rows = result["rows"]
    micro_rows = sum([read(MICRO / f"results_h{h}.json")["rows"] for h in [1, 10, 30]], [])
    receipts = [read(p) for p in sorted(OUT.glob("audit_tf_*m_origin_*.json"))]
    metadata = [read(p) for p in sorted(OUT.glob("tf_*m_origin_*.json"))]
    if len(rows) != 960 or len(micro_rows) != 216 or len(receipts) != 63 or len(metadata) != 630:
        raise ValueError("complete registered matrix required before reporting")
    if sum(a["forecast_artifacts"] for a in receipts) != 630 or sum(a["thresholds"] for a in receipts) != 20160:
        raise ValueError("audit count mismatch")
    if sum(a["fit_count"] for a in metadata) != 567 or read(MICRO / "audit.json")["models"] != 54:
        raise ValueError("fit budget mismatch")
    parent = read(PARENT / "results.json")["rows"]
    parent_lookup = {(r["timeframe_minutes"], r["model"], r["symbol"], r["coverage_target"]): r for r in parent}
    flat_table = []
    for r in rows:
        cell = dict(r); comparison = r["comparisons"]["parent_policy"]
        cell.update({"parent_accuracy": comparison["metrics"]["accuracy_all"], "parent_coverage": comparison["metrics"]["coverage_actual"],
            "delta_parent_pp": 100*comparison["bootstrap"]["delta"] if comparison["bootstrap"]["delta"] is not None else None,
            "simultaneous_lcb_pp": 100*comparison["bootstrap"]["simultaneous_lcb95"] if comparison["bootstrap"]["simultaneous_lcb95"] is not None else None,
            "audit_simultaneous_lcb_pp": 100*comparison["audit_bootstrap"]["simultaneous_lcb95"] if comparison["audit_bootstrap"]["simultaneous_lcb95"] is not None else None})
        flat_table.append(cell)
    fields = ["minutes", "model", "symbol", "coverage_target", "coverage_actual", "n_eligible", "n_emitted", "flat_emitted",
        "accuracy_all", "accuracy_nonflat", "balanced_accuracy", "parent_accuracy", "parent_coverage", "delta_parent_pp",
        "simultaneous_lcb_pp", "audit_simultaneous_lcb_pp", "research_positive_signal", "research_positive_policy"]
    export_tsv(OUT / "statistics_excel_ru.tsv", fields, flat_table)
    micro_table = [{**r, "mid_accuracy_all": r["midprice_diagnostic"]["accuracy_all"],
        "mid_accuracy_nonflat": r["midprice_diagnostic"]["accuracy_nonflat"],
        "mid_flat_emitted": r["midprice_diagnostic"]["flat_emitted"]} for r in micro_rows]
    export_tsv(MICRO / "statistics_excel_ru.tsv", ["horizon_minutes", "model", "features", "coverage_target", "coverage_actual",
        "n_eligible", "n_emitted", "flat_emitted", "accuracy_all", "accuracy_nonflat", "balanced_accuracy",
        "mid_accuracy_all", "mid_accuracy_nonflat", "mid_flat_emitted", "independent_improvement"], micro_table)
    summary = []
    for variant in reg["contract"]["variants"]:
        for minutes in variant["horizons"]:
            selected = [r for r in rows if r["model"] == variant["id"] and r["minutes"] == minutes and r["coverage_target"] == 1]
            delta = [100*r["comparisons"]["parent_policy"]["bootstrap"]["delta"] for r in selected]
            both_months = 0
            for r in selected:
                baseline = parent_lookup[(minutes, variant["parent"], r["symbol"], 1)]
                if all(r["months"][m]["accuracy_all"] > baseline["months"][m]["accuracy_all"] for m in ["2026-08", "2026-09"]):
                    both_months += 1
            summary.append({"model": variant["id"], "minutes": minutes, "median_delta_pp": float(np.median(delta)),
                "minimum_delta_pp": min(delta), "maximum_delta_pp": max(delta), "positive_assets": sum(v > 0 for v in delta),
                "positive_both_months_assets": both_months})
    warnings = [m for m in metadata if m["info"].get("warnings")]
    positives_parent = sum(all((r["comparisons"]["parent_policy"][b]["simultaneous_lcb95"] or 0) > 0 for b in ["bootstrap", "audit_bootstrap"]) for r in rows)
    output_figures = figures(rows, micro_rows, reg)
    report = ["# Проверка гипотез на новых прогнозных моделях — 13 октября 2026", "",
        "Выполнено 621 новое обучение: 567 в основном пакете и 54 в микроструктурном. Рассмотрены все 20 записей архивного журнала и последние direction-эксперименты. Новые модели покрывают 9 тем/12 конфигураций OHLCV и 7 дополнительных микроструктурных тем.",
        "Ни один из 960 основных вариантов asset/horizon/model/coverage не имеет положительного simultaneous paired LCB относительно прежней модели в обоих seed banks. Это отсутствие подтверждения при данном бюджете и периоде, не доказательство нулевой предсказуемости.", "",
        "## Контракт и данные", "",
        "Основной период: 01.08–30.09.2026, ранее раскрытые исследовательские данные. 8 валют, горизонты 1/10/30m, направление следующего полного бара, решение на каждой соответствующей сетке. Up/down/flat не заменяются медианой, dead zone или торговым сигналом. Flat сохранён и считается ошибкой бинарного прогноза в основной accuracy; nonflat accuracy показана отдельно.",
        "Train+calibration = 30 дней, последние 7 — calibration; purge 7 баров на обеих границах. Retrain каждые 3 дня, scaler/quantiles только train, labels созревшие. 4000 одинаковых времён train для обычных моделей; 16000 для зарегистрированной проверки объёма данных, с сохранением исходных 4000. Это увеличение данных, не смена train-window.",
        "Пороги 100/50/20/10% по прошлым confidence scores отдельно для каждой валюты и origin. Test coverage фактический; test quantiles не применяются. Для flat-aware gate направление ровно прежнее; меняется только оценка вероятности успеха отбора.",
        "Контроли: random, train-only majority, momentum. Parent_same_mask проверяет направление на той же новой маске; parent_policy сравнивает обе политики отбора при одинаковом целевом, но не обязательно одинаковом фактическом покрытии.",
        "Общие circular bootstrap блоки по 3 календарных дня, 1000 повторов, два seed bank; max-t поправка на все 4800 сравнений. Отдельный seed не является новыми данными. Историческое полное число адаптивных попыток неизвестно, что отмечено в registry; эти интервалы исследовательские.", "",
        "## Все новые основные конфигурации", "",
        "Δ ниже — медиана изменения full-coverage accuracy по восьми валютам, в процентных пунктах. Знак по месяцам показан как дополнительная описательная проверка, не независимая валидация.", "",
        "| Модель | Что проверено | Горизонт | Median Δ, п.п. | Положительных валют | Положительны оба месяца |", "|---|---|---:|---:|---:|---:|"]
    for a in summary:
        report.append(f"| {a['model']} | {DESCRIPTIONS[a['model']]} | {a['minutes']}m | {a['median_delta_pp']:+.3f} | {a['positive_assets']}/8 | {a['positive_both_months_assets']}/8 |")
    report += ["", f"Положительных simultaneous LCB против parent-policy в обоих банках: {positives_parent}/960. Положительных сигналов против parent_same_mask и всех контролей: {sum(r['research_positive_signal'] for r in rows)}/960. Положительных политик отбора против parent-policy и всех контролей: {sum(r['research_positive_policy'] for r in rows)}/960.",
        f"Предупреждения сходимости: {len(warnings)} forecast artifacts, все MLP65. Лимит 80 итераций недостаточен для критерия LBFGS convergence. Числа относятся к фиксированной ограниченной конфигурации; отсутствие выигрыша не отвергает нейросетевую архитектуру. По test-результату лимиты не увеличивались.",
        "Half-life 90d на 23-дневной части train меняет веса умеренно; это не тест всех возможных recency-окон. Kish ESS в metadata отражает неравномерность весов, а не число независимых наблюдений. Положительные и отрицательные обучающие исходы получают веса.",
        "Полное разделение моделей по валютам ранее ухудшало результаты; здесь проверены только отдельные intercepts при общих slopes. Большое число валют не подменяет индивидуальные результаты: все 960 строк и confusion matrices сохранены.", "",
        "## Наблюдаемые максимумы", "",
        "Эти строки выбраны после расчёта для описания и не являются независимо отобранными победителями.", "",
        "| Горизонт | Модель / валюта | Accuracy | Actual coverage | Emitted n |", "|---|---|---:|---:|---:|"]
    for minutes in [1, 10, 30]:
        for full in [True, False]:
            subset = [r for r in rows if r["minutes"] == minutes and r["accuracy_all"] is not None and (not full or r["coverage_target"] == 1)]
            r = max(subset, key=lambda r: r["accuracy_all"])
            report.append(f"| {minutes}m | {r['model']} / {r['symbol']} | {r['accuracy_all']:.2%} | {r['coverage_actual']:.2%} | {r['n_emitted']} |")
    report += ["", "## Что дала микроструктура", "",
        "BTCUSDT за 01.12.2024 — отдельный однодневный инженерный тест. Первые 60% дня train, следующие 20% calibration, последние 20% test, maturity purge 7*h минут. Все 18 моделей имеют одинаковые допустимые строки. Прогноз через 1/10/30m при решении каждую минуту; labels 10/30m перекрываются. Инференциальные CI по одному дню не выдаются.",
        "Проверены flow-only, book-only, лаги micro-признаков, OI/mark/index/displayed funding, компактные 6 признаков с logistic/MLP65, trade-mid gap и взаимодействия flow/book. Данные derivatives доступны по receipt clock; displayed funding известен в момент публикации и не является реализованным будущим платежом.",
        "Quote extraction независимо совпала с исходным atomic book по spread и возрасту во всех 1440 минутах. Архив с именем incremental L2 фактически состоит из snapshots; sequence completeness не доказана. Jan-2025 book_ticker не смешивался с Dec-2024 данными.", "",
        "| Модель | 1m accuracy | 10m accuracy | 30m accuracy |", "|---|---:|---:|---:|"]
    for v in read(MICRO / "registration.json")["contract"]["models"]:
        rr = [next(r for r in micro_rows if (r["horizon_minutes"], r["model"], r["coverage_target"]) == (m, v["id"], 1)) for m in [1, 10, 30]]
        report.append(f"| {v['id']} | {rr[0]['accuracy_all']:.2%} | {rr[1]['accuracy_all']:.2%} | {rr[2]['accuracy_all']:.2%} |")
    reference = next(r for r in micro_rows if (r["model"], r["horizon_minutes"], r["coverage_target"]) == ("reference_full_boost", 1, 1))
    diagnostic = reference["midprice_diagnostic"]
    report += ["", f"Лучшая минутная модель осталась reference_full_boost: {reference['accuracy_all']:.2%} на {reference['n_emitted']} прогнозах. Для тех же прогнозов диагностическая midprice accuracy: {diagnostic['accuracy_all']:.2%} на всех интервалах, {diagnostic['accuracy_nonflat']:.2%} без flat; flat={diagnostic['flat_emitted']}. Это другая цель, приведённая отдельно; модели обучались на trade-price направлении.",
        "Book-only boosting почти повторяет результат full; flow-only, derivatives, quote gap и новая компактная MLP не улучшили его на полном покрытии. Это локальная атрибуция на одном дне, не глобальное опровержение этих источников информации.",
        "При отказах максимум остался у reference_full_boost/10m: 77,78% на 18 прогнозах. Большая точность на маленьком отобранном наборе не доказывает достижение устойчивых 80–90%.", "",
        "## Архив и ограничения", "",
        "Все 20 архивных гипотез перечислены в HYPOTHESIS_INVENTORY_RU.md. Закрытые торговые стратегии не переоткрывались и не перенастраивались на их раскрытом OOS. Новые эксперименты относятся к другой задаче: прогнозу направления без исполнения и PnL.",
        "Недоступны исходные Glassnode данные/код авторов и publication vintages, синхронные многодневные L2/trades, point-in-time новости/sentiment. Эти гипотезы остаются data-blocked; OHLCV или price forecasts не выдаются за их воспроизведение.",
        "Полный prospective-интервал ещё не завершён. Новый пакет не подмешан в ранее замороженный контракт других моделей. Независимых улучшений, разрешений на реальные деньги и публикации новых торговых артефактов нет.", "",
        "## Дальнейшие действия", "",
        "1. Приоритет — многодневный matched trades/L2 набор: проверить два небольших предиктора price+book и price+book+flow с одной зарегистрированной абляцией. Показывать trade- и midprice цели раздельно.",
        "2. Из OHLCV оставить для узкого подтверждения лаги на 10m и робастный/временной ансамбль на 1m. Наблюдаемые сдвиги малы; это кандидаты следующего опыта, не одобренные модели.",
        "3. Проверять только несколько заранее выбранных вариантов на новом времени после фиксации протокола. Ранее раскрытый август–сентябрь остаётся исследовательским.",
        "4. Если возвращаться к MLP65, отдельно зарегистрировать достаточный optimisation budget и критерий train-only convergence до анализа нового теста; текущий warning нельзя трактовать как бесполезность всей архитектуры.",
        "5. On-chain и sentiment проверять после получения данных с доказуемыми публикационными часами. Увеличение нейросети или GA само по себе не добавит отсутствующую информацию.", "",
        "## Артефакты", "",
        "statistics_excel_ru.tsv — 960 строк × 18 колонок; results.json — все метрики, месяцы, confusion, контроли и paired CI. Micro TSV — 216 строк × 15 колонок; midprice diagnostic отделён от primary.",
        "Проверено 630 основных forecast artifacts, 20160 порогов и 54 micro модели. Все model/data/fold/cutoff/registration hashes сохранены. Credentials/orders/paper/live/runtime/network/promotion=false; финансовых replay нет.", ""]
    for name in output_figures: report += [f"![Forecast comparison]({name})", ""]
    report += ["![Micro full coverage](../direction_micro_hypotheses_v1_20261013/full_coverage_comparison.png)", ""]
    with (OUT / "REPORT_RU.md").open("x", encoding="utf-8") as f: f.write("\n".join(report))
    inventory = ["# Просмотренный архив гипотез", "", "Источник: configs/research_archive/hypotheses_v1.json; SHA сохранён в регистрации.", "",
        "| ID | Архивный статус | Исходный вывод | Действие в новом пакете |", "|---|---|---|---|"]
    for item in reg["archive_hypotheses"]:
        action = "Торговая стратегия не переоткрывалась; применимые данные/проверки рассмотрены отдельно"
        if item["status"] == "DATA_BLOCKED": action = "Остаётся зависимость от данных; текущий displayed funding тестирует только направление"
        inventory.append(f"| {item['id']} | {item['status']} | {str(item.get('verdict','')).replace('|','/')} | {action} |")
    with (OUT / "HYPOTHESIS_INVENTORY_RU.md").open("x", encoding="utf-8") as f: f.write("\n".join(inventory))
    summary_result = write(OUT / "summary.json", {"main_fits": 567, "micro_fits": 54, "forecast_artifacts": 630,
        "main_rows": 960, "micro_rows": 216, "contrasts": 4800, "positive_parent_policy_both_banks": positives_parent,
        "full_coverage_summary": summary, "warning_artifacts": len(warnings), "warning_models": sorted(set(m['variant']['id'] for m in warnings)),
        "archive_status_counts": dict(Counter(a["status"] for a in reg["archive_hypotheses"])),
        "independent_improvement": False, "environment": {"python": sys.version, "executable": sys.executable,
            **{p: importlib.metadata.version(p) for p in ["numpy", "scikit-learn", "pandas", "joblib", "matplotlib"]}}, "safety": reg["safety"]})
    overview = ["# Результаты проверки дополнительных гипотез", "",
        "621 новое обучение завершено. Основной пакет: 12 вариантов/9 тем, 8 валют, 1/10/30m, 21 origin. Микроструктура: 18 моделей × 3 горизонта на одном BTC-дне.",
        "Положительных simultaneous paired LCB против прежней модели в обоих банках: 0/960. Устойчивое независимое улучшение не подтверждено.",
        "Небольшие сдвиги есть у лагов, train-only robust preprocessing, большего train bank и временного ансамбля. Новые micro признаки не превысили прежние 61,11% на полном минутном покрытии; book-only даёт 60,42%.",
        "MLP65 достигла лимита оптимизации во всех 63 основных запусках: вывод об архитектуре ограничен. Старые торговые terminal-rejected гипотезы не возобновлялись.",
        "Полный отчёт с гипотезами, таблицами и графиками: [REPORT_RU.md](direction_hypotheses_v1_20261013/REPORT_RU.md). Архив: [inventory](direction_hypotheses_v1_20261013/HYPOTHESIS_INVENTORY_RU.md).", "",
        "Следующая проверка требует многодневных L2/trades либо действительно нового временного участка. Сеть и торговые разрешения остаются false.", "",
        "![Full coverage delta](direction_hypotheses_v1_20261013/full_coverage_delta.png)"]
    with (BASE / "DIRECTION_HYPOTHESES_REVIEW_20261013.md").open("x", encoding="utf-8") as f: f.write("\n".join(overview))
    manifest = {}
    for folder in [OUT, MICRO]:
        for p in folder.iterdir():
            if p.is_file(): manifest[p.relative_to(ROOT).as_posix()] = digest(p)
    write(BASE / "direction_hypotheses_delivery_20261013.json", {"artifact_sha256": manifest,
        "report_sha256": digest(BASE / "DIRECTION_HYPOTHESES_REVIEW_20261013.md"),
        "summary_sha256": summary_result["sha256"], "safety": reg["safety"]})
    print(json.dumps({"fits": 621, "main_rows": 960, "micro_rows": 216, "figures": len(output_figures)+1,
                      "positive_parent": positives_parent, "artifacts": len(manifest)}))


if __name__ == "__main__": main()
