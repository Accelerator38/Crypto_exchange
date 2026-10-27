"""Report only: revealed-day diagnostics, data gap, and frozen multiday protocol."""
from __future__ import annotations
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_genetic_primus_direction_book_multiday_v1 as run
import numpy as np
import matplotlib.pyplot as plt


NAMES = {"price_logit": "Логистическая, цена", "price_boost": "Boosting, цена",
         "book_logit": "Логистическая, цена + стакан", "book_boost": "Boosting, цена + стакан"}
TARGETS = {"last_trade": "Цена последней сделки", "midprice": "Midprice — середина bid/ask"}
CONTROL_NAMES = {"random": "Random, фиксированный seed", "majority_train_only": "Majority, только train",
                 "momentum_1m": "Momentum, прошлая минута"}


def percent(value):
    return "—" if value is None else ("{:.2f}%".format(100*value)).replace(".", ",")


def main():
    reg, c = run.registered(), run.contract()
    results = run.read(run.OUT / "known_day_results.json")
    audit = run.read(run.OUT / "engineering_audit.json")
    stats_audit = run.read(run.OUT / "statistics_engineering_audit.json")
    preparation = run.read(run.OUT / "known_day_preparation.json")
    entry = run.read(run.OUT / "available_days.json")["entries"][0]
    bank, quality = run.core.load_day(entry["bank_path"], c)
    with np.load(run.OUT / "known_day_engineering/forecast_ledger.npz", allow_pickle=False) as archive:
        ledger = {key: archive[key].copy() for key in archive.files}
    full = [r for r in results["rows"] if r["coverage_target"] == 1.0]
    row_map = {(r["target"], r["model"]): r for r in full}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    short = ["Линейная\nцена", "Boosting\nцена", "Линейная\n+ стакан", "Boosting\n+ стакан"]
    for target, ax in zip(c["targets"], axes):
        values = [100*row_map[(target, model)]["accuracy_all"] for model in c["models"]]
        bars = ax.bar(np.arange(4), values, color=["#91a3b0", "#91a3b0", "#387dac", "#387dac"])
        ax.axhline(50, color="#777777", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(4), short)
        ax.set_title(TARGETS[target], fontsize=11)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Верное направление, % всех допустимых интервалов")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, value+1.5, "{:.2f}%".format(value), ha="center", fontsize=10)
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("BTCUSDT, 1 минута: старый раскрытый день 01.12.2024", fontsize=13)
    fig.text(.5, .02, "288 тестовых интервалов • flat считается ошибкой • не новая независимая валидация", ha="center")
    fig.tight_layout(rect=(0, .06, 1, .94))
    fig.savefig(run.OUT / "full_coverage_accuracy.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.6), sharey=True)
    for target, ax in zip(c["targets"], axes):
        matrix = np.full((4, 4), np.nan)
        annotations = {}
        for i, model in enumerate(c["models"]):
            for j, coverage in enumerate(c["coverage_targets"]):
                row = next(r for r in results["rows"] if r["target"] == target and r["model"] == model
                           and r["coverage_target"] == coverage)
                if row["accuracy_all"] is not None: matrix[i, j] = 100*row["accuracy_all"]
                annotations[(i, j)] = "{}\nn={}".format(percent(row["accuracy_all"]), row["n_emitted"])
        image = ax.imshow(matrix, vmin=40, vmax=80, cmap="viridis", aspect="auto")
        ax.set_xticks(range(4), ["100%", "50%", "20%", "10%"])
        ax.set_yticks(range(4), [NAMES[m] for m in c["models"]])
        ax.set_title(TARGETS[target], fontsize=11)
        ax.set_xlabel("Целевое покрытие; порог фиксирован на calibration")
        for (i, j), annotation in annotations.items():
            ax.text(j, i, annotation, ha="center", va="center",
                    color="black" if matrix[i, j] > 69 else "white", fontsize=10)
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Точность, %", fraction=.025, pad=.025)
    fig.suptitle("Отказы от прогноза: высокая точность при малом n не доказывает устойчивость", fontsize=12)
    fig.subplots_adjust(left=.22, right=.88, bottom=.17, top=.84, wspace=.17)
    fig.savefig(run.OUT / "selective_accuracy.png", dpi=150)
    plt.close(fig)

    controls = []
    for target_no, target in enumerate(c["targets"]):
        for name in c["controls"]:
            if name == "matched_price_model": continue
            controls.append(dict(target=target, name=name,
                **run.selective.evaluate(ledger[name][target_no], ledger["labels"][target_no])))
    label_table = {}
    for i, target in enumerate(c["targets"]):
        labels = ledger["labels"][i]
        label_table[target] = {"down": int((labels < 0).sum()), "flat": int((labels == 0).sum()),
                               "up": int((labels > 0).sum())}
    label_agreement = float(np.mean(ledger["labels"][0] == ledger["labels"][1]))
    confusion = [[int(np.count_nonzero((ledger["labels"][0] == a) & (ledger["labels"][1] == b)))
                  for b in [-1, 0, 1]] for a in [-1, 0, 1]]
    run.write(run.OUT / "target_diagnostics.json", {"controls": controls, "label_counts": label_table,
              "label_agreement": label_agreement, "trade_vs_mid_confusion_down_flat_up": confusion,
              "engineering_only": True, "independent_validation": False})
    lines = ["# Многодневная проверка стаканных прогнозов: готовность и инженерный результат",
             "", "Дата: 13 октября 2026. Статус: **DATA_INSUFFICIENT для многодневной проверки**.", "",
             "## Итог", "",
             "Реализован отдельный offline evaluator для минутного направления BTCUSDT: две компактные модели, "
             "одинаковые price-only контроли, отдельно обучаемые цели last-trade и midprice. "
             "Выполнены восемь инженерных обучений на ранее раскрытом дне. Новая многодневная валидация НЕ выполнена: "
             "из 60 требуемых совпадающих дневных пар trades/L2 имеется одна.", "",
             "Boosting со стаканом получил {} для last-trade и {} для midprice при полном покрытии. "
             "Это 288 минут последней части одного известного дня, не доказательство переносимого преимущества, "
             "достижения 80–90% или торговой прибыли.".format(
                 percent(row_map[("last_trade", "book_boost")]["accuracy_all"]),
                 percent(row_map[("midprice", "book_boost")]["accuracy_all"])), "",
             "## Данные и измерительная семантика", "",
             "- Пара источников: `bitget-futures_trades_2024-12-01_BTCUSDT.csv.gz` и "
             "`bitget-futures_incremental_book_L2_2024-12-01_BTCUSDT.csv.gz` в `C:/Users/anton/Downloads`.",
             "- Book ticker от 01.01.2025 исключён: дата не совпадает. Других совпадающих raw-пар в проверенных "
             "Downloads, Retrodate и Reports не обнаружено.",
             "- 1 440 receipt-clock минут. До новой проверки свежести было 1 410 допустимых минут; теперь {}. "
             "Устаревшая последняя сделка: {} минут; устаревший стакан: {}. Первые 30 минут — warmup.".format(
                 quality["eligible_minutes"], quality["stale_trade_minutes"], quality["stale_book_minutes"]),
             "- Фильтр свежести одинаков для обеих целей и всех моделей: возраст сделки/стакана не более 5 секунд. "
             "В label допустимы только свежие начальная и конечная точки.",
             "- Источник стакана по предыдущему аудиту: 23 346 510 snapshot-строк, а не проверенная цепочка delta. "
             "Sequence completeness не подтверждена; реальные fills здесь не моделируются.",
             "- Все события строго раньше границы минуты. При восстановлении микросекунд из float-миллисекунд "
             "применено округление вместо усечения: это устраняет ошибку преобразования на 1 мкс. "
             "Первый prepare был остановлен этой проверкой до создания банка и до обучения.",
             "- Price-only: 6 признаков; price+book: те же 6 плюс spread, bid/ask depth5, depth imbalance, "
             "изменение глубины, microprice offset и возраст стакана. Линейные модели: 7 и 14 параметров. "
             "Boosting ограничен 32 деревьями, максимум 7 листьев; MLP/GA не добавлялись.", "",
             "Цель — знак изменения цены через одну минуту. Flat сохраняется отдельным исходом и считается "
             "ошибкой бинарного прогноза. Основная accuracy = correct / emitted; nonflat accuracy — дополнительная. "
             "Midprice равен (bid + ask) / 2; это не цена гарантированного исполнения.", "",
             "## Инженерный расчёт на известном дне", "",
             "Разбиение: первые 60% дня для train, следующие 20% для calibration, последние 20% для теста. "
             "Purge 7 минут на обеих границах с проверкой зрелости label. "
             "Test: 19:12–23:59 UTC, 288 решений; endpoint последнего label — 00:00 следующего дня. "
             "Это исключение для однодневной инженерной проверки, НЕ обещанный режим train30d/cal7d/retrain3d.", "",
             "| Модель | Last-trade, все интервалы | Midprice, все интервалы | n на каждую цель |",
             "|---|---:|---:|---:|"]
    for model in c["models"]:
        lines.append("| {} | {} | {} | 288 |".format(NAMES[model], percent(row_map[("last_trade", model)]["accuracy_all"]),
                                                    percent(row_map[("midprice", model)]["accuracy_all"])))
    lines.extend(["", "![Сравнение при полном покрытии](full_coverage_accuracy.png)", "",
                  "Рисунок 1. Проверяется, исчезает ли преимущество стакана при смене цели на midprice. "
                  "На этом дне оно не исчезает; независимая устойчивость не установлена. Пунктир 50% — ориентир, "
                  "а не универсальная ожидаемая accuracy random при наличии flat.", "",
                  "| Цель | Random | Majority train-only | Momentum 1m | Down / Flat / Up |",
                  "|---|---:|---:|---:|---|"])
    for target in c["targets"]:
        values = [next(r for r in controls if r["target"] == target and r["name"] == name)["accuracy_all"]
                  for name in ["random", "majority_train_only", "momentum_1m"]]
        counts = label_table[target]
        lines.append("| {} | {} | {} | {} | {} / {} / {} |".format(TARGETS[target], *(percent(v) for v in values),
                     counts["down"], counts["flat"], counts["up"]))
    lines.extend(["", "Совпадение самих last-trade и midprice labels: {}. "
                  "Цели обучались раздельно; один и тот же сохранённый price-прогноз не выдавался за новую модель.".format(percent(label_agreement)),
                  "", "![Точность с отказами и число прогнозов](selective_accuracy.png)", "",
                  "Рисунок 2. Coverage — целевое значение на calibration; фактическое покрытие теста может отличаться. "
                  "n указан в каждой ячейке. Высокий процент на нескольких десятках прогнозов не является подтверждением "
                  "80–90%. Доверительные интервалы для одного дня намеренно не рассчитывались.", "",
                  "## Протокол настоящей многодневной проверки", "",
                  "1. Получить 60 подряд полных UTC-дней BTCUSDT одной площадки: 30 дней контекста + 30 дней нового "
                  "исторического теста. Даты выбрать по доступности, не по результатам стратегии; test-исходы не изучать "
                  "до фиксации дат и SHA. Это ограниченный исследовательский batch, не обоснованная мощностью финальная проверка.",
                  "2. Квалифицировать по одному дню: bank/source/code SHA, строгий receipt clock, 1440 границ, "
                  "не менее 1350 свежих минут, bid<ask, causal midpoint, согласованные price-признаки. "
                  "Для incremental необходимы начальный snapshot и подтверждённая sequence continuity. "
                  "Автоматическая проверка метаданных и hash не доказывает происхождение/полноту биржевых данных.",
                  "3. Зафиксировать manifest, даты, 10 origins, hashes и cutoff. Rolling window 30 дней содержит "
                  "7 дней calibration; для fit остаётся примерно 23 дня. Purge 7 баров на train→cal и cal→origin. "
                  "Горизонт/шаг примеров/решений — 1 минута; retrain — 3 дня. Запрещены интерполяция и labels через пропуск/границу архива.",
                  "4. На origin один общий time-based sample до 4000 строк, отдельное исключение flat только при fit "
                  "каждой цели. Price/book используют одни и те же обучающие строки данной цели и одинаковую price-часть scaler. "
                  "Всего 8 моделей на origin, максимум 80 fit; не выполнять их до data freeze.",
                  "5. Пороги покрытий 100/50/20/10% только на calibration. Ledger хранит SHA модели через явный "
                  "model-index lookup для каждого прогноза. Цена сделки и midprice не смешиваются.",
                  "6. Единственный финальный анализ после всех 10 origins. Общие 3-календарно-дневные block-bootstrap "
                  "банки, 1000 повторов, два seed. Одна max-t family из 128 сравнений: две модели × две цели × четыре "
                  "покрытия × четыре контроля × два способа сопоставления. Основной сигнал — парная добавка на одинаковой "
                  "candidate-mask; own-policy-mask — дополнительный анализ. Нет emissions в bootstrap → нет положительного вывода.",
                  "7. Новый prospective-контракт от предыдущей работы не изменён. Эти модели туда не включены задним числом. "
                  "При наличии эффекта всё равно потребуется новый согласованный независимый период; accuracy не равна прибыли.", "",
                  "## Проверки и воспроизводимость", "",
                  "| Проверка | Свидетельство | Статус |", "|---|---|---|",
                  "| Синтаксис отдельных новых файлов, без pytest | py_compile | PASS |",
                  "| Метки, gaps, purge, prefix-invariance, scaler, thresholds | checks_*.json, 11 проверок | PASS |",
                  "| Повторный расчёт без обучения: 8 моделей / 32 metrics | engineering_audit.json | PASS |",
                  "| Синтетический полный final scorer: 30 дней / 10 origins / 128 пар | statistics_engineering_audit.json | PASS |",
                  "| Идентичные candidate/parent дают нулевую добавку и обе LCB=0 | statistics_engineering_audit.json | PASS |",
                  "| Один день не может запечатать многодневный тест | engineering_audit.json | FAIL-CLOSED |",
                  "| Реальные многодневные fit / тест | data_freeze.json отсутствует | DATA_INSUFFICIENT |", "",
                  "Фактические обучения этого этапа: 8 исторических инженерных + {} синтетических (два запуска "
                  "проверок по три). Synthetic statistics fixture не обучает моделей: его `.joblib` — текстовые hash-only "
                  "заглушки, не для загрузки. Поле `fits=80` внутри fixture обозначает ожидаемый счётчик протокола, "
                  "а не реальные обучения. Реальных multiday fit: 0.".format(reg["synthetic_fit_disclosure"]), "",
                  "Изменены/добавлены только отдельные файлы direction-book research и registry v18. Старые модели "
                  "не перезаписывались. Все семь разрешений credentials/orders/paper/live/runtime/network/promotion — false. "
                  "Никаких collector, pytest, сетевых запросов, торгов или фоновых процессов не запускалось.", "",
                  "## Остаточные ограничения и следующий шаг", "",
                  "- Главный блокер — данные, не архитектура. Имеется одна из 60 требуемых пар; не хватает 59 дополнительных "
                  "дней, примыкающих к ней, либо полного другого 60-дневного блока. Старый день может быть train/context, "
                  "но не новым test.",
                  "- `prepare` преобразует именно уже предоставленный известный день. Для новых raw L2 потребуется "
                  "отдельно проверить/зафиксировать их преобразователь в стандартизированные minute banks. `qualify`, "
                  "`freeze`, `fit_origin` и `score` готовы для таких банков; raw-features при qualification заново "
                  "не вычисляются. Timestamp provenance — проверяемое утверждение builder, не криптографическое доказательство causal preprocessing.",
                  "- Финальный scorer проверен полностью на синтетике, но ещё не на реальном многодневном датасете. "
                  "30 дней теста и 3-дневный block не гарантируют статистическую мощность и не охватывают все режимы.",
                  "- Следующее действие: получить совпадающие локальные trades/L2 для выбранного блока, проверить "
                  "один новый день, затем фиксировать data manifest и начинать bounded-origin прогоны. Без данных "
                  "не увеличивать число моделей и не повторять старый day-test ради более красивой accuracy.", "",
                  "## Артефакты", "",
                  "- `registration.json`, `configs/genetic_primus_objective_registry_v18.json`: frozen protocol/учёт попыток.",
                  "- `known_day_results.json`: все 32 metric cells; `target_diagnostics.json`: контроли/labels.",
                  "- `known_day_engineering/receipt.json` и `forecast_ledger.npz`: folds, cutoff, thresholds, model SHA.",
                  "- `available_days.json`, `known_day_preparation.json`, `qualification_*`: данные и качество.",
                  "- `engineering_audit.json`, `statistics_engineering_audit.json`: независимый пересчёт и синтетические проверки.",
                  "- Отчёт — Markdown/PNG; DOCX не запрашивался, не создавался и не рендерился.", ""])
    report_path = run.OUT / "REPORT_RU.md"
    with report_path.open("x", encoding="utf-8") as stream: stream.write("\n".join(lines))
    files = list(run.OUT.rglob("*"))
    file_pins = {str(p.relative_to(run.ROOT)).replace("\\", "/"): run.core.file_sha(p)
                 for p in files if p.is_file()}
    for path in [*run.CODE, Path(__file__), run.REGISTRY,
                 run.ROOT / "tools/check_genetic_primus_direction_book_multiday_v1.py",
                 run.ROOT / "tools/check_genetic_primus_direction_book_statistics_v1.py"]:
        file_pins[str(path.relative_to(run.ROOT)).replace("\\", "/")] = run.core.file_sha(path)
    run.write(run.OUT / "delivery_manifest.json", {"artifact_pins": file_pins,
              "historical_fits": 8, "synthetic_estimator_fits": reg["synthetic_fit_disclosure"],
              "real_multiday_fits": 0, "real_multiday_status": "DATA_INSUFFICIENT",
              "independent_validation": False, **{k: False for k in run.core.SAFETY}})
    print(json.dumps({"report": "REPORT_RU.md", "charts": 2, "pinned_artifacts": len(file_pins),
                      "status": "DATA_INSUFFICIENT"}))


if __name__ == "__main__":
    main()
