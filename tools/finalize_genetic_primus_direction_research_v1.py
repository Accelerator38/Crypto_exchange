"""Report and evidence checks only; never retrains or changes registered models."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "tools"))
from exia.genetic_primus import direction_selective_v1 as d
from exia.genetic_primus import direction_microstructure_v1 as micro
from run_genetic_primus_direction_selective_v1 import digest, identity, read, write, registration
from check_genetic_primus_direction_new_interval_v1 import check

BASE = ROOT / "Reports/Exia/Genetic_Primus"
MAIN = BASE / "direction_selective_v1_20261013"
MICRO = BASE / "direction_microstructure_v1_20261013"
CHAIN = BASE / "onchain_replication_v1_20261013"


def future_contract():
    original = MAIN / "new_interval_contract.json"
    v1 = read(original)
    amendment = {k: v for k, v in v1.items() if k != "sha256"}
    amendment.update({"predecessor_path": original.relative_to(ROOT).as_posix(),
        "predecessor_file_sha256": digest(original), "amendment": "Pre-data correction of infeasible universal sample floors",
        "reason": "90 days * 6 decisions/day * 10% coverage = 54 expected 4h predictions; 1000 nonflat observations and 60 emitted days cannot be universal minima.",
        "minimum_emitted_nonflat": 50, "minimum_emitted_days": 30,
        "sample_floor_is_not_power_proof": True, "historical_fit_or_threshold_changes": False,
        "new_data_opened_for_amendment": False,
        "qualification": "Minimum eligibility only. Simultaneous paired LCB against all controls in both banks remains mandatory. Sparse 4h cells may be inconclusive."})
    write(MAIN / "new_interval_contract_v2.json", amendment)
    print(json.dumps({"future_contract_amended_before_data": True, "minimum_nonflat": 50, "minimum_days": 30}))


def registry():
    predecessor = ROOT / "configs/genetic_primus_objective_registry_v14.json"
    previous = read(predecessor)
    entries = []
    for folder, objective, family, budget, filename in [
        (MICRO, "direction_microstructure_v1_20261013", "single_day_direction_microstructure_engineering", 12, "registration.json"),
        (CHAIN, "onchain_replication_v1_20261013", "onchain_replication_readiness_no_data", 0, "readiness.json")]:
        path = folder / filename
        entries.append({"objective_id": objective, "family": family, "registration_path": path.relative_to(ROOT).as_posix(),
            "registration_sha256": read(path)["sha256"], "registration_file_sha256": digest(path),
            "historical_fit_budget": budget, "financial_replays": 0, "independent_validation": False,
            "historical_adaptive_trial_count_complete": False, "promotion_authority": False})
    result = {k: v for k, v in previous.items() if k != "sha256"}
    result.update({"registry_id": "genetic_primus_objective_registry_v15",
                   "previous_registry_path": predecessor.relative_to(ROOT).as_posix(),
                   "previous_registry_sha256": digest(predecessor), "objectives": previous["objectives"] + entries})
    write(ROOT / "configs/genetic_primus_objective_registry_v15.json", result)
    print(json.dumps({"registered_additional_objectives": len(entries), "additional_fit_budget": 12}))


def audit_micro():
    import joblib
    reg = read(MICRO / "registration.json")
    for path, sha in reg["code_pins"].items():
        if digest(path) != sha: raise ValueError("micro engineering code changed")
    with np.load(MICRO / "minute_bank.npz", allow_pickle=False) as a: bank = {k: a[k] for k in a.files}
    eligible = np.isfinite(bank["price_and_microstructure"]).all(axis=1)
    base_ms = 1733011200000
    cells, thresholds = 0, 0
    for h in [1, 10, 30]:
        ix = np.arange(1440-h)
        valid = eligible[ix] & np.isfinite(bank["closes"][ix+h])
        train = ix[valid & (bank["times"][ix+h] <= base_ms + (864-7*h)*60000)]
        cal = ix[valid & (bank["times"][ix] >= base_ms+864*60000) & (bank["times"][ix+h] <= base_ms+(1152-7*h)*60000)]
        for features in ["price_only", "price_and_microstructure"]:
            for model_name in ["logistic", "boosting"]:
                name = f"h{h}_{features}_{model_name}"
                model_path = MICRO / (name + ".joblib")
                model = joblib.load(model_path)
                pc = d.predict(model, bank[features][cal])
                y_train = np.sign(bank["closes"][train+h] - bank["closes"][train])
                train_x = bank[features][train][y_train != 0]
                if not np.array_equal(train_x.mean(axis=0), model["mean"]): raise ValueError("micro scaler audit failed")
                with np.load(MICRO / (name + ".npz"), allow_pickle=False) as saved:
                    if str(saved["model_sha256"]) != digest(model_path): raise ValueError("micro model SHA mismatch")
                    idx = (saved["times"]-base_ms)//60000-1
                    y = np.sign(bank["closes"][idx+h]-bank["closes"][idx]).astype(np.int8)
                    if not np.array_equal(saved["labels"], y): raise ValueError("micro price-sign audit failed")
                    if not np.array_equal(d.predict(model, bank[features][idx]), saved["probability"]): raise ValueError("micro saved parity failed")
                    for k, coverage in enumerate([1, .5, .2, .1]):
                        threshold = d.calibrate(pc, bank["times"][cal], "BTCUSDT", coverage, 20261013)
                        if not np.array_equal(d.emit(saved["probability"], saved["times"], "BTCUSDT", threshold, 20261013), saved["masks"][k]):
                            raise ValueError("micro calibration-only mask audit failed")
                        thresholds += 1
                cells += 1
    # A future suffix changes no feature before the suffix, including volatility/depth change.
    c = 100 + np.arange(1440)*.01
    bars = np.column_stack([c-.01, c+.02, c-.02, c, np.ones(1440)])
    flow = np.ones((1440, 2)); count = np.full(1440, 2)
    book = [{"valid": True, "values": [1, 100, 200, -1/3, 0, 10]} for _ in range(1440)]
    before = micro.feature_bank(bars, flow, count, book)[1]
    bars[720:] *= 2; flow[720:] *= 3
    for i in range(720, 1440): book[i] = {"valid": True, "values": [5, 300, 900, -.5, 4, 20]}
    after = micro.feature_bank(bars, flow, count, book)[1]
    if not np.array_equal(before[:720], after[:720], equal_nan=True): raise ValueError("future suffix changed past features")
    result = {"models": cells, "thresholds": thresholds, "independent_labels": True,
              "saved_model_parity": True, "calibration_only_thresholds": True, "feature_prefix_causality": True,
              "synthetic_checks": micro.direct_checks(), "sequence_completeness_verified": False}
    write(MICRO / "audit.json", result)
    print(json.dumps({"micro_models_audited": cells, "thresholds": thresholds, "future_suffix_causality": True}))


def audit_source():
    import gzip
    reg = read(MICRO / "registration.json")
    path = next(Path(p) for p in reg["source_pins"] if "incremental_book" in p)
    counts, tail = [0, 0], b""
    with gzip.open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            data = tail + block
            cut = data.rfind(b"\n") + 1
            body, tail = data[:cut], data[cut:]
            counts[0] += body.count(b",true,")
            counts[1] += body.count(b",false,")
    if tail: raise ValueError("complete newline-terminated source required")
    if sum(counts) != read(MICRO / "data_audit.json")["book_rows"]:
        raise ValueError("source row audit mismatch")
    result = {"snapshot_rows": counts[0], "delta_rows": counts[1],
              "source_sha256": digest(path), "all_rows_are_snapshots": counts[1] == 0}
    write(MICRO / "source_kind_audit.json", result)
    print(json.dumps(result))


def verified_micro_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read(MICRO / "results.json")["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True, sharey=True)
    for ax, h in zip(axes, [1, 10, 30]):
        for fs in ["price_only", "price_and_microstructure"]:
            for model in ["logistic", "boosting"]:
                rr = sorted([r for r in rows if r["horizon_minutes"] == h and r["feature_set"] == fs
                             and r["model"] == model and r["accuracy_all"] is not None], key=lambda r: r["coverage_target"])
                ax.plot([100*r["coverage_actual"] for r in rr], [100*r["accuracy_all"] for r in rr], "o-", label=f"{fs}/{model}")
                for r in rr:
                    if r["accuracy_all"] >= .7:
                        ax.annotate(f"n={r['n_emitted']}", (100*r["coverage_actual"], 100*r["accuracy_all"]),
                                    xytext=(6, 0), textcoords="offset points", fontsize=8)
        overlap = "non-overlapping" if h == 1 else "overlapping"
        ax.set_title(f"h={h}m | {overlap} labels"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=6, loc="lower right")
    fig.supxlabel("Actual coverage, %"); fig.supylabel("Accuracy including flat, %")
    fig.suptitle("BTC single-day engineering | no independent improvement claim")
    fig.tight_layout(); fig.savefig(MICRO / "microstructure_accuracy_verified.png", dpi=140); plt.close(fig)
    print(json.dumps({"verified_figure": "microstructure_accuracy_verified.png"}))


def report():
    reg = registration()
    result, engineering = read(MAIN / "results.json"), read(MICRO / "results.json")
    rows = result["rows"]
    with (MAIN / "statistics_excel_ru.tsv").open(encoding="utf-8-sig", newline="") as f:
        table = list(csv.DictReader(f, delimiter="\t"))
    if len(table) != 256 or any(len(r) != 12 for r in table): raise ValueError("spreadsheet cells/rows audit failed")
    contract = read(MAIN / "new_interval_contract_v2.json")
    rejected = check(result, contract)
    if rejected["new_period_qualified"] or rejected["independent_positive_cells"]:
        raise ValueError("old period accepted as new")
    write(MAIN / "old_period_rejection.json", rejected)
    text = ["# Выполнение плана исследований направления цены — 13 октября 2026", "",
        "Выполнены 168 OHLCV обучений и 12 отдельных однодневных микроструктурных обучений. Независимое улучшение не подтверждено: новые рыночные данные не открывались.", "",
        "| Пункт | Сделано | Ограничение |", "|---|---|---|",
        "| 1. Знак будущего изменения | Единый up/down/flat, каждый допустимый интервал, 8 валют × 4 горизонта | Flat не превращается в down и не удаляется из основной accuracy |",
        "| 2. Отказы | 100/50/20/10%, порог по предыдущим 7 дням, отдельно по активам/моделям/origin | Фактическое test coverage не подгоняется; на 4h калибровка мала |",
        "| 3. Микроструктура | Trades/L2 за 01.12.2024 BTC, 6 price против 16 price+micro признаков, logistic/boosting | Один день и один актив; только инженерное сравнение |",
        "| 4. On-chain | Валидатор публикационных часов и контракт точного воспроизведения | Нет исходных Glassnode данных/кода авторов; 0 обучений, accuracy отсутствует |",
        "| 5. Новое время | Контракт 14.10.2026–11.01.2027 включительно; старый период отвергается | Новых данных нет, финальный анализ не выполнен, 90 дней не гарантируют мощность |", "",
        "## OHLCV и отказы", "",
        "Ранее раскрытый период: 01.08–30.09.2026. Горизонты 1/10/30/240 минут, один бар вперёд, решение на каждой соответствующей сетке. Retrain каждые 3 дня, scaler train-only, train+calibration окно 30 дней, калибровка 7 дней, purge 7 баров на обеих границах.",
        "Логистическая модель имеет 12 признаков и 13 коэффициентов. Boosting ограничен 32 итерациями и 7 листьями. Архитектура и настройки одинаковы для всех активов и четырёх уровней покрытия. Обучение pooled по восьми активам; пороги и метрики индивидуальные.",
        "Валидационная калибровка использует только уверенность, без истинных направлений. При равной уверенности применяется фиксированный hash времени/актива. В тесте нет quantile/ranking для принудительного покрытия.", "",
        "Ниже максимум по исследовательским результатам каждого горизонта выбран после расчёта, только для описания. Это не четыре независимо отобранных победителя. Full и selective в каждой строке относятся к той же модели/валюте.", "",
        "| Горизонт | Валюта / модель | Accuracy full | Accuracy selective | Target / actual coverage | Прогнозов selective |", "|---|---|---:|---:|---:|---:|"]
    for h in [1, 10, 30, 240]:
        r = max([r for r in rows if r["timeframe_minutes"] == h], key=lambda r: r["accuracy_all"])
        full = next(a for a in rows if (a["timeframe_minutes"], a["model"], a["symbol"], a["coverage_target"]) == (h, r["model"], r["symbol"], 1))
        text.append(f"| {h}m | {r['symbol']} / {r['model']} | {full['accuracy_all']:.2%} | {r['accuracy_all']:.2%} | {r['coverage_target']:.0%} / {r['coverage_actual']:.2%} | {r['n_emitted']} |")
    positives = [r for r in rows if r["research_positive_against_all_controls"]]
    text += ["", f"Семейная поправка max-t: {result['family_contrasts']} сравнений с random/majority/momentum; общие 3-дневные календарные блоки, 2000 повторов и отдельный audit seed bank. {len(positives)} ячеек имеют положительный LCB против всех контролей в обоих банках: все относятся к минутному ADA. Эти результаты исследовательские; seed bank не заменяет новые данные.",
        "У минутного ADA много flat. Поэтому общая accuracy ниже 50%, хотя модель может превосходить случайный прогноз; 50% на графике — визуальный ориентир, не универсальная random accuracy при наличии flat. В results.json есть nonflat accuracy, confusion matrix и actual flat для каждого покрытия.",
        "Абсолютные 80–90% не достигнуты. Увеличение условной accuracy при отказах не означает уменьшения ошибки на всех исходных интервалах. Для сравнения на одном покрытии контроли оцениваются на той же маске решений.",
        "Разница с прежним benchmark не является чистым эффектом отказов: для честной калибровки часть 30-дневного окна теперь отделена от обучения, а seed/обучающая подвыборка зарегистрированы заново. Чистое сравнение coverage проводится внутри этого эксперимента.", "",
        "## Микроструктурный тест", "",
        "BTCUSDT futures, receipt-clock 01.12.2024. Aggressor side соответствует схеме [Tardis](https://docs.tardis.dev/downloadable-csv-files/data-types). Используются подписанный поток сделок, доля агрессивных покупок, число сделок, spread, top-5 bid/ask depth, imbalance, изменение depth, microprice и возраст book.",
        "Сравнение 6 price признаков с 16 price+microstructure на одних допустимых строках. Прогноз через 1/10/30 минут при решениях каждую минуту; длинные labels перекрываются. Training/calibration/test = 60/20/20% дня, maturity purge 7*h минут на обеих границах. Никакой экстраполяции на другие валюты или август–сентябрь.", "",
        "| Горизонт | Модель | Accuracy price-only | Accuracy price+micro | Test n |", "|---|---|---:|---:|---:|"]
    for h in [1, 10, 30]:
        for kind in ["logistic", "boosting"]:
            pair = [next(r for r in engineering["rows"] if (r["horizon_minutes"], r["model"], r["feature_set"], r["coverage_target"]) == (h, kind, fs, 1)) for fs in ["price_only", "price_and_microstructure"]]
            text.append(f"| {h}m | {kind} | {pair[0]['accuracy_all']:.2%} | {pair[1]['accuracy_all']:.2%} | {pair[0]['n_emitted']} |")
    text += ["", "Включать эти числа в общий рейтинг OHLCV моделей нельзя. Один день не даёт достаточного числа независимых календарных блоков; bootstrap CI для него не выдаётся. Отсутствие sequence IDs оставляет недоказуемой полноту L2.",
        "Полный аудит обнаружил 23 346 510 snapshot-строк и 0 delta-строк в файле с названием incremental_book_L2. В этом архиве проверены повторные снимки, а не восстановление настоящего потока incremental deltas. Направление агрессора взято из trades, глубина — из доступных верхних уровней.",
        "При отказах максимум 77,78% получился на 18 прогнозах (10m boosting), 74,47% — на 47 прогнозах (1m logistic). Такие небольшие отобранные наборы не подтверждают достижение устойчивой точности 75–80%.", "",
        "## On-chain: фактический статус", "",
        "У [Omole и Enke](https://link.springer.com/article/10.1186/s40854-024-00643-1) опубликован результат 82,44% для следующего дня, Glassnode и Boruta + CNN-LSTM. В статье данные и код доступны от автора по запросу. Они отсутствуют локально; не отправлялись сообщения автору и не использовались ключи.",
        "Подготовлен отдельный NPZ/receipt валидатор: последовательные дневные часы, имена исходных признаков, hashes авторских материалов, время фактической публикации каждого значения. Поздно опубликованный признак отклоняется. Это проверка перед воспроизведением, не обучение и не подтверждение 82,44%.",
        "Следующая зависимость — получить авторский набор, реализацию и сведения о публикации/пересмотрах. Сначала воспроизвести исходную постановку, затем причинный вариант. В исходной binary постановке flat входит в отрицательный класс; этот показатель должен оставаться отдельным от нашей up/down/flat accuracy.", "",
        "## Новые данные и оставшиеся действия", "",
        "Зафиксирован будущий интервал 14 октября 2026 — 11 января 2027, один финальный анализ без промежуточной настройки. В v2 до открытия данных исправлены невыполнимые универсальные sample floors: минимум 50 nonflat emitted и 30 emitted days на ячейку. Это лишь нижний порог пригодности, не доказательство мощности; положительный simultaneous paired LCB против всех трёх контролей в обоих банках остаётся обязательным. Редкие 4h прогнозы могут дать неопределённый результат.",
        "Контракт v1 сохранён, поправка с его SHA — отдельный immutable v2. Функция проверки отвергает старый период, незавершённый новый период и несовпадающие сравнения. Сам checker не доказывает происхождение новых данных и не мешает произвольно запускать анализ многократно: организационное правило единственного анализа и аудит происхождения остаются необходимыми.",
        "Следующий содержательный эксперимент требует многодневных синхронных trades/L2 по нужным активам либо исходных дневных on-chain материалов. Повторная настройка архитектуры на этих же августе–сентябре не будет независимым улучшением.", "",
        "## Проверка и артефакты", "",
        "Аудит: 168 OHLCV моделей, 5376 порогов, 12 micro моделей и 48 micro порогов; независимая сверка знака с исходной ценой, saved-model parity, train-only scaler, калибровка только по прошлому, причинность признаков при изменении будущего суффикса, TSV 256 строк × 12 столбцов.",
        "Основной отчёт: [direction_selective](direction_selective_v1_20261013/REPORT_RU.md). Микроструктура: [microstructure](direction_microstructure_v1_20261013/REPORT_RU.md). On-chain: [readiness](onchain_replication_v1_20261013/readiness.json).", "",
        "![10m: accuracy versus actual coverage](direction_selective_v1_20261013/coverage_accuracy_10m.png)", "",
        "![Microstructure](direction_microstructure_v1_20261013/microstructure_accuracy_verified.png)", "",
        "Все credentials/orders/paper/live/runtime/network/promotion флаги false. Запросы к рынку, pytest, collector и торговые исполнения не запускались."]
    target = BASE / "DIRECTION_RESEARCH_PROGRESS_20261013.md"
    with target.open("x", encoding="utf-8") as f: f.write("\n".join(text))
    hashes = {}
    for folder in [MAIN, MICRO, CHAIN]:
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.name not in ("checkpoint.json",): hashes[p.relative_to(ROOT).as_posix()] = digest(p)
    write(BASE / "direction_research_delivery_20261013.json", {"artifact_sha256": hashes, "report_sha256": digest(target),
        "main_audit": read(MAIN / "audit.json")["sha256"], "micro_audit": read(MICRO / "audit.json")["sha256"],
        "old_period_rejection": rejected, "tsv_rows": 256, "tsv_columns": 12, "safety": reg["contract"]["safety"]})
    print(json.dumps({"report": target.name, "fits": 180, "main_rows": 256, "micro_rows": 48,
                      "onchain_fits": 0, "independent_improvement": False}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["future_contract", "registry", "audit_micro", "audit_source", "verified_micro_figure", "report"])
    args = parser.parse_args(); globals()[args.stage]()
