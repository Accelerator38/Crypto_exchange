"""Bounded offline experiment stages. One origin per fit command, no networking."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from exia.genetic_primus import direction_selective_v1 as d

CONFIG = ROOT / "configs/genetic_primus_direction_selective_v1.json"
PARENT = ROOT / "Reports/Exia/Genetic_Primus/direction_multiscale_v1_20261012"
OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_selective_v1_20261013"
CODE = [Path(__file__), CONFIG, ROOT / "src/exia/genetic_primus/direction_selective_v1.py"]
START, END = 1785542400000, 1790812800000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                   separators=(",", ":"), default=plain).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    value = dict(value)
    value["sha256"] = identity(value)
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False, default=plain)
    return value


def registration():
    reg = read(OUT / "registration.json")
    check = dict(reg); expected = check.pop("sha256")
    if identity(check) != expected: raise ValueError("registration modified")
    for name, expected in reg["code_pins"].items():
        if digest(ROOT / name) != expected: raise ValueError("registered code/config changed")
    return reg


def load(minutes):
    base = PARENT / f"tf_{minutes}m"
    return {name: np.load(base / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("times", "closes", "features")}


def init():
    contract = read(CONFIG)
    if any(contract["safety"].values()): raise ValueError("offline false permissions required")
    if OUT.exists(): raise ValueError("exclusive new registration required")
    pins = {}
    for minutes in contract["timeframes_minutes"]:
        for name in ("times", "closes", "features"):
            path = PARENT / f"tf_{minutes}m/{name}.npy"
            pins[path.relative_to(ROOT).as_posix()] = digest(path)
    OUT.mkdir()
    reg = write(OUT / "registration.json", {"schema": "direction_selective_registration/1",
        "created_utc": datetime.now(timezone.utc).isoformat(), "contract": contract,
        "code_pins": {p.relative_to(ROOT).as_posix(): digest(p) for p in CODE}, "data_pins": pins,
        "parent_registration_sha256": digest(PARENT / "registration.json"),
        "fit_budget": 168, "threshold_budget": 5376, "independent_validation": False,
        "financial_replays": 0, "safety": contract["safety"]})
    predecessor = ROOT / "configs/genetic_primus_objective_registry_v13.json"
    previous = read(predecessor)
    entry = {"objective_id": contract["objective_id"], "family": "direction_selective_with_causal_calibration",
        "config_path": CONFIG.relative_to(ROOT).as_posix(), "config_sha256": digest(CONFIG),
        "registration_path": (OUT / "registration.json").relative_to(ROOT).as_posix(),
        "registration_sha256": reg["sha256"], "historical_fit_budget": 168,
        "financial_replays": 0, "historical_adaptive_trial_count_complete": False,
        "independent_validation": False, "promotion_authority": False}
    registry = {k: v for k, v in previous.items() if k not in ("sha256", "predecessor")}
    registry.update({"registry_id": "genetic_primus_objective_registry_v14",
        "previous_registry_path": predecessor.relative_to(ROOT).as_posix(),
        "previous_registry_sha256": digest(predecessor), "registration_date_utc": reg["created_utc"],
        "objectives": previous["objectives"] + [entry]})
    write(ROOT / "configs/genetic_primus_objective_registry_v14.json", registry)
    write(OUT / "new_interval_contract.json", {"registration_sha256": reg["sha256"],
        **contract["new_interval"], "safety": contract["safety"],
        "qualification": "No claim until new data pass this frozen protocol. 90 days is not a power guarantee."})
    print(json.dumps({"registered": reg["sha256"], "fits": 168, "historical_only": True}))


def fit_origin(number):
    import joblib
    reg = registration(); cfg = reg["contract"]
    if not 0 <= number < 21: raise ValueError("one bounded origin 0..20 required")
    result = []
    for minutes in cfg["timeframes_minutes"]:
        for name, expected in reg["data_pins"].items():
            if f"tf_{minutes}m/" in name and digest(ROOT / name) != expected:
                raise ValueError("data bank changed")
        bank = load(minutes); times = bank["times"]
        origin, width = START + number * 3 * d.DAY, minutes * 60000
        seed = cfg["seed"] + minutes * 1000 + number
        train, calibration = d.split(times, origin, width, calibration_days=cfg["calibration_days"], seed=seed)
        test = np.flatnonzero((times >= origin) & (times < min(origin + 3 * d.DAY, END)))
        if test[-1] + 1 >= len(times) or len(test) != (min(origin + 3 * d.DAY, END) - origin) // width:
            raise ValueError("complete future labels required")
        label = d.signs(bank["closes"][train], bank["closes"][train + 1])
        x = bank["features"][train].reshape(-1, 12)
        xc, xt = bank["features"][calibration].reshape(-1, 12), bank["features"][test].reshape(-1, 12)
        y_test = d.signs(bank["closes"][test], bank["closes"][test + 1])
        momentum = bank["closes"][test] >= bank["closes"][test - 1]
        majority = np.sum(label == 1, axis=0) >= np.sum(label == -1, axis=0)
        coins = np.random.default_rng(np.random.SeedSequence([cfg["seed"], minutes, number, 50521])).integers(0, 2, y_test.shape).astype(bool)
        fold = {"origin_ms": origin, "train_times_sha256": identity(times[train]),
            "calibration_times_sha256": identity(times[calibration]), "test_times_sha256": identity(times[test]),
            "train_latest_label_ms": int(times[train[-1] + 1]), "calibration_start_ms": int(times[calibration[0]]),
            "calibration_latest_label_ms": int(times[calibration[-1] + 1]), "cutoff_ms": origin - 7 * width,
            "n_train_times": len(train), "n_calibration_times": len(calibration), "n_test_times": len(test)}
        for kind in cfg["models"]:
            prefix = f"tf_{minutes}m_origin_{number:02d}_{kind}"
            meta_path, model_path, predictions_path = [OUT / (prefix + suffix) for suffix in (".json", ".joblib", ".npz")]
            if meta_path.exists():
                meta = read(meta_path)
                if digest(model_path) != meta["model_sha256"] or digest(predictions_path) != meta["predictions_sha256"]:
                    raise ValueError("saved origin modified")
                result.append("reused"); continue
            if model_path.exists() or predictions_path.exists(): raise ValueError("partial artifact: no silent refit")
            model = d.fit(kind, x, label.reshape(-1), seed)
            pc, pt = d.predict(model, xc).reshape(-1, 8), d.predict(model, xt).reshape(-1, 8)
            thresholds, masks = {}, np.empty((4, len(test), 8), bool)
            for j, symbol in enumerate(cfg["symbols"]):
                thresholds[symbol] = []
                for k, coverage in enumerate(cfg["coverage_targets"]):
                    threshold = d.calibrate(pc[:, j], times[calibration], symbol, coverage, cfg["seed"])
                    thresholds[symbol].append(threshold)
                    masks[k, :, j] = d.emit(pt[:, j], times[test], symbol, threshold, cfg["seed"])
            joblib.dump(model, model_path, compress=3)
            restored = joblib.load(model_path)
            if not np.array_equal(pt, d.predict(restored, xt).reshape(-1, 8)):
                raise ValueError("saved model prediction parity failed")
            model_sha = digest(model_path)
            with predictions_path.open("xb") as f:
                np.savez_compressed(f, times=times[test], probability=pt, labels=y_test, masks=masks,
                    random_coin=coins, train_majority=np.broadcast_to(majority, y_test.shape), momentum=momentum,
                    model_sha256=np.array(model_sha))
            info = {k: v for k, v in model.items() if k != "estimator"}
            write(meta_path, {"registration_sha256": reg["sha256"], "fold": fold, "fold_sha256": identity(fold),
                "model": kind, "timeframe_minutes": minutes, "info": info, "thresholds": thresholds,
                "model_sha256": model_sha, "predictions_sha256": digest(predictions_path),
                "scaler_train_only": True, "calibration_labels_used": False, "saved_prediction_parity": True})
            result.append("fitted")
    print(json.dumps({"origin": number, "fitted": result.count("fitted"), "reused": result.count("reused")}))


def score():
    reg = registration(); cfg = reg["contract"]
    days = (END - START) // d.DAY
    banks = [d.block_bank(days, 2000, 3, s) for s in (6101301, 6101302)]
    rows, contrasts, replicates = [], [], [[], []]
    for minutes in cfg["timeframes_minutes"]:
        for kind in cfg["models"]:
            chunks = []
            for number in range(21):
                prefix = f"tf_{minutes}m_origin_{number:02d}_{kind}"
                meta = read(OUT / (prefix + ".json"))
                path = OUT / (prefix + ".npz")
                if digest(path) != meta["predictions_sha256"]: raise ValueError("prediction provenance failed")
                with np.load(path, allow_pickle=False) as a: chunks.append({k: a[k].copy() for k in a.files if k != "model_sha256"})
            times = np.concatenate([a["times"] for a in chunks])
            if np.any(np.diff(times) != minutes * 60000): raise ValueError("continuous test decisions required")
            arrays = {k: np.concatenate([a[k] for a in chunks], axis=1 if k == "masks" else 0)
                      for k in chunks[0] if k != "times"}
            for j, symbol in enumerate(cfg["symbols"]):
                y, p = arrays["labels"][:, j], arrays["probability"][:, j] >= .5
                for k, coverage in enumerate(cfg["coverage_targets"]):
                    mask = arrays["masks"][k, :, j]
                    stats = d.evaluate(p, y, mask)
                    daily = d.daily_counts(times, y, p, mask, START, days)
                    row = {"timeframe_minutes": minutes, "model": kind, "symbol": symbol,
                           "coverage_target": coverage, **stats, "controls": {}, "months": {}}
                    acc_draws = []
                    for b, bank in enumerate(banks):
                        interval, draws = d.resample_statistics(daily, bank)
                        row["bootstrap" if b == 0 else "audit_bootstrap"] = interval
                        acc_draws.append(draws)
                    month_boundary = 1788220800000
                    for month, selected in (("2026-08", times < month_boundary), ("2026-09", times >= month_boundary)):
                        row["months"][month] = d.evaluate(p[selected], y[selected], mask[selected])
                    row_id = len(rows)
                    for control in cfg["controls"]:
                        cp = arrays[control][:, j]
                        cs = d.evaluate(cp, y, mask)
                        row["controls"][control] = {"same_mask": cs, "full_coverage": d.evaluate(cp, y)}
                        cd = d.daily_counts(times, y, cp, mask, START, days)
                        delta = stats["accuracy_all"] - cs["accuracy_all"] if stats["n_emitted"] else 0.0
                        contrasts.append({"row": row_id, "control": control, "delta": delta})
                        for b, bank in enumerate(banks):
                            _, control_draws = d.resample_statistics(cd, bank)
                            replicates[b].append(acc_draws[b] - control_draws)
                    rows.append(row)
    obs = np.array([a["delta"] for a in contrasts])
    critical = []
    for b, name in enumerate(("bootstrap", "audit_bootstrap")):
        rep = np.asarray(replicates[b]).T
        valid = np.isfinite(rep).all(axis=0)
        lower = np.full(len(obs), np.nan)
        if valid.any():
            lower[valid], c = d.simultaneous_lcb(obs[valid], rep[:, valid])
        else:
            c = 0.0
        critical.append(c)
        for j, contrast in enumerate(contrasts):
            cell = rows[contrast["row"]]["controls"][contrast["control"]]
            cell[name] = {"paired_delta": contrast["delta"] if valid[j] else None,
                          "pointwise_ci95": np.quantile(rep[:, j], [.025, .975]).tolist() if valid[j] else None,
                          "simultaneous_lcb95": float(lower[j]) if valid[j] else None}
    for row in rows:
        row["research_positive_against_all_controls"] = all(
            c[b]["simultaneous_lcb95"] is not None and c[b]["simultaneous_lcb95"] > 0
            for c in row["controls"].values() for b in ("bootstrap", "audit_bootstrap"))
        row["independent_improvement"] = False
    result = write(OUT / "results.json", {"registration_sha256": reg["sha256"], "rows": rows,
        "shared_bank_sha256": [hashlib.sha256(b.tobytes()).hexdigest() for b in banks],
        "family_contrasts": len(contrasts), "max_t_critical": critical,
        "historical_only": True, "independent_improvement": False, "safety": cfg["safety"]})
    print(json.dumps({"rows": len(rows), "contrasts": len(contrasts),
        "positive_all_controls": sum(r["research_positive_against_all_controls"] for r in rows),
        "results_sha256": result["sha256"]}))


def report():
    import csv
    import matplotlib.pyplot as plt
    result = read(OUT / "results.json"); rows = result["rows"]
    symbols = registration()["contract"]["symbols"]
    fields = ["timeframe_minutes", "model", "symbol", "coverage_target", "coverage_actual", "n_eligible", "n_emitted",
              "accuracy_all", "accuracy_nonflat", "balanced_accuracy", "flat_emitted", "research_positive_against_all_controls"]
    with (OUT / "statistics_excel_ru.tsv").open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fields, delimiter="\t", extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    figures = []
    for minutes in (1, 10, 30, 240):
        fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharex=True, sharey=True)
        for ax, symbol in zip(axes.flat, symbols):
            for kind in ("logistic", "boosting"):
                selected = sorted([r for r in rows if r["timeframe_minutes"] == minutes and r["symbol"] == symbol and r["model"] == kind], key=lambda a: a["coverage_target"])
                x = [100 * a["coverage_actual"] for a in selected]
                y = [100 * a["accuracy_all"] for a in selected]
                low = [100 * (a["accuracy_all"] - a["bootstrap"]["ci95"][0]) for a in selected]
                high = [100 * (a["bootstrap"]["ci95"][1] - a["accuracy_all"]) for a in selected]
                ax.errorbar(x, y, yerr=[np.maximum(0, low), np.maximum(0, high)], marker="o", capsize=2, label=kind)
            ax.set_title(symbol); ax.axhline(50, color="grey", ls=":"); ax.grid(alpha=.2)
        axes[0, 0].legend(); fig.supxlabel("Actual coverage, %"); fig.supylabel("Accuracy including flat, %")
        fig.suptitle(f"{minutes} min | revealed Aug-Sep 2026 | pointwise 95% calendar-block CI")
        fig.tight_layout(); name = f"coverage_accuracy_{minutes}m.png"; fig.savefig(OUT / name, dpi=130); plt.close(fig); figures.append(name)
    lines = ["# Прогноз направления и отказы: исследовательский расчёт", "",
        "Период: 1 августа — 30 сентября 2026, ранее раскрытые данные. Независимых улучшений не установлено этим экспериментом.", "",
        "Цель: знак close[t+1] − close[t]. Flat сохраняется и считается ошибкой бинарного прогноза; условная точность без flat показана отдельно.",
        "Пороги по каждому активу выбираются по семи предыдущим календарным дням, без исходов и без ранжирования тестовых прогнозов. Фактическое покрытие может отличаться от целевого.",
        "30-дневное окно включает 7 дней калибровки. Между train/calibration и calibration/test — purge 7 баров. Scaler обучается на train. На 4h калибровочная выборка мала; пороги 10% особенно нестабильны.",
        "Общие для всех сравнений circular bootstrap блоки по 3 календарных дня, 2000 повторов; второй seed — проверка вычислительной устойчивости, не новые данные. Max-t поправка на все 768 сравнений с контролями; интервалы графиков точечные.", "",
        "| Горизонт | Модель | Target coverage | Actual coverage, диапазон по активам | Accuracy, диапазон по активам |", "|---|---|---:|---:|---:|"]
    for minutes in (1, 10, 30, 240):
        for kind in ("logistic", "boosting"):
            for coverage in (1, .5, .2, .1):
                subset = [r for r in rows if (r["timeframe_minutes"], r["model"], r["coverage_target"]) == (minutes, kind, coverage)]
                c, a = [100*r["coverage_actual"] for r in subset], [100*r["accuracy_all"] for r in subset]
                lines.append(f"| {minutes}m | {kind} | {coverage:.0%} | {min(c):.2f}–{max(c):.2f}% | {min(a):.2f}–{max(a):.2f}% |")
    lines += ["", "Это диапазоны разных активов, а не единая точность портфеля или рейтинг победителей.", "",
        "Все 256 строк, confusion matrix, flat, месяцы, paired contrasts и оба bootstrap находятся в results.json; таблица TSV содержит основные метрики.",
        "Новый интервал: 14 октября 2026 — 11 января 2027 включительно (90 дней), единственный финальный анализ. Это срок контракта, не доказанная гарантия мощности. Данные отсутствуют; независимая проверка не выполнена.",
        "Микроструктура и on-chain — отдельные эксперименты; результаты этого OHLCV расчёта не выдают их за выполненные.",
        "Credentials/orders/paper/live/runtime/network/promotion остаются false.", ""]
    for name in figures: lines += [f"![Accuracy/coverage]({name})", ""]
    (OUT / "REPORT_RU.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"table_rows": len(rows), "figures": len(figures), "report": "REPORT_RU.md"}))


def audit():
    import joblib
    reg = registration()
    checked, threshold_checked = 0, 0
    for minutes in reg["contract"]["timeframes_minutes"]:
        bank = load(minutes)
        for number in range(21):
            seed = reg["contract"]["seed"] + minutes * 1000 + number
            train, cal = d.split(bank["times"], START + number * 3 * d.DAY, minutes * 60000, seed=seed)
            for kind in reg["contract"]["models"]:
                prefix = f"tf_{minutes}m_origin_{number:02d}_{kind}"
                meta = read(OUT / (prefix + ".json"))
                if digest(OUT / (prefix + ".joblib")) != meta["model_sha256"]: raise ValueError("model SHA mismatch")
                model = joblib.load(OUT / (prefix + ".joblib"))
                pc = d.predict(model, bank["features"][cal].reshape(-1, 12)).reshape(-1, 8)
                train_labels = d.signs(bank["closes"][train], bank["closes"][train + 1]).reshape(-1)
                train_x = bank["features"][train].reshape(-1, 12)[train_labels != 0]
                if not np.array_equal(train_x.mean(axis=0), model["mean"]): raise ValueError("train-only scaler mean mismatch")
                with np.load(OUT / (prefix + ".npz"), allow_pickle=False) as saved:
                    indices = np.searchsorted(bank["times"], saved["times"])
                    reference = np.sign(bank["closes"][indices + 1] - bank["closes"][indices]).astype(np.int8)
                    if not np.array_equal(reference, saved["labels"]): raise ValueError("independent label audit failed")
                    for j, symbol in enumerate(reg["contract"]["symbols"]):
                        for k, threshold in enumerate(meta["thresholds"][symbol]):
                            recalibrated = d.calibrate(pc[:, j], bank["times"][cal], symbol,
                                reg["contract"]["coverage_targets"][k], reg["contract"]["seed"])
                            if threshold != recalibrated: raise ValueError("calibration-only threshold audit failed")
                            expected = d.emit(saved["probability"][:, j], saved["times"], symbol, threshold, reg["contract"]["seed"])
                            if not np.array_equal(expected, saved["masks"][k, :, j]): raise ValueError("threshold audit failed")
                            threshold_checked += 1
                if meta["fold"]["train_latest_label_ms"] > meta["fold"]["calibration_start_ms"] - 7 * minutes * 60000:
                    raise ValueError("purge audit failed")
                if identity(bank["times"][train]) != meta["fold"]["train_times_sha256"]:
                    raise ValueError("fold audit failed")
                checked += 1
    write(OUT / "audit.json", {"models": checked, "thresholds": threshold_checked,
        "independent_price_difference_labels": True, "purge": True, "frozen_code": True,
        "direct_checks": d.direct_checks(), "registration_sha256": reg["sha256"]})
    print(json.dumps({"audited_models": checked, "audited_thresholds": threshold_checked}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["init", "fit", "score", "report", "audit", "checks"])
    parser.add_argument("--origin", type=int)
    args = parser.parse_args()
    if args.stage == "fit": fit_origin(args.origin)
    elif args.stage == "checks": print(json.dumps(d.direct_checks()))
    else: globals()[args.stage]()
