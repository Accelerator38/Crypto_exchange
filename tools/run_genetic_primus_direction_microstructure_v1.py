"""Single-day local gzip experiment, resumable bounded parsing. No network IO."""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["MPLBACKEND"] = "Agg"
import argparse
import csv
import gzip
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from exia.genetic_primus import direction_selective_v1 as d
from exia.genetic_primus import direction_microstructure_v1 as micro
from run_genetic_primus_direction_selective_v1 import digest, identity, read, write

OUT = ROOT / "Reports/Exia/Genetic_Primus/direction_microstructure_v1_20261013"
DOWNLOADS = Path("C:/Users/anton/Downloads")
BOOK = DOWNLOADS / "bitget-futures_incremental_book_L2_2024-12-01_BTCUSDT.csv.gz"
TRADES = DOWNLOADS / "bitget-futures_trades_2024-12-01_BTCUSDT.csv.gz"
START_US = 1733011200000000
SEED = 20261013
CODE = [Path(__file__), ROOT / "src/exia/genetic_primus/direction_microstructure_v1.py",
        ROOT / "src/exia/genetic_primus/direction_selective_v1.py"]


def checkpoint(value):
    target = OUT / "checkpoint.json"
    temporary = OUT / "checkpoint.tmp"
    if temporary.exists(): raise ValueError("incomplete previous checkpoint")
    temporary.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    temporary.replace(target)


def init():
    if OUT.exists(): raise ValueError("exclusive engineering registration required")
    OUT.mkdir()
    cfg = read(ROOT / "configs/genetic_primus_direction_selective_v1.json")
    reg = write(OUT / "registration.json", {"contract": cfg["microstructure"], "safety": cfg["safety"],
        "source_pins": {str(p): digest(p) for p in (BOOK, TRADES)},
        "code_pins": {str(p): digest(p) for p in CODE}, "seed": SEED,
        "feature_sets": {"price_only": 6, "price_and_microstructure": 16},
        "models": ["logistic", "boosting"], "fit_budget": 12,
        "coverage_targets": [1, .5, .2, .1], "minimum_leaf": 16,
        "missing_book_policy": "same eligible current-feature mask for both feature sets; no synthetic book filling",
        "bootstrap": "not inferentially qualified: one day only; no accuracy improvement claim",
        "target": "sign(last_received_trade_price[t+h] - last_received_trade_price[t]); h=1/10/30 minutes",
        "training_thresholds": "first 60% day fit, next 20% calibrate; two 7*h minute maturity gaps",
        "independent_validation": False})
    with gzip.open(BOOK, "rb") as f:
        header = f.readline().decode().strip()
        offset = f.tell()
    if header != "exchange,symbol,timestamp,local_timestamp,is_snapshot,side,price,amount":
        raise ValueError("known Tardis L2 header required")
    checkpoint({"offset": offset, "lines": 0, "pending": None,
                "book": micro.ReceiptBook(START_US).state, "finished": False})
    print(json.dumps({"registered": reg["sha256"], "fits": 12, "engineering_only": True}))


def verify_code():
    reg = read(OUT / "registration.json")
    for p, sha in reg["code_pins"].items():
        if digest(p) != sha: raise ValueError("registered engineering code changed")
    return reg


def parse_book(limit):
    verify_code()
    if not 1 <= limit <= 300000: raise ValueError("one bounded chunk up to 300000 CSV rows")
    c = read(OUT / "checkpoint.json")
    if c["finished"]: raise ValueError("book already completed")
    b = micro.ReceiptBook(START_US, c["book"])
    pending = c["pending"]
    processed = 0
    with gzip.open(BOOK, "rb") as f:
        f.seek(c["offset"])
        for _ in range(limit):
            raw = f.readline()
            if not raw:
                if pending: b.batch(pending["received_us"], pending["snapshot"], pending["updates"])
                b.flush(START_US + d.DAY * 1000)
                pending = None; c["finished"] = True; break
            a = raw.decode().strip().split(",")
            if len(a) != 8 or a[0] != "bitget-futures" or a[1] != "BTCUSDT":
                raise ValueError("exact registered venue/symbol required")
            stamp, received = int(a[2]), int(a[3])
            if not START_US <= received < START_US + d.DAY * 1000:
                raise ValueError("receipt outside registered day")
            key = [stamp, received, a[4]]
            if pending is None or key != pending["key"]:
                if pending: b.batch(pending["received_us"], pending["snapshot"], pending["updates"])
                if a[4] not in ("true", "false"): raise ValueError("snapshot boolean required")
                pending = {"key": key, "received_us": received, "snapshot": a[4] == "true", "updates": []}
            pending["updates"].append([a[5], float(a[6]), float(a[7])])
            processed += 1
        c["offset"] = f.tell()
    c.update({"lines": c["lines"] + processed, "book": b.state, "pending": pending})
    checkpoint(c)
    print(json.dumps({"chunk_rows": processed, "total_rows": c["lines"],
                      "minutes": len(b.state["rows"]), "finished": c["finished"]}))


def trades_and_features():
    reg = verify_code(); c = read(OUT / "checkpoint.json")
    if not c["finished"]: raise ValueError("complete finite book required")
    for p, sha in reg["source_pins"].items():
        if digest(p) != sha: raise ValueError("source changed during bounded parsing")
    bars, notional, count = np.full((1440, 5), np.nan), np.zeros((1440, 2)), np.zeros(1440, int)
    last_receipt = 0; rows = 0
    with gzip.open(TRADES, "rt", encoding="utf-8", newline="") as f:
        for a in csv.DictReader(f):
            if a["symbol"] != "BTCUSDT" or a["exchange"] != "bitget-futures": raise ValueError("venue/symbol mismatch")
            received = int(a["local_timestamp"])
            if received < last_receipt: raise ValueError("trades receipt time reversed")
            last_receipt = received
            i = (received - START_US) // micro.MINUTE_US
            if not 0 <= i < 1440: raise ValueError("trade outside day")
            price, amount = float(a["price"]), float(a["amount"])
            if not np.isfinite([price, amount]).all() or price <= 0 or amount < 0: raise ValueError("valid trade required")
            if count[i] == 0: bars[i] = [price, price, price, price, 0]
            bars[i, 1] = max(bars[i, 1], price); bars[i, 2] = min(bars[i, 2], price)
            bars[i, 3] = price; bars[i, 4] += amount; count[i] += 1
            if a["side"] not in ("buy", "sell"): raise ValueError("aggressor side required")
            notional[i, 0 if a["side"] == "buy" else 1] += price * amount
            rows += 1
    base, full = micro.feature_bank(bars, notional, count, c["book"]["rows"])
    with (OUT / "minute_bank.npz").open("xb") as f:
        np.savez_compressed(f, times=START_US//1000 + (np.arange(1440)+1)*60000, closes=bars[:, 3],
                            price_only=base, price_and_microstructure=full)
    write(OUT / "data_audit.json", {"book_rows": c["lines"], "trade_rows": rows,
        "valid_book_minutes": sum(r["valid"] for r in c["book"]["rows"]), "missing_trade_minutes": int((count == 0).sum()),
        "eligible_current_features": int(np.isfinite(full).all(axis=1).sum()), "sequence_completeness_verified": False,
        "snapshots": c["book"]["snapshots"], "source_pins": reg["source_pins"],
        "bank_sha256": digest(OUT / "minute_bank.npz"), "synthetic_checks": micro.direct_checks()})
    print(json.dumps({"book_rows": c["lines"], "trade_rows": rows, "eligible_minutes": int(np.isfinite(full).all(axis=1).sum())}))


def fit():
    import joblib
    reg = verify_code(); audit = read(OUT / "data_audit.json")
    if digest(OUT / "minute_bank.npz") != audit["bank_sha256"]: raise ValueError("minute bank changed")
    with np.load(OUT / "minute_bank.npz", allow_pickle=False) as a: bank = {k: a[k] for k in a.files}
    times, closes, full = bank["times"], bank["closes"], bank["price_and_microstructure"]
    eligible = np.isfinite(full).all(axis=1)
    base_ms = START_US // 1000
    cal_start, test_start = base_ms + 864*60000, base_ms + 1152*60000
    rows = []
    for horizon in [1, 10, 30]:
        indices = np.arange(1440-horizon)
        usable = eligible[indices] & np.isfinite(closes[indices+horizon])
        train = indices[usable & (times[indices+horizon] <= cal_start - 7*horizon*60000)]
        cal = indices[usable & (times[indices] >= cal_start) & (times[indices+horizon] <= test_start - 7*horizon*60000)]
        test = indices[usable & (times[indices] >= test_start)]
        if min(len(train), len(cal), len(test)) < 16: raise ValueError("adequate chronological splits required")
        y_train = d.signs(closes[train], closes[train+horizon]); y_test = d.signs(closes[test], closes[test+horizon])
        controls = {"random_coin": np.random.default_rng(SEED+horizon).integers(0, 2, len(test)).astype(bool),
                    "train_majority": np.full(len(test), np.sum(y_train == 1) >= np.sum(y_train == -1)),
                    "momentum": closes[test] >= closes[test-1]}
        for feature_set in ("price_only", "price_and_microstructure"):
            for kind in ("logistic", "boosting"):
                model = d.fit(kind, bank[feature_set][train], y_train, SEED+horizon, minimum_leaf=16)
                pc, pt = d.predict(model, bank[feature_set][cal]), d.predict(model, bank[feature_set][test])
                name = f"h{horizon}_{feature_set}_{kind}"
                model_path = OUT / (name + ".joblib")
                if model_path.exists(): raise ValueError("no repeated engineering fits")
                joblib.dump(model, model_path, compress=3)
                thresholds = [d.calibrate(pc, times[cal], "BTCUSDT", v, SEED) for v in [1, .5, .2, .1]]
                masks = np.stack([d.emit(pt, times[test], "BTCUSDT", t, SEED) for t in thresholds])
                with (OUT / (name + ".npz")).open("xb") as f:
                    np.savez_compressed(f, times=times[test], labels=y_test, probability=pt, masks=masks,
                                        model_sha256=np.array(digest(model_path)))
                for k, coverage in enumerate([1, .5, .2, .1]):
                    rows.append({"horizon_minutes": horizon, "feature_set": feature_set, "model": kind,
                        "coverage_target": coverage, **d.evaluate(pt >= .5, y_test, masks[k]),
                        "controls": {n: d.evaluate(p, y_test, masks[k]) for n, p in controls.items()},
                        "n_train": len(train), "n_calibration": len(cal), "threshold": thresholds[k],
                        "scaler_train_only": True, "model_sha256": digest(model_path)})
    write(OUT / "results.json", {"rows": rows, "fit_count": 12, "one_day_only": True,
        "independent_improvement": False, "overlapping_labels": True, "safety": reg["safety"]})
    print(json.dumps({"fits": 12, "rows": len(rows), "improvement_claim": False}))


def report():
    import matplotlib.pyplot as plt
    rows = read(OUT / "results.json")["rows"]
    lines = ["# Микроструктурный инженерный эксперимент", "",
        "BTCUSDT, Bitget futures, 1 декабря 2024. Один день, не независимая проверка устойчивости и не подтверждение улучшения.",
        "Локальные trades и incremental L2. Стакан восстанавливается атомарными пакетами. Используется clock получения; данные ровно на границе решения исключаются. Номера последовательности отсутствуют: незаметную потерю delta доказать нельзя.",
        "Цель: изменение последней полученной цены сделки через 1/10/30 минут. Решение каждую минуту; labels 10/30m перекрываются. Эта постановка отличается от OHLCV benchmark и показана отдельно.",
        "Одинаковые допустимые строки для price-only и price+microstructure. Первые 60% дня — train, следующие 20% — calibration, последние 20% — test; purge 7*h минут на обеих границах. Порог по прошлой калибровке, без test quantiles.",
        "Небольшие выборки, единственный актив/день. Доверительные интервалы и заявления о статистическом улучшении не выдаются.", "",
        "| Горизонт | Признаки | Модель | Test n | Accuracy 100% | Random | Majority | Momentum |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["coverage_target"] == 1:
            controls = [r["controls"][n]["accuracy_all"] for n in ("random_coin", "train_majority", "momentum")]
            lines.append(f"| {r['horizon_minutes']}m | {r['feature_set']} | {r['model']} | {r['n_emitted']} | {r['accuracy_all']:.2%} | {controls[0]:.2%} | {controls[1]:.2%} | {controls[2]:.2%} |")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, h in zip(axes, [1, 10, 30]):
        for fs in ("price_only", "price_and_microstructure"):
            for model in ("logistic", "boosting"):
                rr = sorted([r for r in rows if r["horizon_minutes"] == h and r["feature_set"] == fs and r["model"] == model and r["accuracy_all"] is not None], key=lambda a: a["coverage_target"])
                ax.plot([100*r["coverage_actual"] for r in rr], [100*r["accuracy_all"] for r in rr], "o-", label=f"{fs}/{model}")
        ax.set_title(f"h={h}m, overlapping labels"); ax.grid(alpha=.2)
    axes[0].legend(fontsize=6); fig.supxlabel("Actual coverage, %"); fig.supylabel("Accuracy including flat, %")
    fig.suptitle("BTC single-day engineering; no independent improvement claim"); fig.tight_layout()
    fig.savefig(OUT / "microstructure_accuracy.png", dpi=140); plt.close(fig)
    lines += ["", "![Microstructure](microstructure_accuracy.png)", "", "Все 48 строк с отказами и flat находятся в results.json. Разрешения runtime/network/orders/promotion остаются false."]
    (OUT / "REPORT_RU.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "figures": 1, "report": "REPORT_RU.md"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["init", "book", "features", "fit", "report", "checks"])
    parser.add_argument("--rows", type=int, default=300000)
    args = parser.parse_args()
    if args.stage == "book": parse_book(args.rows)
    elif args.stage == "features": trades_and_features()
    elif args.stage == "checks": print(json.dumps(micro.direct_checks()))
    else: globals()[args.stage]()
