"""Offline audit/summarization of existing origin results; never evaluates models."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609"
OUT = ROOT / "Reports/Exia/Genetic_Primus/ohlcv_backfill_20260923"
DATA = ROOT / "Retrodate/genetic_primus_v8_clean_ohlcv_backfill_20260907_20260921_v2"
NAMES = ("GA_overlay_f7a4ae11", "EMA_TrendConsensus", "NoTrade")
PINS = {
    1: "4e2b05988cdc8224eee370658eaf5e7a64db60ef18f260a0936da5e29d2ff6f8",
    2: "8303592a39609f6d9712017cb3dbf658ef45d75251f3d2463105653ba8577738",
    3: "d9d50d88da287388ac00785346b2dd06c97fef88314dd731d84ce2eca9211d4d",
    4: "a886086c8930a29b4cfd41f7e2ae8e94125226906e2e37adee0f4eb739b59db1",
    5: "861c35eb537e181b73e5fb012242843acf7c4ccd95843d2885597c95135906a0",
    6: "ef53b2725e0523c6f6a5be3cba0159d555f0cd366261f649d4e6fa0dcf004e74",
    7: "9b52b99b94ca21fc2bd56ad25bc8a30c638b737cd25cd66edf44bd45fb28fb3f",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    if path.stat().st_size > 4 * 1024**2:
        raise ValueError("receipt/result size budget exceeded")
    return json.loads(path.read_text("utf-8"))


def compound(values):
    if not values or any(not math.isfinite(x) or x <= -1 for x in values):
        raise ValueError("invalid return sequence")
    return math.prod(1 + x for x in values) - 1


def main():
    target = OUT / "verification_and_summary.json"
    if target.exists():
        raise FileExistsError("summary already exists; no overwrite")
    contract = read(ROOT / "configs/genetic_primus_prospective_3d_v1.json")
    contract_sha = sha(ROOT / "configs/genetic_primus_prospective_3d_v1.json")
    if contract_sha != "0002201ea664d868a33cdc7e4c8ceb6630bf7a072fef8eccee86927ed4f6488f":
        raise ValueError("contract changed")
    windows, ledger_rows = [], 0
    for origin in range(1, 8):
        result_path = RESULTS / f"origin_{origin:02d}.json"
        if sha(result_path) != PINS[origin]:
            raise ValueError("origin result changed")
        data = read(result_path)
        manifest_path = RESULTS / f"data_manifest_origin_{origin:02d}.json"
        manifest = read(manifest_path)
        if (data["contract_sha256"] != contract_sha or data["manifest_sha256"] != sha(manifest_path)
                or data["data_snapshot_sha256"] != manifest["snapshot_sha256"]
                or data["safety"] != contract["safety"] or any(data["safety"].values())
                or data["search_evaluations"] != 0 or data["origin_index"] != origin - 1):
            raise ValueError("origin/manifest/safety identity mismatch")
        if data["purge_bars"] != 7 or data["origin_start"] - data["train_end_exclusive"] != 28 * 3600000:
            raise ValueError("purge mismatch")
        if data["origin_end_exclusive"] - data["origin_start"] != 3 * 86400000:
            raise ValueError("deployment horizon mismatch")
        for source in manifest["snapshot_files"]:
            if sha(ROOT / source["path"]) != source["sha256"]:
                raise ValueError("committed input file changed")
        window = {"origin": origin, "start_ms": data["origin_start"],
                  "end_exclusive_ms": data["origin_end_exclusive"],
                  "result_sha256": PINS[origin], "manifest_sha256": sha(manifest_path),
                  "scaler_fit_values_sha256": data["scaler"]["fit_values_sha256"], "costs": {}}
        for cost in ("normal", "stress"):
            candidates = data["evaluations"][cost]["candidates"]
            if set(candidates) != set(NAMES):
                raise ValueError("candidate set changed")
            window["costs"][cost] = {}
            for name in NAMES:
                candidate = candidates[name]
                metrics = candidate["metrics"]
                if (len(candidate["ledger"]) != 144 or metrics["decision_count"] != 144
                        or len(candidate["daily_returns"]) != 3 or len(candidate["portfolio"]) != 18
                        or metrics["market_readiness"] or metrics["orders_enabled"] or metrics["promotion_authority"]):
                    raise ValueError("candidate coverage/safety mismatch")
                for row in candidate["ledger"]:
                    if (row["model_sha256"] != candidate["model_sha256"]
                            or row["data_snapshot_sha256"] != data["data_snapshot_sha256"]
                            or row["train_rows_sha256"] != data["train_rows_sha256"]
                            or row["scaler_fit_values_sha256"] != data["scaler"]["fit_values_sha256"]
                            or row["execution_timestamp"] < row["signal_timestamp"]):
                        raise ValueError("decision provenance/chronology mismatch")
                    ledger_rows += 1
                checks = [compound(candidate["daily_returns"]),
                          compound([r["net_return"] for r in candidate["portfolio"]]),
                          candidate["portfolio"][-1]["equity"] - 1,
                          math.fsum(t["net_pnl"] for t in candidate["trades"])]
                if any(not math.isclose(x, metrics["net_return"], rel_tol=1e-10, abs_tol=1e-12) for x in checks):
                    raise ValueError("daily/portfolio/trade PnL does not reconcile")
                if len(candidate["trades"]) != metrics["trades"] or metrics["maximum_reconciliation_error"] > 1e-10:
                    raise ValueError("trade count/accounting reconciliation mismatch")
                if name == "NoTrade" and (metrics["net_return"] != 0 or candidate["trades"]):
                    raise ValueError("NoTrade was not flat")
                window["costs"][cost][name] = {k: metrics[k] for k in
                    ("net_return", "max_drawdown", "trades", "total_fees")}
            overlay = candidates[NAMES[0]]["daily_returns"]
            for reference, field in ((NAMES[1], "paired_overlay_minus_ema"), (NAMES[2], "paired_overlay_minus_no_trade")):
                expected = [a - b for a, b in zip(overlay, candidates[reference]["daily_returns"])]
                if expected != data["evaluations"][cost][field]:
                    raise ValueError("paired daily differences mismatch")
        windows.append(window)
    summaries = {}
    for label, selection in (("new_origins_03_07", windows[2:]), ("all_origins_01_07", windows)):
        summaries[label] = {}
        for cost in ("normal", "stress"):
            summaries[label][cost] = {name: {
                "geometrically_chained_window_return": compound([w["costs"][cost][name]["net_return"] for w in selection]),
                "positive_windows": sum(w["costs"][cost][name]["net_return"] > 0 for w in selection),
                "trades": sum(w["costs"][cost][name]["trades"] for w in selection),
            } for name in NAMES}
            summaries[label][cost]["overlay_beats_ema_windows"] = sum(
                w["costs"][cost][NAMES[0]]["net_return"] > w["costs"][cost][NAMES[1]]["net_return"] for w in selection)
    result = {"schema": "genetic_primus.september_backlog_verification/1",
              "contract_sha256": contract_sha, "checker_sha256": sha(Path(__file__)),
              "data_seal_file_sha256": sha(DATA / "dataset_seal_v1.json"),
              "windows": windows, "summaries": summaries, "ledger_rows_checked": ledger_rows,
              "completed_origins": 7, "planned_origins": 10, "search_evaluations": 0,
              "return_semantics": "geometric chaining of separately closed 3-day origins; not continuous live execution",
              "terminal_positions_closed_each_origin": True, "safety": contract["safety"],
              "qualification_eligible": False, "trading_authorized": False,
              "final_promotion_gates_evaluated": False}
    OUT.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"origins_verified": 7, "ledger_rows_checked": ledger_rows,
                      "summary_sha256": sha(target), "promotion": False}))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAIL_CLOSED " + type(exc).__name__ + ": " + str(exc)[:160])
        raise SystemExit(1)
