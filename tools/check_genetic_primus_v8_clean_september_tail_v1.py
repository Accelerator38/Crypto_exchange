"""Read-only verification of nine frozen origins; exclusive summary output.

Does not import an evaluator, fetch prices, retrain models, or grant promotion.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609"
OUT = ROOT / "Reports/Exia/Genetic_Primus/ohlcv_tail_20260928"
DATA = ROOT / "Retrodate/genetic_primus_v8_clean_ohlcv_tail_20260922_20260927"
NAMES = ("GA_overlay_f7a4ae11", "EMA_TrendConsensus", "NoTrade")
PINS = (
    "4e2b05988cdc8224eee370658eaf5e7a64db60ef18f260a0936da5e29d2ff6f8",
    "8303592a39609f6d9712017cb3dbf658ef45d75251f3d2463105653ba8577738",
    "d9d50d88da287388ac00785346b2dd06c97fef88314dd731d84ce2eca9211d4d",
    "a886086c8930a29b4cfd41f7e2ae8e94125226906e2e37adee0f4eb739b59db1",
    "861c35eb537e181b73e5fb012242843acf7c4ccd95843d2885597c95135906a0",
    "ef53b2725e0523c6f6a5be3cba0159d555f0cd366261f649d4e6fa0dcf004e74",
    "9b52b99b94ca21fc2bd56ad25bc8a30c638b737cd25cd66edf44bd45fb28fb3f",
    "7b5cda1448c0981f73f0cd5e69f16113aed2cc0b091f76eb766626a947f4b882",
    "87b11dee0878cefe39388ee0dedeccd20af4cdadd28880bafc0cfdf97b2fe482",
)
CONTRACT_SHA = "0002201ea664d868a33cdc7e4c8ceb6630bf7a072fef8eccee86927ed4f6488f"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compound(values):
    require(all(math.isfinite(x) and x > -1 for x in values), "invalid return")
    return math.prod(1 + x for x in values) - 1


def main():
    target = OUT / "verification_and_summary.json"
    require(not target.exists(), "summary already exists; never overwrite")
    contract_path = ROOT / "configs/genetic_primus_prospective_3d_v1.json"
    require(sha(contract_path) == CONTRACT_SHA, "contract changed")
    contract = read(contract_path)
    seal = read(DATA / "dataset_seal_v1.json")
    body = {k: v for k, v in seal.items() if k != "seal_sha256"}
    seal_sha = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(seal_sha == seal["seal_sha256"] == "48a6bd5a8b91a14d0f9ea93949ca18f8d7cccc3122456feee73ce7298a70be95", "seal hash")
    require(sha(ROOT / "tools/fetch_genetic_primus_v8_clean_ohlcv_tail_v1.py") == seal["code_sha256"], "intake source changed")
    for item in seal["files"] + seal["manifests"]:
        require(sha(ROOT / item["path"]) == item["sha256"], "sealed data changed")
    windows, ledger_rows = [], 0
    for origin, expected_sha in enumerate(PINS, 1):
        path = RESULTS / f"origin_{origin:02d}.json"
        require(sha(path) == expected_sha, "result changed")
        d = read(path)
        manifest_path = RESULTS / f"data_manifest_origin_{origin:02d}.json"
        manifest = read(manifest_path)
        require(d["contract_sha256"] == CONTRACT_SHA and d["manifest_sha256"] == sha(manifest_path)
                and d["data_snapshot_sha256"] == manifest["snapshot_sha256"], "result lineage")
        require(d["safety"] == contract["safety"] and not any(d["safety"].values())
                and d["search_evaluations"] == 0 and d["origin_index"] == origin - 1, "safety/origin/search")
        require(d["purge_bars"] == 7 and d["origin_start"] - d["train_end_exclusive"] == 7 * 14400000
                and d["origin_end_exclusive"] - d["origin_start"] == 259200000, "purge/horizon")
        if windows:
            require(windows[-1]["end_exclusive_ms"] == d["origin_start"], "nonadjacent origins")
        for source in manifest["snapshot_files"]:
            require(sha(ROOT / source["path"]) == source["sha256"], "input file changed")
        window = {"origin": origin, "start_ms": d["origin_start"], "end_exclusive_ms": d["origin_end_exclusive"],
                  "result_sha256": expected_sha, "manifest_sha256": sha(manifest_path),
                  "scaler_fit_values_sha256": d["scaler"]["fit_values_sha256"], "costs": {}}
        for cost in ("normal", "stress"):
            candidates = d["evaluations"][cost]["candidates"]
            require(set(candidates) == set(NAMES), "candidate set")
            window["costs"][cost] = {}
            for name in NAMES:
                c = candidates[name]
                metrics = c["metrics"]
                require(len(c["ledger"]) == metrics["decision_count"] == 144
                        and len(c["daily_returns"]) == 3 and len(c["portfolio"]) == 18, "decision/return counts")
                require(not any(metrics[k] for k in ("market_readiness", "orders_enabled", "promotion_authority")), "promotion disabled")
                for row in c["ledger"]:
                    require(row["model_sha256"] == c["model_sha256"]
                            and row["data_snapshot_sha256"] == d["data_snapshot_sha256"]
                            and row["train_rows_sha256"] == d["train_rows_sha256"]
                            and row["scaler_fit_values_sha256"] == d["scaler"]["fit_values_sha256"]
                            and row["execution_timestamp"] >= row["signal_timestamp"], "ledger lineage/chronology")
                    ledger_rows += 1
                checks = [compound(c["daily_returns"]), compound([r["net_return"] for r in c["portfolio"]]),
                          c["portfolio"][-1]["equity"] - 1, metrics["final_equity"] - 1,
                          math.fsum(t["net_pnl"] for t in c["trades"])]
                require(all(math.isclose(x, metrics["net_return"], rel_tol=1e-10, abs_tol=1e-12) for x in checks), "PnL reconciliation")
                require(len(c["trades"]) == metrics["trades"] and metrics["maximum_reconciliation_error"] <= 1e-10, "trade accounting")
                if name == "NoTrade":
                    require(metrics["net_return"] == 0 and metrics["trades"] == 0, "NoTrade not flat")
                window["costs"][cost][name] = {k: metrics[k] for k in ("net_return", "max_drawdown", "trades", "total_fees")}
            for ref, field in ((NAMES[1], "paired_overlay_minus_ema"), (NAMES[2], "paired_overlay_minus_no_trade")):
                expected = [a - b for a, b in zip(candidates[NAMES[0]]["daily_returns"], candidates[ref]["daily_returns"])]
                require(expected == d["evaluations"][cost][field], "paired returns")
        windows.append(window)
    aggregates = {}
    for label, subset in (("new_two", windows[-2:]), ("all_nine", windows)):
        aggregates[label] = {}
        for cost in ("normal", "stress"):
            aggregates[label][cost] = {name: {
                "chained_return": compound([w["costs"][cost][name]["net_return"] for w in subset]),
                "positive_origins": sum(w["costs"][cost][name]["net_return"] > 0 for w in subset),
                "trades": sum(w["costs"][cost][name]["trades"] for w in subset)} for name in NAMES}
            aggregates[label][cost]["GA_beats_EMA_origins"] = sum(
                w["costs"][cost][NAMES[0]]["net_return"] > w["costs"][cost][NAMES[1]]["net_return"] for w in subset)
    require(ledger_rows == 7776, "ledger total")
    result = {"schema": "v8-clean-september-tail-verification/1", "checker_sha256": sha(Path(__file__)),
              "contract_sha256": CONTRACT_SHA, "data_seal_sha256": seal_sha,
              "windows": windows, "aggregates": aggregates, "ledger_rows_verified": ledger_rows,
              "safety": contract["safety"], "origins_completed": 9, "origins_planned": 10,
              "promotion_gate_evaluated": False, "qualification_allowed": False,
              "return_semantics": "geometric chain of independently closed three-day windows; not continuous realistic execution"}
    OUT.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": "VERIFIED_NO_PROMOTION", "origins": 9, "ledger_rows": ledger_rows,
                      "summary_sha256": sha(target)}))


if __name__ == "__main__":
    main()
