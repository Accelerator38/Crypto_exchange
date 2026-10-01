"""Verify the separately labelled tenth origin and summarize 9+1 outcomes.

Does not make the tenth origin prospective or run the model again.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NINE = ROOT / "Reports/Exia/Genetic_Primus/ohlcv_tail_20260928/verification_and_summary.json"
DIAG = ROOT / "Reports/Exia/Genetic_Primus/september_final_diagnostic_20261001"
RESULT = DIAG / "origin_10_diagnostic.json"
MANIFEST = DIAG / "data_manifest_origin_10_diagnostic.json"
CONFIG = DIAG / "diagnostic_contract.json"
OUT = DIAG / "verification_and_summary.json"
NAMES = ("GA_overlay_f7a4ae11", "EMA_TrendConsensus", "NoTrade")
NINE_SHA = "d1670cb5b3ed239104f04cb3ca46b6ebe54b1f3d4358266bff82126a94795a3e"
RESULT_SHA = "92e8e07a37ec6ba44da77b78a77d84cf75ede6de77154f1fb8562db25393dafb"
MANIFEST_SHA = "93a019fb702d6bf9d9552e651f6129f6e59b93a4e5662317e8b5affe91a257d0"
CONFIG_SHA = "e92a7dffc4ec01a58147afb764760bf5a2aaee59835319c25bd823b9228287bb"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compound(values):
    require(all(math.isfinite(x) and x > -1 for x in values), "invalid return")
    return math.prod(1 + x for x in values) - 1


def main():
    require(not OUT.exists(), "verification already exists")
    for path, expected in ((NINE, NINE_SHA), (RESULT, RESULT_SHA), (MANIFEST, MANIFEST_SHA), (CONFIG, CONFIG_SHA)):
        require(sha(path) == expected, "pinned artifact changed")
    nine, result, manifest, config = map(read, (NINE, RESULT, MANIFEST, CONFIG))
    require(nine["origins_completed"] == 9 and nine["ledger_rows_verified"] == 7776, "nine-origin verification")
    require(config["data_already_revealed"] is True and
            config["selection_use"] == "engineering_diagnostic_only_not_validation_not_tuning_not_promotion", "diagnostic label")
    require(result["diagnostic_only"] is True and result["prospective_validation"] is False
            and result["promotion_eligible"] is False and result["search_evaluations"] == 0, "result label/search")
    require(result["contract_sha256"] == CONFIG_SHA and result["manifest_sha256"] == MANIFEST_SHA
            and result["data_snapshot_sha256"] == manifest["snapshot_sha256"], "result lineage")
    require(result["safety"] == config["safety"] and not any(result["safety"].values()), "safety")
    require(manifest["data_already_revealed"] is True and manifest["safety"] == config["safety"], "manifest label")
    require(result["origin_index"] == 9 and result["purge_bars"] == 7
            and result["origin_start"] == 1790553600000 and result["origin_end_exclusive"] == 1790812800000
            and result["origin_start"] - result["train_end_exclusive"] == 7 * 14400000, "origin/purge")
    require(nine["windows"][-1]["end_exclusive_ms"] == result["origin_start"], "adjacency")
    for item in manifest["snapshot_files"]:
        path = ROOT / item["path"]
        require(sha(path) == item["sha256"] and path.stat().st_size == item["size_bytes"], "snapshot changed")
    details, ledger_count = {}, 0
    for cost in ("normal", "stress"):
        block = result["evaluations"][cost]
        candidates = block["candidates"]
        require(set(candidates) == set(NAMES), "candidates")
        details[cost] = {}
        for name in NAMES:
            c, m = candidates[name], candidates[name]["metrics"]
            require(len(c["ledger"]) == m["decision_count"] == 144 and len(c["daily_returns"]) == 3
                    and len(c["portfolio"]) == 18 and len(c["trades"]) == m["trades"], "candidate counts")
            require(not any(m[k] for k in ("market_readiness", "orders_enabled", "promotion_authority")), "promotion flags")
            for row in c["ledger"]:
                require(row["model_sha256"] == c["model_sha256"]
                        and row["data_snapshot_sha256"] == result["data_snapshot_sha256"]
                        and row["train_rows_sha256"] == result["train_rows_sha256"]
                        and row["scaler_fit_values_sha256"] == result["scaler"]["fit_values_sha256"]
                        and row["execution_timestamp"] >= row["signal_timestamp"], "ledger lineage")
                ledger_count += 1
            checks = [compound(c["daily_returns"]), compound([r["net_return"] for r in c["portfolio"]]),
                      c["portfolio"][-1]["equity"] - 1, m["final_equity"] - 1,
                      math.fsum(t["net_pnl"] for t in c["trades"])]
            require(all(math.isclose(x, m["net_return"], rel_tol=1e-10, abs_tol=1e-12) for x in checks), "PnL")
            require(m["maximum_reconciliation_error"] <= 1e-10, "accounting")
            if name == "NoTrade":
                require(m["net_return"] == 0 and m["trades"] == 0, "NoTrade not flat")
            nine_m = nine["aggregates"]["all_nine"][cost][name]
            details[cost][name] = {"diagnostic_return": m["net_return"],
                                  "descriptive_9_plus_1_return": (1 + nine_m["chained_return"]) * (1 + m["net_return"]) - 1,
                                  "positive_origins_descriptive": nine_m["positive_origins"] + int(m["net_return"] > 0),
                                  "trades_descriptive": nine_m["trades"] + m["trades"]}
        for ref, field in ((NAMES[1], "paired_overlay_minus_ema"), (NAMES[2], "paired_overlay_minus_no_trade")):
            expected = [a - b for a, b in zip(candidates[NAMES[0]]["daily_returns"], candidates[ref]["daily_returns"])]
            require(expected == block[field], "paired daily returns")
    require(ledger_count == 864, "diagnostic ledger count")
    ga, ema = NAMES[:2]
    wins = {cost: sum(w["costs"][cost][ga]["net_return"] > w["costs"][cost][ema]["net_return"] for w in nine["windows"])
            + int(result["evaluations"][cost]["candidates"][ga]["metrics"]["net_return"] >
                  result["evaluations"][cost]["candidates"][ema]["metrics"]["net_return"])
            for cost in ("normal", "stress")}
    require(not (ROOT / "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609/origin_10.json").exists(),
            "diagnostic cannot masquerade as prospective origin 10")
    summary = {"schema": "v8-clean-september-final-diagnostic-verification/1", "checker_sha256": sha(Path(__file__)),
               "nine_origin_summary_sha256": NINE_SHA, "diagnostic_result_sha256": RESULT_SHA,
               "diagnostic_manifest_sha256": MANIFEST_SHA, "diagnostic_contract_sha256": CONFIG_SHA,
               "verified_prospective_origins": 9, "diagnostic_origins": 1,
               "ledger_rows_verified": nine["ledger_rows_verified"] + ledger_count,
               "ga_beats_ema_origins_descriptive": wins, "details": details,
               "prospective_validation_complete": False, "promotion_gate_evaluated": False,
               "qualification_allowed": False, "safety": config["safety"],
               "return_semantics": "9 committed origins plus 1 separate post-period diagnostic; geometrically chained closed windows"}
    with OUT.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print(json.dumps({"status": "VERIFIED_DIAGNOSTIC_NO_PROMOTION", "ledger_rows": summary["ledger_rows_verified"],
                      "summary_sha256": sha(OUT)}, sort_keys=True))


if __name__ == "__main__":
    main()
