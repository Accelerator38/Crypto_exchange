"""Short stdlib-only synthetic label/matrix checks, not a model experiment."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PURE = "src/exia/genetic_primus/synthetic_label_matrix_v1.py"
SELF = "tools/check_genetic_primus_synthetic_label_matrix_v1.py"
CONFIG = "configs/genetic_primus_synthetic_label_matrix_contract_v1.json"
DOC = "docs/GENETIC_PRIMUS_SYNTHETIC_LABEL_MATRIX_V1.md"
PREVIOUS = "Reports/Exia/Genetic_Primus/origin_feature_prefix_v1_20261007"
DEST = "Reports/Exia/Genetic_Primus/synthetic_label_matrix_v1_20261007"
prior = runpy.run_path(str(ROOT / "tools/check_genetic_primus_origin_feature_prefix_v1.py"))
old, api, quality = prior["old"], prior["api"], prior["quality"]
pure = runpy.run_path(str(ROOT / PURE))
ASSERTIONS = 0


def check(ok):
    global ASSERTIONS
    if not ok:
        raise AssertionError("matrix regression")
    ASSERTIONS += 1


def rejected(call):
    try:
        call()
    except ValueError:
        check(True)
    else:
        raise AssertionError("invalid label/matrix accepted")


def materialize(candles, request, contract):
    return pure["materialize"](candles, request, contract, prior["pure"], prior["timing"], api, prior["kernel"])


def repin(candles, request):
    request["expected_snapshot_sha256"] = prior["pure"]["sha"](candles)
    prior["bind_pins"](candles, request)


def regressions(contract):
    b = prior["timing"]["BAR_MS"]
    check(not pure["assess"](contract)["net_objective_registered"])
    for entry, exit_price, expected in ((100, 110, 0.1), (100, 90, -0.1), (100, 100, 0)):
        check(abs(pure["gross_open_target"](entry, exit_price) - expected) < 1e-12)
    for value in (None, True, 0, -1, "100", float("nan"), float("inf"), 10**1000):
        rejected(lambda value=value: pure["gross_open_target"](value, 100))
        rejected(lambda value=value: pure["gross_open_target"](100, value))
    rejected(lambda: pure["gross_open_target"](1e-308, 1e308))
    examples = []
    for lag in (0, 1, 120000, b, b + 1):
        candles, request = prior["fixture"](lag=lag)
        result = materialize(candles, request, contract)
        manifest = result["manifest"]
        check(len(result["x"]) == len(result["y"]) == manifest["scaler_plan"]["row_count"])
        check(manifest["shape"] == [len(result["y"]), 2])
        check(manifest["synthetic_target_counts"]["positive"] == len(result["y"]))
        for row in manifest["training_rows"]:
            check(row["label_available_at_ms"] <= row["training_cutoff_ms"])
            check(row["exit_open_ms"] - row["entry_open_ms"] == 18 * b)
            check(row["feature_available_at_ms"] <= row["decision_asof_ms"])
        for row in result["deployment"]:
            check(set(row) == {"symbol", "decision_asof_ms", "features", "proposal_long", "feature_available_at_ms"})
            check(set(row["features"]) == set(pure["FEATURES"]))
        mutated = copy.deepcopy(candles)
        for row in mutated:
            if row["available_at_ms"] > request["decision_asof_ms"]:
                row["open"] = "UNAVAILABLE_LABEL_SENTINEL"
                row["close"] = None
        changed_request = copy.deepcopy(request)
        changed_request["expected_snapshot_sha256"] = prior["pure"]["sha"](mutated)
        changed = materialize(mutated, changed_request, contract)
        check(changed["x"] == result["x"] and changed["y"] == result["y"])
        check(changed["deployment"] == result["deployment"])
        check(changed["manifest"]["paired_xy_sha256"] == manifest["paired_xy_sha256"])
        rejected(lambda: materialize(mutated, request, contract))
        examples.append({"synthetic_lag_ms": lag, "shape": manifest["shape"],
                         "distinct_decisions": manifest["distinct_training_decisions"],
                         "target_counts": manifest["synthetic_target_counts"]})
    candles, request = prior["fixture"]()
    baseline = materialize(candles, request, contract)
    for kind in ("flat", "declining", "mixed"):
        rows, req = prior["fixture"](flat=kind == "flat")
        if kind != "flat":
            for row in rows:
                i = row["bar_open_ms"] // b
                value = (200 - i * 0.3 if kind == "declining" else 100 + (i % 37) * 0.25)
                value += 7 if row["symbol"] == "SYN_B" else 0
                row.update(open=value, high=value + 1, low=value - 1, close=value)
            repin(rows, req)
        result = materialize(rows, req, contract)
        check(len(result["y"]) == 46)
        counts = result["manifest"]["synthetic_target_counts"]
        check(sum(counts.values()) == 46)
        if kind == "flat":
            check(counts["zero"] == 46 and all(not r["proposal_long"] for r in result["deployment"]))
        elif kind == "declining":
            check(counts["negative"] == 46 and all(not r["proposal_long"] for r in result["deployment"]))
        else:
            check(counts["negative"] > 0 and counts["positive"] > 0)
        examples.append({"synthetic_path": kind, "shape": result["manifest"]["shape"], "target_counts": counts})
    # Re-pinned historical open affects y, not close-only x or deployment values.
    altered = copy.deepcopy(candles)
    next(r for r in altered if r["bar_open_ms"] == 73 * b and r["symbol"] == "SYN_A")["open"] += 0.5
    req = copy.deepcopy(request)
    repin(altered, req)
    changed = materialize(altered, req, contract)
    check(changed["y"] != baseline["y"])
    check(changed["x"] == baseline["x"])
    check(changed["deployment"] == baseline["deployment"])
    check(changed["manifest"]["paired_xy_sha256"] != baseline["manifest"]["paired_xy_sha256"])
    # An earlier nominal open is never substituted for the strictly delayed open.
    check(min(r["entry_open_ms"] for r in baseline["manifest"]["training_rows"]) == 51 * b)
    delayed_req = copy.deepcopy(request)
    delayed_req["execution_delay_ms"] = b - 1
    prior["bind_pins"](candles, delayed_req)
    delayed = materialize(candles, delayed_req, contract)
    check(delayed["y"] == baseline["y"])
    late = copy.deepcopy(candles)
    next(r for r in late if r["bar_open_ms"] == 91 * b and r["symbol"] == "SYN_A")["available_at_ms"] = 92 * b + 1
    late_req = copy.deepcopy(request)
    repin(late, late_req)
    late_result = materialize(late, late_req, contract)
    check(len(late_result["y"]) == 44)
    check(all(r["decision_asof_ms"] != 72 * b for r in late_result["row_ids"]))
    # Different symbol publication times remain valid when the panel is mature.
    within = copy.deepcopy(candles)
    next(r for r in within if r["bar_open_ms"] == 69 * b and r["symbol"] == "SYN_A")["available_at_ms"] = 70 * b + 1
    within_req = copy.deepcopy(request)
    within_req["candidate_decisions_ms"] = [50 * b, 72 * b]
    repin(within, within_req)
    within_result = materialize(within, within_req, contract)
    check(len(within_result["y"]) == 4)
    first_pair = within_result["manifest"]["training_rows"][:2]
    check(all(r["panel_label_available_at_ms"] == 70 * b + 1 for r in first_pair))
    check(first_pair[0]["label_available_at_ms"] != first_pair[1]["label_available_at_ms"])
    empty_req = copy.deepcopy(request)
    empty_req["candidate_decisions_ms"] = [98 * b]
    prior["bind_pins"](candles, empty_req)
    empty = materialize(candles, empty_req, contract)
    check(empty["x"] == [] and empty["y"] == [] and empty["manifest"]["shape"] == [0, 2])
    rows, req = prior["fixture"](origin_bar=118)
    next_origin = materialize(rows, req, contract)
    check(next_origin["manifest"]["shape"] == [82, 2])
    check(next_origin["manifest"]["paired_xy_sha256"] != baseline["manifest"]["paired_xy_sha256"])
    for key, value in (("target_kind", "net"), ("cost_model", 0), ("drop_losses", True),
                       ("drop_flat_or_abstention_rows", True), ("fill_evidence_verified", True),
                       ("horizon_bars", 17), ("purge_bars", 6), ("bar_ms", True)):
        changed_contract = copy.deepcopy(contract)
        changed_contract[key] = value
        rejected(lambda changed_contract=changed_contract: materialize(candles, request, changed_contract))
    for key in pure["SAFETY"]:
        changed_contract = copy.deepcopy(contract)
        changed_contract["safety"][key] = True
        rejected(lambda changed_contract=changed_contract: pure["assess"](changed_contract))
    return examples, baseline["manifest"]


def build():
    previous = quality["read"](PREVIOUS + "/completion_receipt.json")
    bank = dict(previous["inputs"])
    quality["unchanged"](bank)
    for name, expected in previous["output_identities"].items():
        path = PREVIOUS + "/" + name
        if old["identity"](path) != expected:
            raise ValueError("previous output identity changed")
        bank[path] = expected
    for path in (PREVIOUS + "/completion_receipt.json", PURE, SELF, CONFIG, DOC):
        bank[path] = old["identity"](path)
    contract = quality["read"](CONFIG)
    examples, manifest = regressions(contract)
    if any(name.split(".")[0] in old["BLOCKED"] for name in sys.modules):
        raise ValueError("forbidden module loaded")
    quality["unchanged"](bank)
    result = {"schema": "exia.genetic_primus.synthetic_label_matrix_check/1", "inputs": bank,
              "direct_assertions": ASSERTIONS, "contract_sha256": pure["sha"](contract),
              "examples": examples, "baseline_manifest": manifest,
              "parquet_reads": 0, "scaler_fit_count": 0, "model_fit_count": 0,
              "market_financial_evaluation_count": 0, "synthetic_only": True,
              "market_research_launch_ready": False, "real_money_ready": False, "safety": contract["safety"]}
    body = api["canonical"](result) + b"\n"
    receipt = {"schema": "exia.genetic_primus.synthetic_label_matrix_receipt/1", "inputs": bank,
               "output_identities": {"result.json": {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}},
               "safety": contract["safety"]}
    return result, receipt


def main():
    if sys.argv[1:] not in (["--run"], ["--verify"]):
        raise ValueError("use --run or --verify")
    result, receipt = build()
    if sys.argv[1:] == ["--run"]:
        old["local"](DEST + "/result.json").parent.mkdir(exist_ok=False)
        for name, value in (("result.json", result), ("completion_receipt.json", receipt)):
            old["emit"](DEST + "/" + name, value, api)
        print(f"COMPLETE: assertions={ASSERTIONS}; inputs={len(result['inputs'])}; parquet=0; fit=0")
    else:
        if quality["read"](DEST + "/result.json") != result or quality["read"](DEST + "/completion_receipt.json") != receipt:
            raise ValueError("result/receipt mismatch")
        print(f"VERIFIED: assertions={ASSERTIONS}; receipt OK; parquet=0; fit=0")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("STOPPED: " + type(exc).__name__ + "; no training/network")
        raise SystemExit(1)
