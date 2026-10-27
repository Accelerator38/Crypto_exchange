"""Deterministic synthetic origin integration; no model/scaler fit or parquet."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PURE = "src/exia/genetic_primus/origin_feature_prefix_v1.py"
SELF = "tools/check_genetic_primus_origin_feature_prefix_v1.py"
DOC = "docs/GENETIC_PRIMUS_ORIGIN_FEATURE_PREFIX_V1.md"
KERNEL = "src/exia/genetic_primus/asof_feature_builder_v1.py"
PREVIOUS = "Reports/Exia/Genetic_Primus/ohlcv_availability_v1_20261007"
DEST = "Reports/Exia/Genetic_Primus/origin_feature_prefix_v1_20261007"
prior = runpy.run_path(str(ROOT / "tools/check_genetic_primus_ohlcv_availability_v1.py"))
old, api = prior["old"], prior["api"]
quality = prior["prior"]
timing = prior["pure"]
pure = runpy.run_path(str(ROOT / PURE))
kernel = runpy.run_path(str(ROOT / KERNEL))
ASSERTIONS = 0


def check(ok):
    global ASSERTIONS
    if not ok:
        raise AssertionError("origin prefix regression")
    ASSERTIONS += 1


def rejected(call):
    try:
        call()
    except ValueError:
        check(True)
    else:
        raise AssertionError("invalid origin input accepted")


def fixture(origin_bar=100, lag=0, flat=False):
    b = timing["BAR_MS"]
    candles = []
    for i in range(150):
        for symbol, offset in (("SYN_A", 0), ("SYN_B", 7)):
            value = 100.0 + offset + (0 if flat else i * 0.2 + (i % 3) * 0.01)
            candles.append({"symbol": symbol, "revision_id": "synthetic_revision_1", "bar_open_ms": i * b,
                            "bar_end_exclusive_ms": (i + 1) * b, "available_at_ms": (i + 1) * b + lag,
                            "open": value, "high": value + 1, "low": value - 1, "close": value, "volume": 1})
    request = {"schema": "exia.genetic_primus.synthetic_origin_context/1", "origin_start_ms": origin_bar * b,
               "training_asof_ms": (origin_bar - 1) * b, "decision_asof_ms": origin_bar * b,
               "history_start_ms": 0, "coverage_start_ms": 0, "coverage_end_ms": 150 * b,
               "ingestion_lag_ms": lag, "execution_delay_ms": 0, "universe": ["SYN_A", "SYN_B"],
               "revision_id": "synthetic_revision_1", "candidate_decisions_ms": list(range(50 * b, (origin_bar - 1) * b, b)),
               "expected_snapshot_sha256": pure["sha"](candles), "prefix_pins": {}}
    bind_pins(candles, request)
    return candles, request


def bind_pins(candles, request):
    """Fixture oracle uses bounded enumeration, independent of closed-prefix arithmetic."""
    b = timing["BAR_MS"]
    wanted = {request["decision_asof_ms"]}
    cutoff = request["training_asof_ms"] - 7 * b
    for candidate in request["candidate_decisions_ms"]:
        window = api["execution_window"](candidate, candidate, request["execution_delay_ms"])
        entry, exit_open = window["planned_entry_open_ms"], window["planned_exit_open_ms"]
        rows = [r for r in candles if r["bar_open_ms"] in (entry, exit_open)]
        if (request["coverage_start_ms"] <= entry and exit_open < request["coverage_end_ms"]
            and len(rows) == 2 * len(request["universe"])
            and max(r["available_at_ms"] for r in rows) <= cutoff):
            wanted.add(candidate)
    request["prefix_pins"] = {str(asof): pure["sha"]([r for r in candles
        if request["history_start_ms"] <= r["bar_open_ms"] and r["bar_open_ms"] + b + request["ingestion_lag_ms"] <= asof])
        for asof in sorted(wanted)}


def prepare(candles, request):
    return pure["prepare_origin"](candles, request, timing, api, kernel)


def regressions():
    b = timing["BAR_MS"]
    examples = []
    for lag in (0, 1, 120000, b, b + 1):
        candles, request = fixture(lag=lag)
        plan = prepare(candles, request)
        check(plan["scaler_plan"]["row_count"] > 0 and not plan["scaler_plan"]["statistics_computed"])
        for row in plan["scaler_train_rows"]:
            check(row["label_available_at_ms"] <= 92 * b)
            check(row["feature_available_at_ms"] <= row["decision_asof_ms"])
            check(row["value_dependencies"]["bars"] == 13 and row["proposal_dependencies"]["bars"] >= 48)
        check(all(r["decision_asof_ms"] < request["training_asof_ms"] for r in plan["scaler_train_rows"]))
        for row in plan["decision_features"]:
            check(row["proposal_dependencies"]["last_bar_ms"] + b + lag <= request["decision_asof_ms"])
        # Changing or poisoning not-yet-visible prices cannot affect features or scaler input bank.
        mutated = copy.deepcopy(candles)
        for row in mutated:
            if row["available_at_ms"] > request["decision_asof_ms"]:
                row["close"] = "UNAVAILABLE_SENTINEL"
                row["volume"] = None
        changed_request = copy.deepcopy(request)
        changed_request["expected_snapshot_sha256"] = pure["sha"](mutated)
        changed = prepare(mutated, changed_request)
        check(changed["decision_features"] == plan["decision_features"])
        check(changed["scaler_plan"] == plan["scaler_plan"])
        check(changed["scaler_train_rows"] == plan["scaler_train_rows"])
        rejected(lambda: prepare(mutated, request))
        examples.append({"synthetic_lag_ms": lag, "scaler_rows": plan["scaler_plan"]["row_count"],
                         "last_train_decision_ms": max(r["decision_asof_ms"] for r in plan["scaler_train_rows"]),
                         "decision_last_dependency_ms": plan["decision_features"][0]["proposal_dependencies"]["last_bar_ms"]})
    candles, request = fixture()
    baseline = prepare(candles, request)
    check(max(r["decision_asof_ms"] for r in baseline["scaler_train_rows"]) == 72 * b)
    check(baseline["scaler_plan"]["row_count"] == 46)
    # Lag-zero features/proposal must match the frozen old pure kernel numerically.
    for asof in (50 * b, 72 * b, 100 * b):
        legacy_request = {"schema": "exia.genetic_primus.synthetic_asof_request/1",
            "evidence_class": kernel["EVIDENCE"], "decision_asof_ms": asof,
            "history_start_ms": 0, "universe": request["universe"], "revision_id": request["revision_id"],
            "expected_prefix_sha256": request["prefix_pins"][str(asof)],
            "kernel_definition_sha256": kernel["sha"](kernel["DEFINITION"])}
        legacy = kernel["build_asof_features"](candles, legacy_request)
        records = (baseline["decision_features"] if asof == 100 * b else
                   [r for r in baseline["scaler_train_rows"] if r["decision_asof_ms"] == asof])
        for older, newer in zip(legacy["rows"], records):
            check(pure["sha"](older["features"]) == newer["feature_values_sha256"])
            check(older["proposal_long"] == newer["proposal_long"])
    # New information at deployment may affect its features but never scaler training rows.
    deployment_changed = copy.deepcopy(candles)
    next(r for r in deployment_changed if r["bar_open_ms"] == 99 * b and r["symbol"] == "SYN_A")["close"] += 0.2
    deployment_request = copy.deepcopy(request)
    deployment_request["expected_snapshot_sha256"] = pure["sha"](deployment_changed)
    deployment_request["prefix_pins"][str(100 * b)] = pure["sha"]([r for r in deployment_changed if r["bar_open_ms"] < 100 * b])
    deployment_plan = prepare(deployment_changed, deployment_request)
    check(deployment_plan["decision_features"] != baseline["decision_features"])
    check(deployment_plan["scaler_plan"] == baseline["scaler_plan"])
    check(deployment_plan["scaler_train_rows"] == baseline["scaler_train_rows"])
    for key, value in (("training_asof_ms", 100 * b), ("training_asof_ms", 101 * b),
                       ("decision_asof_ms", 99 * b), ("decision_asof_ms", 101 * b),
                       ("history_start_ms", b), ("ingestion_lag_ms", None),
                       ("universe", ["BTC/USDT"]), ("candidate_decisions_ms", [50 * b, 50 * b])):
        changed = copy.deepcopy(request)
        changed[key] = value
        rejected(lambda changed=changed: prepare(candles, changed))
    changed = copy.deepcopy(request)
    changed["coverage_end_ms"] = 119 * b  # Exact planned exit is outside half-open coverage.
    rejected(lambda: pure["origin_context"](changed, timing, api))
    empty_request = copy.deepcopy(request)
    empty_request["candidate_decisions_ms"] = [98 * b]
    bind_pins(candles, empty_request)
    empty = prepare(candles, empty_request)
    check(empty["scaler_plan"]["row_count"] == 0 and not empty["scaler_plan"]["fit_allowed"])
    for kind in ("missing", "duplicate", "visible_price", "late_old_row", "early_claim", "revision"):
        rows = copy.deepcopy(candles)
        if kind == "missing":
            rows.pop(10)
        elif kind == "duplicate":
            rows.append(copy.deepcopy(rows[10]))
        elif kind == "visible_price":
            rows[10]["close"] += 0.1
        elif kind == "late_old_row":
            rows[10]["available_at_ms"] = 101 * b
        elif kind == "early_claim":
            rows[10]["available_at_ms"] = rows[10]["bar_open_ms"]
        else:
            rows[10]["revision_id"] = "different"
        changed = copy.deepcopy(request)
        changed["expected_snapshot_sha256"] = pure["sha"](rows)
        rejected(lambda rows=rows, changed=changed: prepare(rows, changed))
    # A late exit row must remove the whole candidate, including both symbols.
    late = copy.deepcopy(candles)
    next(r for r in late if r["bar_open_ms"] == 91 * b and r["symbol"] == "SYN_A")["available_at_ms"] = 93 * b
    changed = copy.deepcopy(request)
    changed["expected_snapshot_sha256"] = pure["sha"](late)
    bind_pins(late, changed)
    late_plan = prepare(late, changed)
    check(late_plan["scaler_plan"]["row_count"] == 44)
    check(all(r["decision_asof_ms"] != 72 * b for r in late_plan["scaler_train_rows"]))
    # A separate origin changes its training bank; no scaler state is shared or fitted.
    next_candles, next_request = fixture(origin_bar=118)
    next_plan = prepare(next_candles, next_request)
    check(next_plan["scaler_plan"]["row_count"] == 82)
    check(next_plan["scaler_plan"]["row_ids_sha256"] != baseline["scaler_plan"]["row_ids_sha256"])
    flat_candles, flat_request = fixture(flat=True)
    flat = prepare(flat_candles, flat_request)
    check(all(not r["proposal_long"] for r in flat["decision_features"]))
    check(flat["scaler_plan"]["row_count"] == baseline["scaler_plan"]["row_count"])
    return examples, baseline["scaler_plan"]


def build():
    previous = quality["read"](PREVIOUS + "/completion_receipt.json")
    bank = dict(previous["inputs"])
    quality["unchanged"](bank)
    for name, expected in previous["output_identities"].items():
        path = PREVIOUS + "/" + name
        if old["identity"](path) != expected:
            raise ValueError("previous output identity changed")
        bank[path] = expected
    for path in (PREVIOUS + "/completion_receipt.json", PURE, SELF, DOC, KERNEL):
        bank[path] = old["identity"](path)
    examples, scaler = regressions()
    if any(name.split(".")[0] in old["BLOCKED"] for name in sys.modules):
        raise ValueError("forbidden module loaded")
    quality["unchanged"](bank)
    result = {"schema": "exia.genetic_primus.synthetic_origin_prefix_check/1", "inputs": bank,
              "direct_assertions": ASSERTIONS, "examples": examples, "baseline_scaler_plan": scaler,
              "parquet_reads": 0, "scaler_fit_count": 0, "model_fit_count": 0,
              "financial_evaluation_count": 0, "safety": previous["safety"],
              "synthetic_only": True, "historical_results_requalified": False,
              "market_research_launch_ready": False, "real_money_ready": False}
    body = api["canonical"](result) + b"\n"
    receipt = {"schema": "exia.genetic_primus.synthetic_origin_prefix_receipt/1", "inputs": bank,
               "output_identities": {"result.json": {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}},
               "safety": previous["safety"]}
    return result, receipt


def main():
    if sys.argv[1:] not in (["--run"], ["--verify"]):
        raise ValueError("use --run or --verify")
    result, receipt = build()
    if sys.argv[1:] == ["--run"]:
        old["local"](DEST + "/result.json").parent.mkdir(exist_ok=False)
        for name, value in (("result.json", result), ("completion_receipt.json", receipt)):
            old["emit"](DEST + "/" + name, value, api)
        print(f"COMPLETE: assertions={ASSERTIONS}; inputs={len(result['inputs'])}; parquet=0; scaler/model fit=0")
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
