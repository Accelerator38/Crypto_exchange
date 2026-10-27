"""Draft protocol + full8-shaped fabricated-row integration. No parquet/fit."""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
PURE = "src/exia/genetic_primus/offline_market_feature_adapter_v1.py"
SELF = "tools/check_genetic_primus_offline_market_feature_adapter_v1.py"
CONFIG = "configs/genetic_primus_offline_market_engineering_protocol_v1.json"
DOC = "docs/GENETIC_PRIMUS_OFFLINE_MARKET_FEATURE_ADAPTER_V1.md"
PREVIOUS = "Reports/Exia/Genetic_Primus/launch_preflight_v1_20261007"
DEST = "Reports/Exia/Genetic_Primus/offline_market_feature_adapter_v1_20261007"
MANIFEST = "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609/data_manifest_origin_09.json"
prior = runpy.run_path(str(ROOT / "tools/check_genetic_primus_launch_preflight_v1.py"))
old, api, quality = prior["old"], prior["api"], prior["quality"]
pure = runpy.run_path(str(ROOT / PURE))
kernel = runpy.run_path(str(ROOT / "src/exia/genetic_primus/asof_feature_builder_v1.py"))
ASSERTIONS = 0


def check(ok):
    global ASSERTIONS
    if not ok:
        raise AssertionError("market adapter regression")
    ASSERTIONS += 1


def rejected(call):
    try:
        call()
    except ValueError:
        check(True)
    else:
        raise AssertionError("bad market adapter input accepted")


def fake_bar(start, index, flat=False):
    # Real universe names, entirely fabricated prices. Not decoded market evidence.
    result = []
    for offset, symbol in enumerate(pure["SYMBOLS"]):
        value = 100.0 + offset + (0 if flat else index * 0.2 + (index % 3) * 0.01)
        result.append({"timestamp": start, "symbol": symbol, "open": value,
                       "high": value + 1, "low": value - 1, "close": value,
                       "volume": 1, "source_rows": 240})
    return result


def regressions(protocol, manifest, manifest_sha):
    gate = pure["protocol_gate"](protocol, manifest, manifest_sha)
    check(gate["metadata_binding_passed"] and not gate["fit_allowed"] and not gate["market_data_read_allowed"])
    check(len(gate["unresolved"]) == 14)
    b, history = pure["BAR_MS"], protocol["clock"]["history_start_ms"]
    make = lambda lag: pure["MarketFeatureState"](manifest["metadata"], manifest_sha, history, lag, "a" * 64)
    examples = []
    for lag in (0, 1, b + 1):
        state = make(lag)
        asof = history + 100 * b
        closes = {symbol: [] for symbol in pure["SYMBOLS"]}
        for i in range(100):
            rows = fake_bar(history + i * b, i)
            if history + (i + 1) * b + lag > asof:
                break
            check(state.feed_bar(rows, asof))
            for row in rows:
                closes[row["symbol"]].append(row["close"])
        snapshot = state.snapshot(asof)
        check(snapshot["history_bars"] == (100 if lag == 0 else 99 if lag == 1 else 98))
        for row in snapshot["rows"]:
            values = closes[row["symbol"]]
            e12, e48 = kernel["ema"](values, 12), kernel["ema"](values, 48)
            path = sum(abs(y - x) for x, y in zip(values[-13:-1], values[-12:]))
            check(abs(row["features"]["return_12"] - (values[-1] / values[-13] - 1)) < 1e-14)
            check(abs(row["features"]["efficiency_12"] - abs(values[-1] - values[-13]) / path) < 1e-14)
            proposal = e12[-1] / e48[-1] - 1 > 0 and e12[-1] / e12[-4] - 1 > 0 and row["features"]["return_12"] > 0 and row["features"]["efficiency_12"] >= 0.30
            check(row["proposal_long"] == proposal)
            check(row["feature_available_at_ms"] <= asof and row["proposal_dependencies"]["first_bar_ms"] == history)
        future = fake_bar(state.next_bar, state.count)
        for row in future:
            row["close"] = "UNPUBLISHED_SENTINEL"
        check(state.feed_bar(future, asof) is False)
        check(state.snapshot(asof) == snapshot)
        valid_future = fake_bar(state.next_bar, state.count)
        ready = state.next_bar + b + lag
        check(state.feed_bar(valid_future, ready))
        check(state.snapshot(ready)["history_bars"] == snapshot["history_bars"] + 1)
        rejected(lambda state=state, asof=asof: state.snapshot(asof))
        check(sum(len(v["rows"]) for v in state.states.values()) == 8 * 13)
        examples.append({"fabricated_full8_lag_ms": lag, "history_bars_at_decision": snapshot["history_bars"],
                         "snapshot_sha256": pure["sha"](snapshot), "raw_prices_persisted": False})
    state = make(0)
    check(state.feed_bar(fake_bar(history, 0), history + b))
    before = copy.deepcopy((state.states, state.count, state.next_bar, state.prefix_sha))
    broken = fake_bar(history + b, 1)
    broken[-1]["source_rows"] = 239
    rejected(lambda: state.feed_bar(broken, history + 2 * b))
    check((state.states, state.count, state.next_bar, state.prefix_sha) == before)
    rejected(lambda: state.snapshot(history + b))
    for kind in ("duplicate", "gap", "invalid_value"):
        rows = fake_bar(history + b, 1)
        if kind == "duplicate":
            rows[-1] = copy.deepcopy(rows[0])
        elif kind == "gap":
            for row in rows:
                row["timestamp"] += b
        else:
            rows[0]["close"] = None
        rejected(lambda rows=rows: state.feed_bar(rows, history + 3 * b))
    rejected(lambda: make(None))
    rejected(lambda: make(True))
    flat = make(0)
    for i in range(48):
        flat.feed_bar(fake_bar(history + i * b, i, flat=True), history + 48 * b)
    check(all(r["features"] == {"return_12": 0.0, "efficiency_12": 0.0} and not r["proposal_long"] for r in flat.snapshot(history + 48 * b)["rows"]))
    schedule = pure["candidate_calendar"](history + 100 * b, history + 99 * b, history, 0)
    check(schedule == [history + 64 * b, history + 82 * b])
    check(all((history + 100 * b - x) % (18 * b) == 0 for x in schedule))
    rejected(lambda: pure["candidate_calendar"](history + 100 * b, history + 100 * b, history, 0))
    changed = copy.deepcopy(protocol)
    changed["clock"]["candidate_calendar"] = "every_4h"
    rejected(lambda: pure["protocol_gate"](changed, manifest, manifest_sha))
    changed = copy.deepcopy(protocol)
    changed["authorization"]["fit_for_this_protocol"] = True
    rejected(lambda: pure["protocol_gate"](changed, manifest, manifest_sha))
    return gate, examples


def build():
    previous = quality["read"](PREVIOUS + "/completion_receipt.json")
    bank = dict(previous["inputs"])
    quality["unchanged"](bank)
    for name, expected in previous["output_identities"].items():
        path = PREVIOUS + "/" + name
        if old["identity"](path) != expected:
            raise ValueError("previous identity changed")
        bank[path] = expected
    for path in (PREVIOUS + "/completion_receipt.json", PURE, SELF, CONFIG, DOC, MANIFEST,
                 "src/exia/genetic_primus/features.py", "src/exia/genetic_primus/rolling_linear_retrain_v1.py"):
        bank[path] = old["identity"](path)
    protocol, manifest = quality["read"](CONFIG), quality["read"](MANIFEST)
    record = quality["read"]("Reports/Exia/Genetic_Primus/forward_preregistration_v1_20261007/record.json")["record"]
    for key in ("history_start_ms", "origin_start_ms", "training_asof_ms", "nominal_end_exclusive_ms"):
        old_key = {"history_start_ms": "planned_training_start_ms", "nominal_end_exclusive_ms": "nominal_deployment_end_exclusive_ms"}.get(key, key)
        check(protocol["clock"][key] == record["engineering_plan"][old_key])
    gate, examples = regressions(protocol, manifest, bank[MANIFEST]["sha256"])
    if any(name.split(".")[0] in old["BLOCKED"] for name in sys.modules):
        raise ValueError("forbidden dependency loaded")
    quality["unchanged"](bank)
    result = {"schema": "exia.genetic_primus.offline_market_adapter_check/1", "inputs": bank,
              "direct_assertions": ASSERTIONS, "protocol_sha256": pure["sha"](protocol), "protocol_gate": gate,
              "examples": examples, "fixture_prices_fabricated_not_market_evidence": True,
              "legacy_label_path_not_imported_or_requalified": True,
              "parquet_reads": 0, "scaler_fit_count": 0, "model_fit_count": 0,
              "market_financial_evaluation_count": 0, "market_research_launch_ready": False,
              "real_money_ready": False, "safety": protocol["safety"]}
    body = api["canonical"](result) + b"\n"
    receipt = {"schema": "exia.genetic_primus.offline_market_adapter_receipt/1", "inputs": bank,
               "output_identities": {"result.json": {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}},
               "safety": protocol["safety"]}
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
