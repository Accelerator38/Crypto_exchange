"""Direct tiny stdlib regressions for causal metadata isolation, never pytest."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-only", action="store_true")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_causal_origin_payload_v1.py"))
    api, snapshot, pins, bundle, result, summary, bank = builder["prepare"]()
    checks = 0

    def check(condition):
        nonlocal checks
        if not condition:
            raise AssertionError("metadata regression")
        checks += 1

    def reject(call, fragment=None):
        try:
            call()
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            if fragment and fragment not in str(exc):
                raise AssertionError("wrong rejection: " + str(exc)) from exc
            check(True)
        else:
            raise AssertionError("invalid payload accepted")

    def refresh(part, value):
        value = copy.deepcopy(value)
        field = "model_sha256" if part == "model" else "payload_sha256"
        value.pop(field, None)
        if "rows" in value:
            value["rows_sha256"] = api["sha"](value["rows"])
        return builder["sealed"](api, value, field)

    def bad(part, mutate, fragment=None, reseal=True):
        changed = copy.deepcopy(bundle)
        mutate(changed[part])
        if reseal:
            changed[part] = refresh(part, changed[part])
            if part == "model":
                changed["deploy"]["decision_model_sha256"] = changed["model"]["model_sha256"]
                changed["deploy"] = refresh("deploy", changed["deploy"])
        reject(lambda: api["validate_bundle"](changed, pins), fragment)

    if not args.artifacts_only:
        check(result == api["validate_bundle"](bundle, pins))
        check(api["validate_deploy"](bundle["deploy"], pins) == result["deploy"])
        check(result["training"]["completed_rows"] == 8)
        check(result["training"]["negative_outcomes"] == result["training"]["positive_outcomes"] == 4)
        check(result["training"]["train_cutoff_ms"] == pins["context"]["training_asof_ms"] - 7 * api["DT"])
        check(bundle["deploy"]["decision_model_sha256"] == result["model"]["model_sha256"])
        check(summary["new_model_fits"] == summary["new_market_strategy_evaluations"] == 0)
        check(all(value is False for value in result["safety"].values()))
        check(all(result[k] is False for k in (
            "actual_feature_causality_verified", "actual_scaler_fitting_verified", "actual_training_latency_verified",
            "actual_revision_provenance_verified", "actual_grid_verified", "entry_model_sha_verified",
            "accounting_integration_verified", "market_research_launch_ready", "real_money_ready")))
        check(hashlib.sha256(api["canonical_bytes"](snapshot) + b"\n").hexdigest() == pins["binding"]["data_file_sha256"])
        check(api["sha"](snapshot) == pins["binding"]["data_content_sha256"])
        # Label-free constructor needs no labels, future candles or market arrays.
        constructed = builder["make_deploy"](api, pins, snapshot["deploy_feature_rows"], bundle["model"]["model_sha256"])
        check(constructed == bundle["deploy"])
        future_bank = {"gross_return_3d": 1e6, "label_available_at_ms": pins["context"]["deploy_end_exclusive_ms"]}
        future_bank["gross_return_3d"] = -0.99
        check(builder["make_deploy"](api, pins, snapshot["deploy_feature_rows"], bundle["model"]["model_sha256"]) == constructed)
        for column in ("gross_return_3d", "forward_gross_3d", "label_start_ms", "label_end_exclusive_ms", "label_available_at_ms", "terminal_close"):
            bad("deploy", lambda p, c=column: p["rows"][0].update({c: 1}), "deploy row")
        bad("deploy", lambda p: p.update({"labels": []}), "deploy features only")
        bad("deploy", lambda p: p["rows"][0]["features"].update({"future_return": 0.1}), "two deploy features")
        bad("deploy", lambda p: p["rows"][0].update({"feature_available_at_ms": pins["context"]["decision_asof_ms"] + 1}), "feature availability")
        bad("deploy", lambda p: p["rows"][0].update({"feature_bar_end_exclusive_ms": pins["context"]["deploy_start_ms"]}), "feature availability")
        bad("deploy", lambda p: p["rows"][0].update({"feature_bar_open_ms": 1}), "feature availability")
        bad("deploy", lambda p: p["rows"][0].update({"revision_id": "synthetic_revision_2"}), "row revision")
        bad("deploy", lambda p: p["rows"].pop(), "plain bounded list")
        bad("deploy", lambda p: p["rows"].reverse(), "deploy symbol")
        bad("deploy", lambda p: p["rows"][0].update({"proposal_long": 1}), "typed proposal")
        bad("deploy", lambda p: p["rows"][0]["features"].update({"efficiency_12": 2.0}), "indicator domain")
        bad("deploy", lambda p: p["rows"][0]["features"].update({"return_12": True}), "numeric scalar")
        bad("deploy", lambda p: p["rows"][0]["features"].update({"return_12": 0.03}), "deploy row hash")
        bad("deploy", lambda p: p.update({"decision_model_sha256": "0" * 64}), "active decision model")
        bad("deploy", lambda p: p.update({"decision_model_sha256": "A" * 64}), "lowercase SHA")
        bad("deploy", lambda p: p.update({"evidence_class": "market_verified"}), "only synthetic")
        bad("training", lambda p: p.update({"train_cutoff_ms": pins["context"]["deploy_start_ms"] - 7 * api["DT"]}), "training_asof")
        bad("training", lambda p: p["rows"][0].update({"label_available_at_ms": p["train_cutoff_ms"] + 1}), "training label")
        bad("training", lambda p: p["rows"][0].update({"label_available_at_ms": p["rows"][0]["label_end_exclusive_ms"] - 1}), "training label")
        bad("training", lambda p: p["rows"][0].update({"label_end_exclusive_ms": p["rows"][0]["label_end_exclusive_ms"] + 1}), "training label")
        bad("training", lambda p: p["rows"][0].update({"feature_available_at_ms": p["rows"][0]["label_start_ms"] + 1}), "feature availability")
        bad("training", lambda p: p["rows"][0].update({"gross_return_3d": -1.0}), "nonpositive")
        bad("training", lambda p: p["rows"][0].update({"gross_return_3d": 0.5}), "training rows hash")
        bad("training", lambda p: p["rows"].reverse(), "unordered training")
        bad("training", lambda p: p["rows"].append(copy.deepcopy(p["rows"][0])), "training rows")
        bad("training", lambda p: p["rows"].pop(0), "label group")
        bad("training", lambda p: p.update({"rows": p["rows"][:2] + p["rows"][4:]}), "missing or overlapping")
        bad("training", lambda p: p.update({"rows": p["rows"] * 65}), "size cap")
        bad("training", lambda p: p.update({"train_cutoff_ms": True}), "integer UTC")
        bad("training", lambda p: p.update({"payload_sha256": "0" * 64}), "mutated payload", reseal=False)
        for part in ("training", "deploy", "model"):
            bad(part, lambda p: p["context"].update({"origin_id": "synthetic_origin_02"}), "identity changed")
            bad(part, lambda p: p["binding"].update({"revision_id": "synthetic_revision_2"}), "identity changed")
        bad("model", lambda p: p.update({"fit_mode": "trained"}), "not an actual trained")
        bad("model", lambda p: p.update({"fit_started_at_ms": p["fit_completed_at_ms"] + 1}), "causally available")
        bad("model", lambda p: p.update({"model_available_at_ms": pins["context"]["decision_asof_ms"] + 1}), "causally available")
        bad("model", lambda p: p.update({"training_asof_ms": p["training_asof_ms"] - 1}), "causally available")
        bad("model", lambda p: p.update({"model_available_at_ms": False}), "integer UTC")
        bad("model", lambda p: p.update({"scaler_fit_rows_sha256": "0" * 64}), "lineage mismatch")
        bad("model", lambda p: p.update({"train_rows_sha256": "0" * 64}), "lineage mismatch")
        bad("model", lambda p: p.update({"train_payload_sha256": "0" * 64}), "lineage mismatch")
        bad("model", lambda p: p["scaler"].update({"fit_rows_sha256": "0" * 64}), "scaler input lineage")
        bad("model", lambda p: p["scaler"].update({"iqr": [0, 1]}), "denominator")
        bad("model", lambda p: p["scaler"].update({"median": [0]}), "plain bounded list")
        bad("model", lambda p: p["scaler"].update({"median": [0.2, 0.3]}), "scaler payload hash")
        bad("model", lambda p: p.update({"coefficients": [0, 1]}), "plain bounded list")
        bad("model", lambda p: p.update({"coefficients": [True, 1, 1]}), "numeric scalar")
        bad("model", lambda p: p.update({"features": list(reversed(api["FEATURES"]))}), "feature order")
        bad("model", lambda p: p.update({"model_sha256": "0" * 64}), "mutated payload", reseal=False)
        bad("model", lambda p: p.update({"promotion_authority": True}), "manual model fixture")
        reject(lambda: api["validate_bundle"]({**bundle, "future_labels": []}, pins), "isolated origin bundle")
        reject(lambda: api["strict_json"]('{"a":1,"a":2}'), "duplicate")
        reject(lambda: api["strict_json"]('{"a":NaN}'), "nonfinite")
        reject(lambda: api["strict_json"]('{"a":Infinity}'), "nonfinite")
        reject(lambda: api["sha"]({"a": float("nan")}))
        check(api["sha"]({"a": 1, "b": 2}) == api["sha"]({"b": 2, "a": 1}))
        tree = ast.parse(builder["read_bounded"](builder["HELPER"]).decode("utf-8"))
        imports = {node.module if isinstance(node, ast.ImportFrom) else alias.name
                   for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))
                   for alias in node.names}
        check(imports <= {"__future__", "hashlib", "json", "math", "re"})
    else:
        output, read = builder["OUTPUT"], builder["read_bounded"]
        receipt = api["strict_json"](read(output / "completion_receipt.json").decode("utf-8"))
        check(receipt["schema"] == "exia.genetic_primus.causal_origin_payload_receipt/1")
        check(receipt["inputs"] == bank)
        check(receipt["inputs_verified_unchanged_before_and_after"] is True)
        check(receipt["proposal_semantic_sha256"] == builder["PROPOSAL_SHA"])
        check(receipt["market_research_launch_ready"] is False and receipt["real_money_ready"] is False)
        check(receipt["safety"] == summary["safety"])
        expected = {"synthetic_snapshot.json": snapshot, "fixture_pins.json": pins, "payloads.json": bundle,
                    "guard_result.json": result, "summary.json": summary}
        check(set(receipt["output_byte_sha256"]) == set(expected))
        for name, value in expected.items():
            data = read(output / name)
            check(hashlib.sha256(data).hexdigest() == receipt["output_byte_sha256"][name])
            check(api["strict_json"](data.decode("utf-8")) == value)
        snapshot_bytes = read(output / "synthetic_snapshot.json")
        check(hashlib.sha256(snapshot_bytes).hexdigest() == pins["binding"]["data_file_sha256"])
        check(api["validate_bundle"](bundle, pins) == result)
    builder["verify_inputs"](bank)
    check("numpy" not in sys.modules and "pandas" not in sys.modules and "websockets" not in sys.modules)
    check(not any(name == "exia" or name.startswith("exia.") for name in sys.modules))
    print(f"{checks} metadata assertions passed; manual fixtures only; fits=0; network/runtime=off")


if __name__ == "__main__":
    main()
