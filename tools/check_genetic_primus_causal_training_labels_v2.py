"""Train identity must not depend on deployment prices or their file hash."""
from __future__ import annotations

import argparse
import ast
import copy
from fractions import Fraction
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
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_causal_training_labels_v2.py"))
    api, labels, features, guard, old, case, train, deploy, result, summary, bank = builder["prepare"]()
    checks, oracle_fields, max_error = 0, 0, 0.0

    def check(condition):
        nonlocal checks
        if not condition:
            raise AssertionError("v2 independent source regression")
        checks += 1

    def reject(call, fragment=None):
        try:
            call()
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            if fragment and fragment not in str(exc):
                raise AssertionError("wrong rejection: " + str(exc)) from exc
            check(True)
        else:
            raise AssertionError("invalid input accepted")

    def evaluate(rows, request):
        return api["build_origin_inputs"](rows, request, labels, features, guard)

    def training_only(rows, request):
        return api["build_training_inputs"](rows, request, labels, features, guard)

    if not args.artifacts_only:
        rows, request = case["candles"], case["request"]
        training_request = {k: v for k, v in request.items() if k != "expected_deploy_source_sha256"}
        training_request["schema"] = "exia.genetic_primus.synthetic_training_request/2"
        info = result["training_inputs"]
        check(training_only(rows[:244], training_request) == (train, info))
        check(training_only(rows, training_request) == (train, info))
        check(evaluate(rows[:278], request) == (train, deploy, result))
        check(evaluate(list(reversed(rows)), request) == (train, deploy, result))
        # Visible at deployment, but not at training-asof. Training identity
        # remains stable; only the deploy source and feature candidate change.
        changed = copy.deepcopy(rows)
        changed[272].update({"close": rows[272]["close"] + 5, "high": rows[272]["high"] + 6})
        changed_request = copy.deepcopy(request)
        changed_request["expected_deploy_source_sha256"] = labels["sha"](changed[:278])
        new_train, new_deploy, different = evaluate(changed, changed_request)
        check(new_train == train)
        check(different["training_inputs"] == info)
        check(different["training_inputs"]["training"]["payload_sha256"] == info["training"]["payload_sha256"])
        check(new_deploy != deploy)
        check(different["deploy_features"]["feature_rows_sha256"] != result["deploy_features"]["feature_rows_sha256"])
        check(training_only(changed, training_request) == (train, info))
        # Preserve the v1 defect as a negative control, not a corrected-history claim.
        _, old_result = labels["build_origin_inputs"](rows, request, features, guard)
        _, old_changed = labels["build_origin_inputs"](changed, changed_request, features, guard)
        check(old_changed["training"]["payload_sha256"] != old_result["training"]["payload_sha256"])
        no_deploy = rows[:272] + rows[273:]
        check(training_only(no_deploy, training_request) == (train, info))
        reject(lambda: evaluate(no_deploy, request), "exact asof grid")
        invalid_future = copy.deepcopy(rows)
        invalid_future[280]["close"] = float("nan")
        check(evaluate(invalid_future, request) == (train, deploy, result))
        contaminated = dict(training_request)
        contaminated["expected_deploy_source_sha256"] = request["expected_deploy_source_sha256"]
        reject(lambda: training_only(rows, contaminated), "training-only request")
        wanted_future = copy.deepcopy(training_request)
        wanted_future["origin_starts_ms"].append(wanted_future["origin_starts_ms"][-1] + labels["HORIZON"])
        reject(lambda: training_only(rows, wanted_future), "future label requested")
        reject(lambda: training_only(rows[1:], training_request), "missing/unavailable training")
        delayed = copy.deepcopy(rows)
        delayed[100]["available_at_ms"] = info["training"]["train_cutoff_ms"] + 1
        reject(lambda: training_only(delayed, training_request), "missing/unavailable training")
        revised = copy.deepcopy(rows)
        revised[0]["revision_id"] = "synthetic_revision_2"
        reject(lambda: training_only(revised, training_request), "visible training revision")
        check(guard["validate_training"](info["training"], info["training_pins"]) == info["training_guard"])
        check(info["training"]["binding"]["data_content_sha256"] == api["sha"](train))
        check(info["training"]["binding"]["data_file_sha256"] == hashlib.sha256(api["canonical_bytes"](train) + b"\n").hexdigest())
        check(result["deploy_source_sha256"] == api["sha"](deploy))
        check(result["deploy_source_file_sha256"] == hashlib.sha256(api["canonical_bytes"](deploy) + b"\n").hexdigest())
        check(info["training"]["binding"]["data_content_sha256"] != result["deploy_source_sha256"])
        check(info["training"]["binding"]["feature_definition_sha256"] == result["deploy_features"]["kernel_definition_sha256"])
        check(info["scaler_candidate_input"]["opportunities"] == 7)
        check(info["scaler_candidate_input"]["negative_opportunities"] == 2)
        check(info["scaler_candidate_input"]["rows_sha256"] != info["training_guard"]["scaler_fit_rows_sha256"])
        check(info["training_guard"]["completed_rows"] == 8)
        by_key = {(r["bar_open_ms"], r["symbol"]): r for r in train["source"]}
        for row in info["training"]["rows"]:
            start, end, symbol = row["label_start_ms"], row["label_end_exclusive_ms"], row["symbol"]
            expected = Fraction(str(by_key[(end - labels["DT"], symbol)]["close"])) / Fraction(str(by_key[(start, symbol)]["open"])) - 1
            error = abs(row["gross_return_3d"] - float(expected))
            oracle_fields += 1
            max_error = max(max_error, error)
            check(error <= 1e-12)
            check(end <= row["label_available_at_ms"] <= info["training"]["train_cutoff_ms"])
            check(row["feature_available_at_ms"] < start)
        check(all(value is False for value in result["safety"].values()))
        check(result["new_model_fits"] == result["new_market_strategy_evaluations"] == 0)
        check(result["minimum_opportunities_64_passed"] is False and result["ess_gate_verified"] is False)
        check(all(result[k] is False for k in ("model_ready", "entry_accounting_integrated", "real_execution_labels_verified", "market_data_provenance_verified", "actual_compute_latency_verified", "market_research_launch_ready", "real_money_ready", "full_old_model_bundle_compatible")))
        check(info["scaler_candidate_input"]["actual_scaler_fitted"] is False)
        check(result["payload_sha256"] == api["sha"]({k: v for k, v in result.items() if k != "payload_sha256"}))
        tree = ast.parse(old["read_bounded"](builder["HELPER"]).decode("utf-8"))
        imports = {n.module if isinstance(n, ast.ImportFrom) else alias.name for n in ast.walk(tree)
                   if isinstance(n, (ast.Import, ast.ImportFrom)) for alias in n.names}
        check(imports <= {"__future__", "hashlib", "json"})
    else:
        output, read = builder["OUTPUT"], old["read_bounded"]
        receipt = guard["strict_json"](read(output / "completion_receipt.json").decode("utf-8"))
        check(receipt["schema"] == "exia.genetic_primus.causal_training_label_receipt/2")
        check(receipt["inputs"] == bank)
        check(receipt["inputs_verified_unchanged_before_and_after"] is True)
        check(receipt["origin_input_semantic_sha256"] == result["payload_sha256"])
        check(receipt["safety"] == result["safety"])
        check(receipt["market_research_launch_ready"] is False and receipt["real_money_ready"] is False)
        expected = {"synthetic_candles.json": case, "training_snapshot.json": train, "deploy_snapshot.json": deploy,
                    "origin_inputs.json": result, "summary.json": summary}
        check(set(receipt["output_byte_sha256"]) == set(expected))
        for name, value in expected.items():
            data = read(output / name)
            check(hashlib.sha256(data).hexdigest() == receipt["output_byte_sha256"][name])
            check(guard["strict_json"](data.decode("utf-8")) == value)
        check(hashlib.sha256(read(output / "training_snapshot.json")).hexdigest() == result["training_inputs"]["training"]["binding"]["data_file_sha256"])
        check(hashlib.sha256(read(output / "deploy_snapshot.json")).hexdigest() == result["deploy_source_file_sha256"])
    old["verify_inputs"](bank)
    check("numpy" not in sys.modules and "pandas" not in sys.modules and "websockets" not in sys.modules)
    check(not any(name == "exia" or name.startswith("exia.") for name in sys.modules))
    print(f"{checks} v2 assertions passed; label_oracle={oracle_fields}, max_error={max_error:.3g}; independent training identity, fits=0")


if __name__ == "__main__":
    main()
