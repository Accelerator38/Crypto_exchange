"""Bounded direct completed-label/origin integration regressions; no pytest."""
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
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_causal_training_labels_v1.py"))
    api, features, guard, case, consumed, result, summary, bank = builder["prepare"]()
    checks, label_fields, max_error = 0, 0, 0.0

    def check(condition):
        nonlocal checks
        if not condition:
            raise AssertionError("causal origin regression")
        checks += 1

    def reject(call, fragment=None):
        try:
            call()
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            if fragment and fragment not in str(exc):
                raise AssertionError("wrong rejection: " + str(exc)) from exc
            check(True)
        else:
            raise AssertionError("invalid origin accepted")

    def evaluate(rows, request):
        return api["build_origin_inputs"](rows, request, features, guard)

    if not args.artifacts_only:
        rows, request = case["candles"], case["request"]
        check(evaluate(rows, request) == (consumed, result))
        check(evaluate(list(reversed(rows)), request) == (consumed, result))
        check(evaluate(rows[:278], request) == (consumed, result))
        future = copy.deepcopy(rows)
        for row in future[278:]:
            row.update({"open": 100_000.0, "high": 110_000.0, "low": 90_000.0, "close": 105_000.0, "volume": 1e8})
        check(evaluate(future, request) == (consumed, result))
        future[278]["close"] = float("nan")
        check(evaluate(future, request) == (consumed, result))
        late_revision = copy.deepcopy(rows[0])
        late_revision.update({"revision_id": "synthetic_revision_2", "available_at_ms": request["context"]["decision_asof_ms"] + 1})
        check(evaluate(rows + [late_revision], request) == (consumed, result))
        late_revision["available_at_ms"] = result["train_cutoff_ms"]
        reject(lambda: evaluate(rows + [late_revision], request), "visible training revision")
        check(result["training"]["binding"]["feature_definition_sha256"] == result["deploy_features"]["kernel_definition_sha256"])
        check(all(f["kernel_definition_sha256"] == result["shared_feature_definition_sha256"] for f in result["historical_feature_lineage"]))
        check(result["training_guard"] == guard["validate_training"](result["training"], result["trusted_fixture_pins"]))
        check(result["train_cutoff_ms"] == request["context"]["training_asof_ms"] - 7 * api["DT"])
        check(result["train_cutoff_ms"] != request["context"]["deploy_start_ms"] - 7 * api["DT"])
        check(hashlib.sha256(api["canonical_bytes"](consumed) + b"\n").hexdigest() == result["training"]["binding"]["data_file_sha256"])
        check(api["sha"](consumed) == result["training"]["binding"]["data_content_sha256"])
        check(result["training_guard"]["completed_rows"] == 8 and len(result["deploy_features"]["rows"]) == 2)
        check(result["training_guard"]["negative_outcomes"] > 0 and result["training_guard"]["positive_outcomes"] > 0)
        candidate = result["scaler_candidate_input"]
        check(candidate["negative_opportunities"] > 0 and candidate["positive_opportunities"] > 0)
        selected = [r for r in result["training"]["rows"] if r["proposal_long"]]
        check(candidate["opportunities"] == len(selected))
        check(candidate["rows"] == [r["features"] for r in selected])
        check(candidate["rows_sha256"] == api["sha"](candidate["rows"]))
        check(candidate["rows_sha256"] != result["training_guard"]["scaler_fit_rows_sha256"])
        check(result["training_guard_scaler_hash_scope"] == "all_training_rows_not_opportunity_filtered")
        check(result["full_old_model_bundle_compatible"] is False)
        check(result["minimum_opportunities_64_passed"] is False and result["ess_gate_verified"] is False)
        check(all(result[k] is False for k in ("model_ready", "entry_accounting_integrated", "real_execution_labels_verified", "market_data_provenance_verified", "actual_compute_latency_verified", "market_research_launch_ready", "real_money_ready")))
        check(candidate["actual_scaler_fitted"] is False and result["new_model_fits"] == result["new_market_strategy_evaluations"] == 0)
        check(all(value is False for value in result["safety"].values()))
        check(result["payload_sha256"] == api["sha"]({k: v for k, v in result.items() if k != "payload_sha256"}))
        # Independent price-ratio oracle, plus full-path availability checks.
        by_identity = {(r["bar_open_ms"], r["symbol"]): r for r in rows}
        for row in result["training"]["rows"]:
            start, end, symbol = row["label_start_ms"], row["label_end_exclusive_ms"], row["symbol"]
            first = by_identity[(start, symbol)]
            last = by_identity[(end - api["DT"], symbol)]
            expected = Fraction(str(last["close"])) / Fraction(str(first["open"])) - 1
            error = abs(row["gross_return_3d"] - float(expected))
            max_error = max(max_error, error)
            label_fields += 1
            check(error <= 1e-12)
            check(row["label_available_at_ms"] == max(by_identity[(t, symbol)]["available_at_ms"] for t in range(start, end, api["DT"])))
            check(end <= row["label_available_at_ms"] <= result["train_cutoff_ms"])
            check(row["feature_available_at_ms"] == start - api["DT"] + request["historical_feature_offset_ms"])
            check(row["feature_bar_end_exclusive_ms"] < start)
        for row in result["deploy_features"]["rows"]:
            check(set(row) == set(guard["DEPLOY_ROW_KEYS"]))
            check("gross_return_3d" not in row and "label_available_at_ms" not in row)
        reject(lambda: evaluate(rows[1:], request), "missing/unavailable training")
        reject(lambda: evaluate(rows + [copy.deepcopy(rows[0])], request), "duplicate visible training")
        delayed = copy.deepcopy(rows)
        delayed[100]["available_at_ms"] = result["train_cutoff_ms"] + 1
        reject(lambda: evaluate(delayed, request), "missing/unavailable training")
        changed = copy.deepcopy(rows)
        changed[0]["revision_id"] = "synthetic_revision_2"
        reject(lambda: evaluate(changed, request), "visible training revision")
        changed = copy.deepcopy(rows)
        changed[243]["close"] -= 0.01
        reject(lambda: evaluate(changed, request), "training source changed")
        # Historical features cannot be reconstructed using a candle that was
        # only available later, even if the raw training archive knows it now.
        delayed_history = copy.deepcopy(rows)
        delayed_history[96]["available_at_ms"] = request["origin_starts_ms"][0] - api["DT"] + 2_501
        revised_request = copy.deepcopy(request)
        revised_request["expected_training_source_sha256"] = api["sha"](delayed_history[:244])
        revised_request["expected_deploy_source_sha256"] = api["sha"](delayed_history[:278])
        reject(lambda: evaluate(delayed_history, revised_request), "missing/unavailable candle in exact asof grid")
        # Future relative to the first historical feature deadline, but completed
        # and known by training cutoff: affects its label, not its prior features.
        altered = copy.deepcopy(rows)
        altered[134].update({"close": rows[134]["close"] + 5.0, "high": rows[134]["high"] + 6.0})
        different_request = copy.deepcopy(request)
        different_request["expected_training_source_sha256"] = api["sha"](altered[:244])
        different_request["expected_deploy_source_sha256"] = api["sha"](altered[:278])
        _, different = evaluate(altered, different_request)
        check(different["training"]["rows"][0]["features"] == result["training"]["rows"][0]["features"])
        check(different["training"]["rows"][0]["gross_return_3d"] != result["training"]["rows"][0]["gross_return_3d"])
        future_origins = copy.deepcopy(request)
        future_origins["origin_starts_ms"].append(future_origins["origin_starts_ms"][-1] + api["HORIZON"])
        reject(lambda: evaluate(rows, future_origins), "future label requested")
        for key, value, fragment in (
            ("origin_starts_ms", request["origin_starts_ms"][:1] + request["origin_starts_ms"][2:], "contiguous"),
            ("origin_starts_ms", [True], "integer UTC"),
            ("historical_feature_offset_ms", True, "typed pinned"),
            ("historical_feature_offset_ms", 2_501, "deadline mismatch"),
            ("label_definition_sha256", "0" * 64, "label definition changed"),
            ("kernel_definition_sha256", "0" * 64, "one feature kernel"),
            ("expected_training_source_sha256", "0" * 64, "training source changed"),
            ("expected_deploy_source_sha256", "0" * 64, "visible prefix changed"),
            ("evidence_class", "market", "only synthetic"),
        ):
            altered_request = copy.deepcopy(request)
            altered_request[key] = value
            reject(lambda q=altered_request: evaluate(rows, q), fragment)
        wrong_cutoff = copy.deepcopy(result["training"])
        wrong_cutoff["train_cutoff_ms"] = request["context"]["deploy_start_ms"] - 7 * api["DT"]
        wrong_cutoff.pop("payload_sha256")
        wrong_cutoff["payload_sha256"] = api["sha"](wrong_cutoff)
        reject(lambda: guard["validate_training"](wrong_cutoff, result["trusted_fixture_pins"]), "training_asof")
        reject(lambda: evaluate(rows * 4, request), "source candle budget")
        tree = ast.parse(builder["read_bounded"](builder["HELPER"]).decode("utf-8"))
        imports = {node.module if isinstance(node, ast.ImportFrom) else alias.name for node in ast.walk(tree)
                   if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
        check(imports <= {"__future__", "hashlib", "json"})
    else:
        output, read = builder["OUTPUT"], builder["read_bounded"]
        receipt = guard["strict_json"](read(output / "completion_receipt.json").decode("utf-8"))
        check(receipt["schema"] == "exia.genetic_primus.causal_training_label_receipt/1")
        check(receipt["inputs"] == bank)
        check(receipt["inputs_verified_unchanged_before_and_after"] is True)
        check(receipt["origin_input_semantic_sha256"] == result["payload_sha256"])
        check(receipt["safety"] == result["safety"])
        check(receipt["market_research_launch_ready"] is False and receipt["real_money_ready"] is False)
        expected = {"synthetic_candles.json": case, "consumed_snapshot.json": consumed, "origin_inputs.json": result,
                    "label_definition.json": api["LABEL_DEFINITION"], "summary.json": summary}
        check(set(receipt["output_byte_sha256"]) == set(expected))
        for name, value in expected.items():
            data = read(output / name)
            check(hashlib.sha256(data).hexdigest() == receipt["output_byte_sha256"][name])
            check(guard["strict_json"](data.decode("utf-8")) == value)
        check(hashlib.sha256(read(output / "consumed_snapshot.json")).hexdigest() == result["training"]["binding"]["data_file_sha256"])
        stored = guard["strict_json"](read(output / "synthetic_candles.json").decode("utf-8"))
        check(evaluate(stored["candles"], stored["request"]) == (consumed, result))
    builder["verify_inputs"](bank)
    check("numpy" not in sys.modules and "pandas" not in sys.modules and "websockets" not in sys.modules)
    check(not any(name == "exia" or name.startswith("exia.") for name in sys.modules))
    print(f"{checks} assertions passed; label_oracle={label_fields}, max_error={max_error:.3g}; synthetic only, fits=0")


if __name__ == "__main__":
    main()
