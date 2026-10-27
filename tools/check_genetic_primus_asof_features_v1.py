"""Tiny as-of metamorphic checks and independent rational indicator oracle."""
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


def rational_oracle(closes):
    values = [Fraction(str(value)) for value in closes]

    def recurrence(span):
        weight = Fraction(2, span + 1)
        series = [values[0]]
        for value in values[1:]:
            series.append(series[-1] + weight * (value - series[-1]))
        return series

    recent = values[len(values) - 1 - 12:]
    path = sum(abs(recent[index] - recent[index - 1]) for index in range(1, len(recent)))
    ret = recent[-1] / recent[0] - 1
    efficiency = Fraction(0) if path == 0 else abs(recent[-1] - recent[0]) / path
    e12, e48 = recurrence(12), recurrence(48)
    gap, slope = e12[-1] / e48[-1] - 1, e12[-1] / e12[-4] - 1
    proposal = path != 0 and gap > 0 and slope > 0 and ret > 0 and efficiency >= Fraction(3, 10)
    return {"return_12": ret, "efficiency_12": efficiency, "ema_gap_12_48": gap,
            "ema_slope_12_3": slope, "proposal": bool(proposal), "flat": path == 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-only", action="store_true")
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_asof_features_v1.py"))
    api, guard, case, result, patterns, summary, bank = builder["prepare"]()
    checks, oracle_fields, max_error = 0, 0, 0.0

    def check(condition):
        nonlocal checks
        if not condition:
            raise AssertionError("asof regression")
        checks += 1

    def reject(call, fragment=None):
        try:
            call()
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            if fragment and fragment not in str(exc):
                raise AssertionError("wrong rejection: " + str(exc)) from exc
            check(True)
        else:
            raise AssertionError("invalid asof input accepted")

    if not args.artifacts_only:
        request, rows = case["request"], case["candles"]
        prefix = copy.deepcopy(rows[:96])
        check(api["build_asof_features"](prefix, request) == result)
        check(api["build_asof_features"](list(reversed(rows)), request) == result)
        modified = copy.deepcopy(rows)
        for row in modified[96:]:
            row.update({"close": 90_000.0, "open": 100_000.0, "high": 110_000.0,
                        "low": 80_000.0, "volume": 1e9})
        check(api["build_asof_features"](modified, request) == result)
        # Future OHLCV values are not read even if nonfinite. Their typed time
        # envelopes and the overall input cap remain mandatory for admission.
        modified[96]["close"] = float("nan")
        check(api["build_asof_features"](modified, request) == result)
        revision = copy.deepcopy(prefix[-1])
        revision.update({"revision_id": "synthetic_revision_2", "available_at_ms": request["decision_asof_ms"] + 1,
                         "close": 1e6, "high": 1e6 + 1})
        check(api["build_asof_features"](rows + [revision], request) == result)
        visible_revision = copy.deepcopy(revision)
        visible_revision["available_at_ms"] = request["decision_asof_ms"]
        reject(lambda: api["build_asof_features"](rows + [visible_revision], request), "visible revision")
        reject(lambda: api["build_asof_features"](rows[1:], request), "exact asof grid")
        reject(lambda: api["build_asof_features"](rows[:95] + rows[96:], request), "exact asof grid")
        reject(lambda: api["build_asof_features"](rows + [copy.deepcopy(rows[0])], request), "duplicate visible")
        delayed = copy.deepcopy(rows)
        delayed[95]["available_at_ms"] = request["decision_asof_ms"] + 1
        reject(lambda: api["build_asof_features"](delayed, request), "exact asof grid")
        changed = copy.deepcopy(rows)
        changed[95]["close"] -= 0.01
        reject(lambda: api["build_asof_features"](changed, request), "visible prefix changed")
        # A separately accepted new visible snapshot should change features;
        # prefix invariance must not be a hardcoded/static output artifact.
        different_prefix = copy.deepcopy(prefix)
        different_prefix[94].update({"open": 50.0, "high": 51.0, "low": 49.0, "close": 50.0})
        different_request = copy.deepcopy(request)
        different_request["expected_prefix_sha256"] = api["sha"](different_prefix)
        check(api["build_asof_features"](different_prefix, different_request)["feature_rows_sha256"] != result["feature_rows_sha256"])
        for name, value, fragment in (
            ("available_at_ms", rows[0]["bar_end_exclusive_ms"] - 1, "before candle closed"),
            ("bar_open_ms", 1, "candle envelope"),
            ("bar_end_exclusive_ms", rows[0]["bar_end_exclusive_ms"] + 1, "candle envelope"),
            ("available_at_ms", True, "integer UTC"),
            ("revision_id", "synthetic_revision_2", "visible revision"),
            ("symbol", "SYN_OTHER", "universe changed"),
            ("close", float("nan"), "finite OHLCV"),
            ("close", float("inf"), "finite OHLCV"),
            ("close", True, "finite OHLCV"),
            ("volume", -1.0, "OHLCV domain"),
            ("high", 0.0, "OHLCV domain"),
            ("low", rows[0]["high"] + 1, "OHLCV domain"),
        ):
            altered = copy.deepcopy(rows)
            altered[0][name] = value
            reject(lambda r=altered: api["build_asof_features"](r, request), fragment)
        altered = copy.deepcopy(rows)
        altered[0]["forward_gross_3d"] = 0.1
        reject(lambda: api["build_asof_features"](altered, request), "OHLCV row")
        for name, value, fragment in (
            ("evidence_class", "real_market", "only synthetic"),
            ("history_start_ms", 1, "aligned fixed history"),
            ("decision_asof_ms", True, "integer UTC"),
            ("history_start_ms", request["history_start_ms"] + api["DT"], "warmup"),
            ("expected_prefix_sha256", "0" * 64, "visible prefix changed"),
            ("expected_prefix_sha256", "G" * 64, "lowercase SHA"),
            ("kernel_definition_sha256", "0" * 64, "kernel definition changed"),
            ("universe", ["SYN_B", "SYN_A"], "sorted unique"),
            ("universe", ["SYN_A", "SYN_A"], "sorted unique"),
        ):
            altered = copy.deepcopy(request)
            altered[name] = value
            reject(lambda q=altered: api["build_asof_features"](rows, q), fragment)
        too_early = copy.deepcopy(request)
        too_early["decision_asof_ms"] = request["decision_asof_ms"] - 1_501
        reject(lambda: api["build_asof_features"](rows, too_early), "exact asof grid")
        reject(lambda: api["build_asof_features"](rows * 10, request), "input row budget")
        # Advancing asof may incorporate a newly available bar; keep the EMA
        # seed fixed, and pin the new visible snapshot separately.
        next_request = copy.deepcopy(request)
        next_request["decision_asof_ms"] += api["DT"]
        next_request["expected_prefix_sha256"] = api["sha"](rows[:98])
        next_result = api["build_asof_features"](rows, next_request)
        check(next_result["feature_end_exclusive_ms"] == result["feature_end_exclusive_ms"] + api["DT"])
        check(next_result["payload_sha256"] != result["payload_sha256"])
        check(result["payload_sha256"] == api["sha"]({k: v for k, v in result.items() if k != "payload_sha256"}))
        check(result["feature_rows_sha256"] == api["sha"](result["rows"]))
        check(all(value is False for value in result["safety"].values()))
        check(summary["new_model_fits"] == summary["new_market_strategy_evaluations"] == 0)
        check(result["availability_is_declared_not_measured"] is True)
        check(result["real_indicator_equivalence_verified"] is False and result["market_data_provenance_verified"] is False)
        check(summary["full_prior_model_bundle_compatible"] is False)
        for name in ("mixed", "rising", "falling", "flat", "alternating"):
            registered = builder["fixture"](api, name)
            actual = result if name == "mixed" else patterns[name]
            visible = registered["candles"][:96]
            for row, diag in zip(actual["rows"], actual["diagnostics"]):
                closes = [r["close"] for r in visible if r["symbol"] == row["symbol"]]
                expected = rational_oracle(closes)
                for field in ("return_12", "efficiency_12", "ema_gap_12_48", "ema_slope_12_3"):
                    value = row["features"][field] if field in row["features"] else diag[field]
                    error = abs(value - float(expected[field]))
                    max_error = max(max_error, error)
                    oracle_fields += 1
                    check(error <= 1e-12)
                check(row["proposal_long"] is expected["proposal"])
                check(diag["flat_path_abstention"] is expected["flat"])
                guard["keys"](row, guard["DEPLOY_ROW_KEYS"], "deploy row")
                guard["feature_row"](row, {"universe": registered["request"]["universe"]},
                                     {"revision_id": "synthetic_revision_1"}, registered["request"]["decision_asof_ms"])
                check(True)
            check(api["build_asof_features"](registered["candles"][:96], registered["request"]) == actual)
        check(patterns["rising"]["rows"][0]["proposal_long"] is True)
        check(patterns["falling"]["rows"][0]["proposal_long"] is False)
        check(all(r["features"] == {"return_12": 0.0, "efficiency_12": 0.0} and r["proposal_long"] is False for r in patterns["flat"]["rows"]))
        tree = ast.parse(builder["read_bounded"](builder["HELPER"]).decode("utf-8"))
        imports = {n.module if isinstance(n, ast.ImportFrom) else alias.name for n in ast.walk(tree)
                   if isinstance(n, (ast.Import, ast.ImportFrom)) for alias in n.names}
        check(imports <= {"__future__", "hashlib", "json", "math", "re"})
    else:
        output, read = builder["OUTPUT"], builder["read_bounded"]
        receipt = guard["strict_json"](read(output / "completion_receipt.json").decode("utf-8"))
        check(receipt["schema"] == "exia.genetic_primus.asof_feature_engineering_receipt/1")
        check(receipt["inputs"] == bank)
        check(receipt["inputs_verified_unchanged_before_and_after"] is True)
        check(receipt["prefix_sha256"] == result["prefix_sha256"])
        check(receipt["kernel_definition_sha256"] == result["kernel_definition_sha256"])
        check(receipt["safety"] == summary["safety"])
        check(receipt["market_research_launch_ready"] is False and receipt["real_money_ready"] is False)
        expected = {"synthetic_candles.json": case, "kernel_definition.json": api["DEFINITION"],
                    "feature_candidate.json": result, "pattern_results.json": patterns, "summary.json": summary}
        check(set(receipt["output_byte_sha256"]) == set(expected))
        for name, value in expected.items():
            data = read(output / name)
            check(hashlib.sha256(data).hexdigest() == receipt["output_byte_sha256"][name])
            check(guard["strict_json"](data.decode("utf-8")) == value)
        stored = guard["strict_json"](read(output / "synthetic_candles.json").decode("utf-8"))
        check(api["build_asof_features"](stored["candles"], stored["request"]) == result)
    builder["verify_inputs"](bank)
    check("numpy" not in sys.modules and "pandas" not in sys.modules and "websockets" not in sys.modules)
    check(not any(n == "exia" or n.startswith("exia.") for n in sys.modules))
    print(f"{checks} assertions passed; oracle_fields={oracle_fields}, max_error={max_error:.3g}; synthetic only, fits=0")


if __name__ == "__main__":
    main()
