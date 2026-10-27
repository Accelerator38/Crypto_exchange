"""Offline engineering feature fixtures only; no collector, model fit or trading."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/asof_features_v1_20261007"
PREVIOUS = ROOT / "Reports/Exia/Genetic_Primus/causal_origin_payload_v1_20261007"
HELPER = ROOT / "src/exia/genetic_primus/asof_feature_builder_v1.py"
GUARD = ROOT / "src/exia/genetic_primus/causal_origin_payload_v1.py"
CAP = 1_048_576


def read_bounded(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("metadata input path scope")
    before = path.stat()
    if before.st_size > CAP:
        raise ValueError("metadata input size cap")
    with path.open("rb") as stream:
        data = stream.read(CAP + 1)
    after = path.stat()
    if (len(data) != before.st_size or len(data) > CAP
            or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
        raise ValueError("metadata changed during read")
    return data


def fixture(api, name="mixed"):
    if name not in ("mixed", "rising", "falling", "flat", "alternating"):
        raise ValueError("unregistered artificial pattern")
    dt = api["DT"]
    start = 10_000 * dt  # Logical fixture epoch, not a real market date.
    end = start + 48 * dt
    universe = ["SYN_A", "SYN_B"]
    rows = []
    for i in range(56):
        for rank, symbol in enumerate(universe):
            pattern = ("rising" if rank == 0 else "falling") if name == "mixed" else name

            def close_at(index):
                if pattern == "rising":
                    return 100.0 + rank * 20 + index * 0.25
                if pattern == "falling":
                    return 100.0 + rank * 20 - index * 0.25
                if pattern == "alternating":
                    return 100.0 + rank * 20 + (index % 2)
                return 100.0 + rank * 20

            opened = close_at(max(0, i - 1))
            close = close_at(i)
            t = start + i * dt
            rows.append({"symbol": symbol, "revision_id": "synthetic_revision_1", "bar_open_ms": t,
                         "bar_end_exclusive_ms": t + dt, "available_at_ms": t + dt + 1_000,
                         "open": opened, "high": max(opened, close) + 0.125,
                         "low": min(opened, close) - 0.125, "close": close, "volume": 10.0 + i})
    prefix = rows[:96]
    request = {"schema": "exia.genetic_primus.synthetic_asof_request/1", "evidence_class": api["EVIDENCE"],
               "decision_asof_ms": end + 2_500, "history_start_ms": start, "universe": universe,
               "revision_id": "synthetic_revision_1", "expected_prefix_sha256": api["sha"](prefix),
               "kernel_definition_sha256": api["sha"](api["DEFINITION"])}
    return {"name": name, "request": request, "candles": rows,
            "description": "96 available prefix rows plus 16 later rows; entirely artificial"}


def prepare():
    sys.dont_write_bytecode = True
    api, guard = runpy.run_path(str(HELPER)), runpy.run_path(str(GUARD))
    bank = {}

    def capture(path):
        data = read_bounded(path)
        bank[path.relative_to(ROOT).as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    for path in (HELPER, GUARD, Path(__file__).resolve(), ROOT / "tools/check_genetic_primus_asof_features_v1.py"):
        capture(path)
    previous = guard["strict_json"](capture(PREVIOUS / "completion_receipt.json").decode("utf-8"))
    if (set(previous["output_byte_sha256"]) != {"synthetic_snapshot.json", "fixture_pins.json", "payloads.json", "guard_result.json", "summary.json"}
            or previous["market_research_launch_ready"] is not False or previous["real_money_ready"] is not False
            or set(previous["safety"]) != set(api["SAFETY_KEYS"])
            or any(previous["safety"][k] is not False for k in api["SAFETY_KEYS"])):
        raise ValueError("predecessor metadata/authority scope")
    for name, expected in previous["output_byte_sha256"].items():
        data = capture(PREVIOUS / name)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("predecessor artifact changed")
    for relative, expected in previous["inputs"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("input path escape")
        data = capture(ROOT / relative)
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("predecessor source/config/metadata changed")
    case = fixture(api)
    result = api["build_asof_features"](case["candles"], case["request"])
    # Row-schema compatibility only. No model identity or trained bundle admission.
    ctx = {"universe": case["request"]["universe"]}
    for row in result["rows"]:
        guard["keys"](row, guard["DEPLOY_ROW_KEYS"], "computed deploy row")
        guard["feature_row"](row, ctx, {"revision_id": case["request"]["revision_id"]}, case["request"]["decision_asof_ms"])
    pattern_results = {}
    for name in ("rising", "falling", "flat", "alternating"):
        registered = fixture(api, name)
        pattern_results[name] = api["build_asof_features"](registered["candles"], registered["request"])
    summary = {"schema": "exia.genetic_primus.asof_feature_engineering_summary/1",
               "evidence_class": api["EVIDENCE"], "primary_prefix_candles": 96, "primary_later_candles": 16,
               "symbols": 2, "history_bars_per_symbol": 48, "feature_rows": 2,
               "registered_primary_and_pattern_cases": 5, "prior_deploy_row_schema_compatible": True,
               "full_prior_model_bundle_compatible": False, "flat_path_policy": "explicit_zero_and_abstain_not_legacy_nan_drop",
               "real_indicator_equivalence_verified": False, "market_data_provenance_verified": False,
               "actual_compute_latency_verified": False, "market_data_read": False,
               "new_model_fits": 0, "new_market_strategy_evaluations": 0,
               "market_research_launch_ready": False, "real_money_ready": False,
               "safety": dict(previous["safety"]),
               "next_action": "causal_training_label_asof_builder_and_origin_metadata_integration_on_synthetic_data_before_fitting"}
    return api, guard, case, result, pattern_results, summary, bank


def verify_inputs(bank):
    for relative, expected in bank.items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("input path escape")
        data = read_bounded(ROOT / relative)
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("input changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    api, _, case, result, patterns, summary, bank = prepare()
    verify_inputs(bank)
    if not args.verify_only:
        OUTPUT.mkdir(exist_ok=False)
        hashes = {}
        for name, value in (("synthetic_candles.json", case), ("kernel_definition.json", api["DEFINITION"]),
                            ("feature_candidate.json", result), ("pattern_results.json", patterns), ("summary.json", summary)):
            data = api["canonical_bytes"](value) + b"\n"
            with (OUTPUT / name).open("xb") as stream:
                stream.write(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        verify_inputs(bank)
        receipt = {"schema": "exia.genetic_primus.asof_feature_engineering_receipt/1",
                   "receipt_kind": "bounded_synthetic_asof_features_not_training_or_market_evidence",
                   "inputs": bank, "inputs_verified_unchanged_before_and_after": True,
                   "output_byte_sha256": hashes, "prefix_sha256": result["prefix_sha256"],
                   "kernel_definition_sha256": result["kernel_definition_sha256"],
                   "market_research_launch_ready": False, "real_money_ready": False, "safety": summary["safety"]}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical_bytes"](receipt) + b"\n")
    print("asof fixtures: 96 retained + 16 ignored candles; 2 feature rows; 5 registered cases; fits=0, authority=false")


if __name__ == "__main__":
    main()
