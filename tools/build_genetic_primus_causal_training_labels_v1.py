"""Bounded artificial completed labels + label-free deploy integration; no fit."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/causal_training_labels_v1_20261007"
PREVIOUS = ROOT / "Reports/Exia/Genetic_Primus/asof_features_v1_20261007"
HELPER = ROOT / "src/exia/genetic_primus/causal_training_labels_v1.py"
FEATURES = ROOT / "src/exia/genetic_primus/asof_feature_builder_v1.py"
GUARD = ROOT / "src/exia/genetic_primus/causal_origin_payload_v1.py"
CAP = 1_048_576


def read_bounded(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("metadata path scope")
    before = path.stat()
    if before.st_size > CAP:
        raise ValueError("metadata file size cap")
    with path.open("rb") as stream:
        data = stream.read(CAP + 1)
    after = path.stat()
    if (len(data) != before.st_size or len(data) > CAP
            or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
        raise ValueError("metadata changed during read")
    return data


def fixture(api, features):
    dt = api["DT"]
    seed = 10_000 * dt  # Logical synthetic epoch, no historical market interval.
    universe = ["SYN_A", "SYN_B"]
    rows = []
    closes = {"SYN_A": 100.0, "SYN_B": 200.0}
    for index in range(148):
        for symbol in universe:
            opened = closes[symbol]
            step = (0.5 if not 68 <= index < 86 else -0.5) if symbol == "SYN_A" else (0.25 if not 104 <= index < 122 else -0.5)
            closed = opened if index == 0 else opened + step
            closes[symbol] = closed
            t = seed + index * dt
            rows.append({"symbol": symbol, "revision_id": "synthetic_revision_1", "bar_open_ms": t,
                         "bar_end_exclusive_ms": t + dt, "available_at_ms": t + dt + 1_000,
                         "open": opened, "high": max(opened, closed) + 0.125,
                         "low": min(opened, closed) - 0.125, "close": closed, "volume": 10.0 + index})
    deploy = seed + 140 * dt
    ctx = {"origin_id": "synthetic_origin_01", "deploy_start_ms": deploy,
           "deploy_end_exclusive_ms": deploy + api["HORIZON"], "training_asof_ms": seed + 134 * dt + 3_000,
           "decision_asof_ms": seed + 139 * dt + 2_500, "bar_ms": dt, "purge_bars": 7,
           "universe": universe}
    request = {"schema": "exia.genetic_primus.synthetic_causal_origin_request/1", "evidence_class": api["EVIDENCE"],
               "context": ctx, "history_start_ms": seed, "origin_starts_ms": [seed + i * dt for i in (50, 68, 86, 104)],
               "revision_id": "synthetic_revision_1", "historical_feature_offset_ms": 2_500,
               "kernel_definition_sha256": features["sha"](features["DEFINITION"]),
               "label_definition_sha256": api["sha"](api["LABEL_DEFINITION"]),
               "expected_training_source_sha256": api["sha"](rows[:244]),
               "expected_deploy_source_sha256": api["sha"](rows[:278])}
    return {"schema": "exia.genetic_primus.synthetic_origin_candles/1", "evidence_class": api["EVIDENCE"],
            "request": request, "candles": rows, "prices_are_artificial": True}


def prepare():
    sys.dont_write_bytecode = True
    api = runpy.run_path(str(HELPER))
    features, guard = runpy.run_path(str(FEATURES)), runpy.run_path(str(GUARD))
    bank = {}

    def capture(path):
        data = read_bounded(path)
        bank[path.relative_to(ROOT).as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    for path in (HELPER, FEATURES, GUARD, Path(__file__).resolve(), ROOT / "tools/check_genetic_primus_causal_training_labels_v1.py"):
        capture(path)
    previous = guard["strict_json"](capture(PREVIOUS / "completion_receipt.json").decode("utf-8"))
    if (set(previous["output_byte_sha256"]) != {"synthetic_candles.json", "kernel_definition.json", "feature_candidate.json", "pattern_results.json", "summary.json"}
            or previous["market_research_launch_ready"] is not False or previous["real_money_ready"] is not False
            or set(previous["safety"]) != set(guard["SAFETY_KEYS"])
            or any(previous["safety"][k] is not False for k in guard["SAFETY_KEYS"])):
        raise ValueError("predecessor scope/authority")
    for name, expected in previous["output_byte_sha256"].items():
        data = capture(PREVIOUS / name)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("predecessor output changed")
    for relative, expected in previous["inputs"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("input path escape")
        data = capture(ROOT / relative)
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("predecessor source/metadata changed")
    case = fixture(api, features)
    consumed, result = api["build_origin_inputs"](case["candles"], case["request"], features, guard)
    candidate = result["scaler_candidate_input"]
    summary = {"schema": "exia.genetic_primus.causal_training_label_summary/1", "evidence_class": api["EVIDENCE"],
               "source_candles": len(case["candles"]), "consumed_training_candles": len(consumed["training_source"]),
               "consumed_deploy_candles": len(consumed["deploy_source"]),
               "completed_training_origins": len(case["request"]["origin_starts_ms"]),
               "completed_training_rows": result["training_guard"]["completed_rows"],
               "negative_labels": result["training_guard"]["negative_outcomes"],
               "positive_labels": result["training_guard"]["positive_outcomes"],
               "proposal_opportunities": candidate["opportunities"],
               "negative_opportunities": candidate["negative_opportunities"],
               "positive_opportunities": candidate["positive_opportunities"],
               "shared_train_deploy_feature_definition": True, "gross_labels_not_execution_pnl": True,
               "minimum_opportunities_64_passed": result["minimum_opportunities_64_passed"],
               "ess_gate_verified": False, "actual_scaler_fitted": False, "model_ready": False,
               "market_data_read": False, "new_model_fits": 0, "new_market_strategy_evaluations": 0,
               "market_research_launch_ready": False, "real_money_ready": False,
               "safety": dict(result["safety"]),
               "next_action": "pre_fit_origin_admission_adapter_with_explicit_timing_and_sample_gates_no_training"}
    return api, features, guard, case, consumed, result, summary, bank


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
    api, _, _, case, consumed, result, summary, bank = prepare()
    verify_inputs(bank)
    if not args.verify_only:
        OUTPUT.mkdir(exist_ok=False)
        hashes = {}
        for name, value in (("synthetic_candles.json", case), ("consumed_snapshot.json", consumed),
                            ("origin_inputs.json", result), ("label_definition.json", api["LABEL_DEFINITION"]), ("summary.json", summary)):
            data = api["canonical_bytes"](value) + b"\n"
            with (OUTPUT / name).open("xb") as stream:
                stream.write(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        verify_inputs(bank)
        receipt = {"schema": "exia.genetic_primus.causal_training_label_receipt/1",
                   "receipt_kind": "synthetic_completed_labels_and_origin_inputs_not_fitting_or_market_evidence",
                   "inputs": bank, "inputs_verified_unchanged_before_and_after": True, "output_byte_sha256": hashes,
                   "origin_input_semantic_sha256": result["payload_sha256"],
                   "market_research_launch_ready": False, "real_money_ready": False, "safety": result["safety"]}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical_bytes"](receipt) + b"\n")
    print(f"synthetic origin: 8 labels, {summary['proposal_opportunities']} opportunities incl. {summary['negative_opportunities']} losses; shared kernel; fits=0, authority=false")


if __name__ == "__main__":
    main()
