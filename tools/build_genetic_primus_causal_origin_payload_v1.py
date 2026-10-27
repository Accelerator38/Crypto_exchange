"""Publish eight artificial training rows and two label-free deploy rows, no fit."""
from __future__ import annotations

import argparse
import copy
import hashlib
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/causal_origin_payload_v1_20261007"
PREVIOUS = ROOT / "Reports/Exia/Genetic_Primus/forward_research_audit_v1_20261006"
HELPER = ROOT / "src/exia/genetic_primus/causal_origin_payload_v1.py"
PROPOSAL_SHA = "a5abd987351f2ef60ad359fd396f6c2879e4c45ee166e08fde7ec2807b94e637"
CAP = 1_048_576


def read_bounded(path):
    path = path.resolve(strict=True)
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("metadata input outside repository")
    before = path.stat()
    if before.st_size > CAP:
        raise ValueError("metadata size limit")
    with path.open("rb") as stream:
        data = stream.read(CAP + 1)
    after = path.stat()
    if (len(data) != before.st_size or len(data) > CAP
            or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
        raise ValueError("metadata changed during read")
    return data


def sealed(api, value, field="payload_sha256"):
    result = copy.deepcopy(value)
    result[field] = api["sha"](result)
    return result


def make_deploy(api, pins, rows, model_sha):
    """No training/labels/terminal prices argument by construction."""
    return sealed(api, {
        "schema": "exia.genetic_primus.deploy_features_payload/1",
        "evidence_class": api["EVIDENCE"], "context": copy.deepcopy(pins["context"]),
        "binding": copy.deepcopy(pins["binding"]), "rows": copy.deepcopy(rows),
        "rows_sha256": api["sha"](rows), "decision_model_sha256": model_sha,
    })


def fixture(api):
    dt, horizon = api["DT"], api["HORIZON"]
    deploy = 5_000 * dt  # Logical synthetic epoch, not a market interval.
    universe = ["SYN_A", "SYN_B"]
    revision = "synthetic_revision_1"
    ctx = {"origin_id": "synthetic_origin_01", "deploy_start_ms": deploy,
           "deploy_end_exclusive_ms": deploy + horizon, "training_asof_ms": deploy - 4 * dt,
           "decision_asof_ms": deploy - dt + 2_000, "bar_ms": dt, "purge_bars": 7,
           "universe": universe}
    train_rows = []
    for index in range(4):
        begin = deploy - (10 - index) * horizon
        for rank, symbol in enumerate(universe):
            train_rows.append({
                "row_id": str(begin) + ":" + symbol, "symbol": symbol, "revision_id": revision,
                "feature_bar_open_ms": begin - dt, "feature_bar_end_exclusive_ms": begin,
                "feature_available_at_ms": begin,
                "features": {"return_12": 0.01 * (index + 1), "efficiency_12": 0.40 + rank * 0.1},
                "proposal_long": True, "label_start_ms": begin,
                "label_end_exclusive_ms": begin + horizon, "label_available_at_ms": begin + horizon,
                "gross_return_3d": (1 if (index + rank) % 2 else -1) * 0.02,
            })
    deploy_rows = [{
        "symbol": symbol, "revision_id": revision, "feature_bar_open_ms": deploy - 2 * dt,
        "feature_bar_end_exclusive_ms": deploy - dt, "feature_available_at_ms": deploy - dt + 1_000,
        "features": {"return_12": 0.01, "efficiency_12": 0.5}, "proposal_long": bool(rank == 0),
    } for rank, symbol in enumerate(universe)]
    snapshot = {"schema": "exia.genetic_primus.synthetic_metadata_snapshot/1",
                "evidence_class": api["EVIDENCE"], "revision_id": revision, "universe": universe,
                "training_rows": train_rows, "deploy_feature_rows": deploy_rows,
                "prices_present": False, "actual_indicator_calculation": False}
    # file identity refers to these exact serialized artificial bytes, not OHLCV.
    bind = {"data_file_sha256": hashlib.sha256(api["canonical_bytes"](snapshot) + b"\n").hexdigest(),
            "data_content_sha256": api["sha"](snapshot),
            "complete_grid_sha256": api["sha"]([[r["label_start_ms"], r["symbol"]] for r in train_rows]),
            "timestamp_semantics_sha256": api["sha"]("bar_open_epoch_ms;availability_declared_synthetic"),
            "feature_definition_sha256": api["sha"]("manual_fixed_fixture_values_not_market_indicators"),
            "label_definition_sha256": api["sha"]("manual_3d_gross_outcomes_not_execution_aligned_labels"),
            "revision_id": revision, "universe_sha256": api["sha"](universe)}
    pins = {"context": ctx, "binding": bind, "training_rows_sha256": api["sha"](train_rows),
            "deploy_feature_rows_sha256": api["sha"](deploy_rows)}
    training = sealed(api, {"schema": "exia.genetic_primus.completed_training_payload/1",
        "evidence_class": api["EVIDENCE"], "context": copy.deepcopy(ctx), "binding": copy.deepcopy(bind),
        "train_cutoff_ms": ctx["training_asof_ms"] - 7 * dt,
        "rows": copy.deepcopy(train_rows), "rows_sha256": api["sha"](train_rows)})
    info = api["validate_training"](training, pins)
    scaler = {"feature_names": list(api["FEATURES"]), "median": [0.0, 0.0], "iqr": [1.0, 1.0],
              "fit_rows_sha256": info["scaler_fit_rows_sha256"], "derivation": "manual_fixture_not_fit"}
    model = sealed(api, {"schema": "exia.genetic_primus.synthetic_causal_model_payload/1",
        "evidence_class": api["EVIDENCE"], "fit_mode": "manual_fixture_not_fit",
        "context": copy.deepcopy(ctx), "binding": copy.deepcopy(bind), "features": list(api["FEATURES"]),
        "train_payload_sha256": info["train_payload_sha256"], "train_rows_sha256": info["train_rows_sha256"],
        "scaler_fit_rows_sha256": info["scaler_fit_rows_sha256"], "scaler": scaler,
        "scaler_payload_sha256": api["sha"](scaler), "coefficients": [0.0, 0.1, 0.1],
        "training_asof_ms": ctx["training_asof_ms"], "fit_started_at_ms": ctx["training_asof_ms"] + 1_000,
        "fit_completed_at_ms": ctx["training_asof_ms"] + dt,
        "model_available_at_ms": ctx["training_asof_ms"] + 2 * dt,
    }, "model_sha256")
    deployment = make_deploy(api, pins, deploy_rows, model["model_sha256"])
    return snapshot, pins, {"training": training, "deploy": deployment, "model": model}


def prepare():
    sys.dont_write_bytecode = True
    api = runpy.run_path(str(HELPER))  # Own pure helper only, no research package.
    bank = {}

    def capture(path):
        data = read_bounded(path)
        bank[path.relative_to(ROOT).as_posix()] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        return data

    for path in (HELPER, Path(__file__).resolve(), ROOT / "tools/check_genetic_primus_causal_origin_payload_v1.py"):
        capture(path)
    predecessor = api["strict_json"](capture(PREVIOUS / "completion_receipt.json").decode("utf-8"))
    for name, expected in predecessor["output_byte_sha256"].items():
        if name not in {"proposal.json", "audit.json", "summary.json"}:
            raise ValueError("unexpected predecessor output")
        data = capture(PREVIOUS / name)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("predecessor output changed")
    proposal = api["strict_json"](capture(ROOT / "configs/genetic_primus_forward_research_proposal_v1.json").decode("utf-8"))
    if api["sha"](proposal) != PROPOSAL_SHA or proposal["status"] != "draft_not_executable":
        raise ValueError("separate draft changed")
    if (set(proposal["safety"]) != set(api["SAFETY_KEYS"])
            or any(proposal["safety"][k] is not False for k in api["SAFETY_KEYS"])):
        raise ValueError("safety authority changed")
    if (predecessor["market_research_launch_ready"] is not False
            or predecessor["real_money_ready"] is not False or predecessor["safety"] != proposal["safety"]):
        raise ValueError("predecessor authority changed")
    # Rehash only the predecessor's explicit source/config/metadata bank;
    # never read its historical market data or global dependency fingerprint.
    for relative, expected in predecessor["inputs"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("predecessor path escape")
        data = capture(ROOT / relative)
        if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
            raise ValueError("predecessor input changed")
    snapshot, pins, bundle = fixture(api)
    result = api["validate_bundle"](bundle, pins)
    summary = {"schema": "exia.genetic_primus.causal_origin_payload_summary/1",
               "evidence_class": api["EVIDENCE"], "training_rows": len(bundle["training"]["rows"]),
               "deploy_rows": len(bundle["deploy"]["rows"]),
               "negative_training_outcomes": result["training"]["negative_outcomes"],
               "positive_training_outcomes": result["training"]["positive_outcomes"],
               "new_model_fits": 0, "new_market_strategy_evaluations": 0,
               "manual_model_and_scaler_not_fit": True, "market_data_read": False,
               "old_models_or_collectors_executed": False, "market_research_launch_ready": False,
               "real_money_ready": False, "safety": dict(proposal["safety"]),
               "proposal_semantic_sha256": PROPOSAL_SHA,
               "next_action": "feature_prefix_invariance_and_asof_builder_on_synthetic_candles_before_market_or_training"}
    return api, snapshot, pins, bundle, result, summary, bank


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
    api, snapshot, pins, bundle, result, summary, bank = prepare()
    verify_inputs(bank)
    if not args.verify_only:
        OUTPUT.mkdir(exist_ok=False)
        hashes = {}
        for name, value in (("synthetic_snapshot.json", snapshot), ("fixture_pins.json", pins),
                            ("payloads.json", bundle), ("guard_result.json", result), ("summary.json", summary)):
            data = api["canonical_bytes"](value) + b"\n"
            with (OUTPUT / name).open("xb") as stream:
                stream.write(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        verify_inputs(bank)
        receipt = {"schema": "exia.genetic_primus.causal_origin_payload_receipt/1",
                   "receipt_kind": "synthetic_payload_coherence_not_trained_model_or_market_causality",
                   "inputs": bank, "inputs_verified_unchanged_before_and_after": True,
                   "output_byte_sha256": hashes, "proposal_semantic_sha256": PROPOSAL_SHA,
                   "market_research_launch_ready": False, "real_money_ready": False, "safety": summary["safety"]}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical_bytes"](receipt) + b"\n")
    print("synthetic payloads: 8 completed rows (4 losses), 2 label-free deploy rows; fits=0; all authority=false")


if __name__ == "__main__":
    main()
