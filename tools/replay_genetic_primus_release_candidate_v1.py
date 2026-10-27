"""Explicit, bounded local replay. No network, training or runtime integration.

Input is a JSON bundle with settings, rules, signed models and ordered events.
Output is a NEW file in the scoped release report directory. No fallback to
OHLCV, automatic parameter estimation, forced closing or production approval.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "exia" / "genetic_primus"))
from release_contract_v1 import object_sha, validate_contract
from execution_replay_v1 import Replay, dec

SOURCE_FILES = (
    "src/exia/genetic_primus/release_contract_v1.py",
    "src/exia/genetic_primus/execution_replay_v1.py",
    "tools/replay_genetic_primus_release_candidate_v1.py",
)


def implementation_identity(root=ROOT):
    pins = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    return {"files": pins, "sha256": object_sha(pins)}


def replay_bundle(bundle, contract, implementation_sha256):
    """Pure entry point; paths and file-size limits live only in main()."""
    required = {"schema_version", "settings", "rules", "models", "events"}
    if not isinstance(bundle, dict) or set(bundle) != required or bundle["schema_version"] != "exia.genetic_primus.execution_bundle/1":
        raise ValueError("unsupported replay bundle")
    check = validate_contract(contract)
    if not check["structurally_valid"]:
        raise ValueError("invalid release contract")
    settings, events = bundle["settings"], bundle["events"]
    if settings["contract_sha256"] != check["contract_sha256"]:
        raise ValueError("bundle contract mismatch")
    if not set(bundle["rules"]).issubset(set(contract["symbol_map"].values())):
        raise ValueError("instrument outside release universe")
    for contract_key, setting_key in (
            ("capital_usdt", "initial_cash"), ("maximum_order_notional_usdt", "max_order_notional"),
            ("maximum_loss_usdt", "max_loss"), ("maximum_drawdown_fraction", "max_drawdown")):
        budget = contract["risk_budget"][contract_key]
        if budget is not None and dec(budget) != dec(settings[setting_key]):
            raise ValueError("execution risk profile differs from frozen contract")
    if not isinstance(events, list) or len(events) > min(10000, settings["max_events"]):
        raise ValueError("bounded events list required")
    dataset_sha = object_sha({"rules": bundle["rules"], "events": events})
    if settings["dataset_sha256"] != dataset_sha:
        raise ValueError("execution dataset hash mismatch")
    # Independent recomputation checks ALL scaler/policy/provenance fields, not
    # only an attacker-supplied digest. Hashes are integrity, not authenticity.
    from release_contract_v1 import bind_model
    for signed in bundle["models"]:
        body = signed["model"]
        p = body["provenance"]
        if p["implementation_sha256"] != implementation_sha256:
            raise ValueError("execution implementation hash mismatch")
        rebuilt = bind_model(contract, origin_start_ms=body["origin_start_ms"], scaler=body["scaler"],
                             data_sha256=p["data_sha256"], train_rows_sha256=p["train_rows_sha256"],
                             folds_sha256=p["folds_sha256"], resample_bank_sha256=p["resample_bank_sha256"],
                             audit_bank_sha256=p["audit_bank_sha256"], implementation_sha256=implementation_sha256)
        if rebuilt != signed:
            raise ValueError("release model manifest mismatch")
    replay = Replay(settings=settings, rules=bundle["rules"], models=bundle["models"])
    for event in events:
        replay.ingest(event)
    result = replay.result()
    result["input_bundle_sha256"] = object_sha(bundle)
    result["release_check"] = check
    result["checkpoint"] = replay.checkpoint()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="local JSON, at most 2 MiB")
    parser.add_argument("--output", required=True, help="NEW scoped JSON result")
    args = parser.parse_args()
    source = (ROOT / args.input).resolve()
    destination = (ROOT / args.output).resolve()
    allowed = (ROOT / "Reports/Exia/Genetic_Primus/release_execution_v1_20260908").resolve()
    if not source.is_relative_to(ROOT) or source.suffix != ".json" or not source.is_file():
        raise ValueError("explicit local workspace JSON input required")
    if source.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("input exceeds 2 MiB safety budget")
    if destination.parent != allowed or destination.suffix != ".json" or destination.exists():
        raise ValueError("output must be a NEW JSON in the scoped release report directory")
    # Bounded read also protects against an input growing after the stat check.
    with source.open("rb") as stream:
        raw = stream.read(2 * 1024 * 1024 + 1)
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("input grew beyond the safety budget")
    config = json.loads((ROOT / "configs/genetic_primus_release_candidate_v1.json").read_text(encoding="utf-8"))
    check = validate_contract(config, root=ROOT)
    if not check["structurally_valid"]:
        raise ValueError("frozen research source integrity failure")
    identity = implementation_identity()
    result = replay_bundle(json.loads(raw.decode("utf-8")), config, identity["sha256"])
    result["release_check"] = check
    result["implementation"] = identity
    result["input_file_sha256"] = hashlib.sha256(raw).hexdigest()
    allowed.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=True, indent=2, allow_nan=False)
    print(json.dumps({"event_count": result["event_count"], "evidence_kind": result["evidence_kind"],
                      "flat_and_reconciled": result["flat_and_reconciled"],
                      "risk_latched": result["risk_latched"], "state_sha256": result["state_sha256"],
                      "trading_authorized": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
