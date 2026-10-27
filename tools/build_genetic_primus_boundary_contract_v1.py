"""Publish engineering fixtures/provenance; never execute the trading evaluator."""
from __future__ import annotations

import ast
import json
import runpy
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "Reports/Exia/Genetic_Primus/"
OUTPUT = ROOT / (BASE + "boundary_engineering_v1_20261005")
PREVIOUS = ROOT / (BASE + "accounting_semantics_audit_v1_20261005")
PREVIOUS_RECEIPT_SHA = "fa42dc5bcb947b88328ea5ce298c3a25b22833cfeddf1fd9fa3c86304a6414a9"
MODULE = ROOT / "src/exia/genetic_primus/boundary_contract_v1.py"
CONFIG = ROOT / "configs/genetic_primus_boundary_engineering_v1.json"


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def invalid(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def prepare():
    bank = {}
    def read(path, expected=None):
        path = Path(path).resolve()
        rel = path.relative_to(ROOT).as_posix()
        if not path.is_file() or path.stat().st_size > 8_000_000:
            raise ValueError("missing or oversized engineering input")
        raw = path.read_bytes()
        byte_sha = sha256(raw).hexdigest()
        if expected is not None and byte_sha != expected:
            raise ValueError(f"input hash mismatch: {rel}")
        bank[rel] = {"path": rel, "byte_sha256": byte_sha, "bytes": len(raw)}
        return raw

    previous = strict_json(read(PREVIOUS / "completion_receipt.json", PREVIOUS_RECEIPT_SHA))
    if previous["inputs_verified_unchanged_before_and_after"] is not True:
        raise ValueError("prior audit not complete")
    old_builder = ROOT / "tools/build_genetic_primus_accounting_semantics_audit_v1.py"
    old_module = ROOT / "src/exia/genetic_primus/accounting_semantics_audit_v1.py"
    read(old_builder, previous["builder_byte_sha256"])
    read(old_module, previous["module_byte_sha256"])
    # Only the pure metadata builder is loaded; referenced trading code is bytes/AST.
    old = runpy.run_path(str(old_builder))
    old_api, old_audit, old_summary, old_inputs, old_verify, _, _ = old["prepare"]()
    old_verify()  # Populates the predecessor's full frozen inventory input bank.
    if old_api["canonical"](list(old_inputs.values())) != old_api["canonical"](previous["input_files"]):
        raise ValueError("previous input bank mismatch")
    for name, expected_value in (("accounting_audit.json", old_audit), ("summary.json", old_summary)):
        raw = read(PREVIOUS / name, previous["output_byte_sha256"][name])
        if old_api["canonical"](strict_json(raw)) != old_api["canonical"](expected_value):
            raise ValueError("previous audit changed")
    if old_summary["new_model_fits"] != 0 or old_summary["new_strategy_evaluation_runs_started"] != 0:
        raise ValueError("not an engineering-only predecessor")

    read(MODULE); read(Path(__file__))
    api = runpy.run_path(str(MODULE))
    contract = api["validate_contract"](strict_json(read(CONFIG)))
    for rel, expected in contract["reviewed_source_references"].items():
        read(ROOT / rel, expected)
    raw_accounting = (ROOT / "src/exia/genetic_primus/accounting_v2.py").read_bytes()
    tree = ast.parse(raw_accounting.decode("utf-8-sig"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "exit_sleeve"
             and len(node.args) == 4 and ast.unparse(node.args[3]) ==
             "'retrain_forced_exit' if reset else 'target_change'"]
    if len(calls) != 1 or ast.unparse(calls[0].args[1]) != "float(opens[row, column])":
        raise ValueError("reviewed reset is no longer open-price exit")

    day = api["DAY_MS"]
    plan = api["boundary_plan"](contract, 0, 9 * day)
    transitions = []
    for position in (-1, 0, 1):
        for target in (-1, 0, 1):
            result = api["boundary_transition"](
                contract, position=position, target=target,
                entry_model_sha256="1" * 64 if position else None,
                decision_model_sha256="2" * 64, origin_open_ms=3 * day,
                signal_close_ms=3 * day, model_available_ms=3 * day,
                interval_end_ms=9 * day)
            transitions.append({"position_before": position, "target": target, "result": result})
    identities = [api["long_cash_identity_fixture"](1.0, 100.0, price, fee)
                  for price in (90.0, 110.0) for fee in (0.0, 16.0)]
    for item in identities:
        if abs(item["gross_pnl"] - item["fees"] - item["net_pnl"]) > 1e-12:
            raise ValueError("artificial cash identity failed")
    artifact = {
        "schema": "exia.genetic_primus.boundary_engineering_fixtures/1",
        "contract_byte_sha256": bank[CONFIG.relative_to(ROOT).as_posix()]["byte_sha256"],
        "contract_semantic_sha256": api["digest"](contract),
        "fixture_timeline": "synthetic_epoch_origin_not_a_market_period",
        "boundary_plan": plan, "direction_transitions": transitions,
        "long_cash_identities": identities,
        "reference_source_review": {"reset_call_line": calls[0].lineno,
                                    "reset_price": "candle_open_not_previous_close",
                                    "runtime_dependency_identity_verified": False},
        "financial_result": False, "launch_completed": False,
        "original_financial_evaluations_unchanged": True,
        "safety": dict(api["SAFETY"]),
    }
    summary = {"schema": "exia.genetic_primus.boundary_engineering_summary/1",
               "synthetic_origins": 3, "direction_transition_fixtures": 9,
               "single_long_cash_identity_fixtures": 4,
               "reviewed_source_reference_hashes": len(contract["reviewed_source_references"]),
               "historical_semantics_repaired": False, "existing_runners_modified": False,
               "real_evaluator_adapter_verified": False,
               "runtime_dependency_graph_verified": False,
               "new_model_fits": 0, "new_financial_evaluations": 0,
               "original_inventory_files_verified": 1784,
               "safety": dict(api["SAFETY"])}
    def verify():
        old_verify()
        for item in bank.values():
            path = ROOT / item["path"]
            if path.stat().st_size != item["bytes"] or sha256(path.read_bytes()).hexdigest() != item["byte_sha256"]:
                raise ValueError("engineering input changed")
    return api, contract, artifact, summary, bank, verify


def main():
    dry = sys.argv[1:] == ["--verify-only"]
    if sys.argv[1:] and not dry:
        raise ValueError("only --verify-only supported")
    if not dry and OUTPUT.exists():
        raise FileExistsError("existing engineering snapshot must not be overwritten")
    api, contract, artifact, summary, bank, verify = prepare()
    verify()
    if not dry:
        OUTPUT.mkdir()
        hashes = {}
        for name, value in (("fixtures.json", artifact), ("summary.json", summary)):
            raw = api["canonical"](value) + b"\n"
            with (OUTPUT / name).open("xb") as stream:
                stream.write(raw)
            hashes[name] = sha256(raw).hexdigest()
        verify()
        receipt = {
            "schema": "exia.genetic_primus.boundary_engineering_receipt/1",
            "receipt_kind": "engineering_only_not_financial_launch_receipt",
            "previous_receipt_byte_sha256": PREVIOUS_RECEIPT_SHA,
            "inputs": sorted(bank.values(), key=lambda item: item["path"]),
            "output_byte_sha256": hashes,
            "fixture_semantic_sha256": api["digest"](artifact),
            "contract_semantic_sha256": api["digest"](contract),
            "python_executable_observed": sys.executable,
            "python_version_observed": sys.version,
            "source_reference_bank_sha256": api["digest"](contract["reviewed_source_references"]),
            "inputs_verified_unchanged_before_and_after": True,
            "future_launch_requirements": contract["launch_gate"],
            "runtime_dependency_graph_verified": False,
            "safety": dict(api["SAFETY"]),
        }
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical"](receipt) + b"\n")
        verify()
    print(f"{'preflight' if dry else 'published'}: 3 synthetic origins, 9 transitions, 4 algebra fixtures")
    print("1784 originals unchanged; carry/terminal explicit; no evaluator/training/network/runtime")


if __name__ == "__main__":
    main()
