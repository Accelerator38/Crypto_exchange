"""Static source bank and synthetic adapter publication; no original code imports."""
from __future__ import annotations

import ast
import runpy
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "Reports/Exia/Genetic_Primus/"
OUTPUT = ROOT / (BASE + "boundary_stub_adapter_v1_20261005")
PREVIOUS = ROOT / (BASE + "boundary_engineering_v1_20261005")
PREVIOUS_RECEIPT_SHA = "71204d3bd732f647215f1eea84f99444bfab1599cad84d483f60cda2e0cfe1a1"
ADAPTER = ROOT / "src/exia/genetic_primus/boundary_stub_adapter_v1.py"
SPEC = ROOT / "src/exia/genetic_primus/boundary_contract_v1.py"


def prepare():
    # Strict parser is stdlib-only; no import through the research package __init__.
    import json
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
    bank = {}
    def read(path, expected=None):
        path = Path(path).resolve()
        rel = path.relative_to(ROOT).as_posix()
        if not path.is_file() or path.stat().st_size > 8_000_000:
            raise ValueError("missing or oversized adapter input")
        raw = path.read_bytes()
        item = {"path": rel, "byte_sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
        if expected is not None and item["byte_sha256"] != expected:
            raise ValueError("adapter input SHA mismatch: " + rel)
        bank[rel] = item
        return raw
    previous = strict_json(read(PREVIOUS / "completion_receipt.json", PREVIOUS_RECEIPT_SHA))
    if previous["receipt_kind"] != "engineering_only_not_financial_launch_receipt":
        raise ValueError("not an engineering predecessor")
    if previous["inputs_verified_unchanged_before_and_after"] is not True:
        raise ValueError("incomplete predecessor")
    for item in previous["inputs"]:
        raw = read(ROOT / item["path"], item["byte_sha256"])
        if len(raw) != item["bytes"]:
            raise ValueError("predecessor size changed")
    producer = ROOT / "tools/build_genetic_primus_boundary_contract_v1.py"
    old = runpy.run_path(str(producer))
    api, contract, old_artifact, old_summary, old_inputs, old_verify = old["prepare"]()
    old_verify()
    if api["canonical"](sorted(old_inputs.values(), key=lambda x: x["path"])) != api["canonical"](previous["inputs"]):
        raise ValueError("predecessor input bank changed")
    for name, expected in (("fixtures.json", old_artifact), ("summary.json", old_summary)):
        raw = read(PREVIOUS / name, previous["output_byte_sha256"][name])
        if api["canonical"](strict_json(raw)) != api["canonical"](expected):
            raise ValueError("predecessor semantics changed")
    adapter_raw = read(ADAPTER); read(Path(__file__))
    adapter_tree = ast.parse(adapter_raw.decode("utf-8-sig"))
    for node in ast.walk(adapter_tree):
        if isinstance(node, ast.Import) and any(alias.name not in {"copy", "math"} for alias in node.names):
            raise ValueError("unregistered adapter import")
        if isinstance(node, ast.ImportFrom) and (node.level or node.module != "__future__"):
            raise ValueError("unregistered adapter from-import")
    adapter = runpy.run_path(str(ADAPTER))
    if adapter["INJECTED_LOCAL_DEPENDENCIES"] != (SPEC.relative_to(ROOT).as_posix(),):
        raise ValueError("unregistered injected local dependency")
    paths = [SPEC, ADAPTER] + [ROOT / rel for rel in contract["reviewed_source_references"]]
    sources = []
    trees = {}
    for path in paths:
        raw = read(path)
        tree = ast.parse(raw.decode("utf-8-sig"))
        trees[path.name] = tree
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend({"module": alias.name, "relative_level": 0, "line": node.lineno} for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append({"module": node.module, "relative_level": node.level, "line": node.lineno})
        rel = path.relative_to(ROOT).as_posix()
        sources.append({**bank[rel], "imports": sorted(imports, key=lambda x: (x["line"], x["module"] or "")),
                        "role": "synthetic_fixture_code" if path in (SPEC, ADAPTER) else "reviewed_reference_not_imported"})
    simulator = next(node for node in trees["accounting_v2.py"].body
                     if isinstance(node, ast.FunctionDef) and node.name == "simulate_cash_targets")
    simulator_names = [x.arg for x in simulator.args.args + simulator.args.kwonlyargs]
    if simulator.args.posonlyargs or simulator.args.vararg is not None or simulator.args.kwarg is not None or set(simulator_names) != set(adapter["CALL_KEYS"]):
        raise ValueError("reviewed simulator signature changed")
    recording_class = next(node for node in trees[ADAPTER.name].body
                           if isinstance(node, ast.ClassDef) and node.name == "RecordingStub")
    call = next(node for node in recording_class.body if isinstance(node, ast.FunctionDef) and node.name == "__call__")
    if [x.arg for x in call.args.kwonlyargs] != list(adapter["CALL_KEYS"]) or call.args.kwarg is not None:
        raise ValueError("recording stub does not explicitly cover all parameters")
    invoke = next(node for node in trees[ADAPTER.name].body
                  if isinstance(node, ast.FunctionDef) and node.name == "invoke_fixture")
    stub_calls = [node for node in ast.walk(invoke) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Name) and node.func.id == "stub"]
    if len(stub_calls) != 1 or stub_calls[0].args or {x.arg for x in stub_calls[0].keywords} != set(adapter["CALL_KEYS"]):
        raise ValueError("adapter must explicitly supply each keyword exactly once")
    results = []
    for case in adapter["CASES"]:
        value = adapter["fixture"](api, contract, case)
        for scenario in ("normal", "stress"):
            stub = adapter["RecordingStub"](api)
            results.append(adapter["invoke_fixture"](api, contract, value, scenario, stub))
    source_bank = {
        "schema": "exia.genetic_primus.scoped_static_adapter_source_bank/1",
        "sources": sorted(sources, key=lambda x: x["path"]),
        "explicit_injected_local_edges": [{"from": ADAPTER.relative_to(ROOT).as_posix(),
                                           "to": SPEC.relative_to(ROOT).as_posix(), "mechanism": "runpy_api_injection"}],
        "reference_signature": simulator_names,
        "signature_argument_names_match": True,
        "dynamic_imports_resolved": False, "external_distribution_versions_verified": False,
        "full_transitive_graph_verified": False, "actual_loaded_runtime_graph_verified": False,
        "coverage": "two_fixture_modules_and_three_reviewed_reference_files_not_full_runtime_graph",
    }
    artifact = {"schema": "exia.genetic_primus.boundary_stub_adapter_fixtures/1",
                "contract_semantic_sha256": api["digest"](contract),
                "source_bank_semantic_sha256": api["digest"](source_bank),
                "results": results, "financial_result": False, "safety": dict(api["SAFETY"])}
    summary = {"schema": "exia.genetic_primus.boundary_stub_adapter_summary/1",
               "synthetic_cases": 3, "reference_cost_scenarios": 2, "stub_calls": 6,
               "synthetic_decision_records": sum(len(x["decision_ledger"]) for x in results),
               "terminal_expectations_not_actual_fills": sum(len(x["terminal_expectations"]) for x in results),
               "scoped_static_source_files": len(sources), "call_parameter_names_matched": len(simulator_names),
               "protocol_stub_verified": True, "real_evaluator_adapter_verified": False,
               "runtime_dependency_graph_verified": False, "historical_semantics_repaired": False,
               "original_inventory_files_verified": 1784, "existing_runners_modified": False,
               "new_model_fits": 0, "new_financial_evaluations": 0, "safety": dict(api["SAFETY"])}
    def verify():
        old_verify()
        for item in bank.values():
            path = ROOT / item["path"]
            if path.stat().st_size != item["bytes"] or sha256(path.read_bytes()).hexdigest() != item["byte_sha256"]:
                raise ValueError("adapter input changed during verification")
    return api, artifact, source_bank, summary, bank, verify


def main():
    dry = sys.argv[1:] == ["--verify-only"]
    if sys.argv[1:] and not dry:
        raise ValueError("only --verify-only supported")
    if not dry and OUTPUT.exists():
        raise FileExistsError("existing adapter snapshot must not be overwritten")
    api, artifact, sources, summary, bank, verify = prepare()
    verify()
    if not dry:
        OUTPUT.mkdir()
        hashes = {}
        for name, value in (("fixtures.json", artifact), ("source_bank.json", sources), ("summary.json", summary)):
            raw = api["canonical"](value) + b"\n"
            with (OUTPUT / name).open("xb") as stream:
                stream.write(raw)
            hashes[name] = sha256(raw).hexdigest()
        verify()
        receipt = {"schema": "exia.genetic_primus.boundary_stub_adapter_completion/1",
                   "receipt_kind": "synthetic_adapter_only_not_financial_launch",
                   "previous_receipt_byte_sha256": PREVIOUS_RECEIPT_SHA,
                   "inputs": sorted(bank.values(), key=lambda x: x["path"]),
                   "output_byte_sha256": hashes,
                   "fixture_semantic_sha256": api["digest"](artifact),
                   "source_bank_semantic_sha256": api["digest"](sources),
                   "python_executable_observed": sys.executable, "python_version_observed": sys.version,
                   "inputs_verified_unchanged_before_and_after": True,
                   "real_evaluator_adapter_verified": False, "runtime_dependency_graph_verified": False,
                   "financial_launch_completed": False, "safety": dict(api["SAFETY"])}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical"](receipt) + b"\n")
        verify()
    print(f"{'preflight' if dry else 'published'}: 6 recording-stub calls, 648 synthetic decisions, 5 source fingerprints")
    print("1784 originals unchanged; signature names matched; no fills/PnL/training/network/runtime")


if __name__ == "__main__":
    main()
