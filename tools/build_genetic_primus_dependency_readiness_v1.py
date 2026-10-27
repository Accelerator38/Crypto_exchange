"""Observe bounded loaded-file fingerprints and publish fail-closed readiness."""
from __future__ import annotations

import json
import runpy
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "Reports/Exia/Genetic_Primus/"
OUTPUT = ROOT / (BASE + "dependency_readiness_v1_20261006")
PREVIOUS = ROOT / (BASE + "strict_synthetic_wrapper_v1_20261006")
PREVIOUS_RECEIPT_SHA = "99c9b74e077df4844fb84aa565806f7c090c86225739999d102aa03d8535f76a"
MODULE = ROOT / "src/exia/genetic_primus/observed_dependency_readiness_v1.py"


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
        raw = path.read_bytes()
        if len(raw) > 8_000_000 or (expected is not None and sha256(raw).hexdigest() != expected):
            raise ValueError("readiness metadata hash/size mismatch")
        bank[str(path)] = {"path": str(path), "byte_sha256": sha256(raw).hexdigest(), "bytes": len(raw)}
        return raw
    receipt = strict_json(read(PREVIOUS / "completion_receipt.json", PREVIOUS_RECEIPT_SHA))
    if receipt["receipt_kind"] != "synthetic_closed_interval_wrapper_not_market_launch":
        raise ValueError("unexpected predecessor")
    if receipt["inputs_verified_unchanged_before_and_after"] is not True:
        raise ValueError("incomplete predecessor")
    for item in receipt["inputs"]:
        path = Path(item["path"]) if item["external_package_file"] else ROOT / item["path"]
        if len(read(path, item["byte_sha256"])) != item["bytes"]:
            raise ValueError("predecessor size changed")
    outputs = {}
    for name, expected_sha in receipt["output_byte_sha256"].items():
        outputs[name] = strict_json(read(PREVIOUS / name, expected_sha))
    metadata = runpy.run_path(str(ROOT / "tools/build_genetic_primus_boundary_stub_adapter_v1.py"))
    api, _, _, _, _, original_verify = metadata["prepare"]()
    original_verify()
    read(MODULE); read(Path(__file__))
    observation = runpy.run_path(str(MODULE))
    probe = runpy.run_path(str(ROOT / "src/exia/genetic_primus/synthetic_accounting_probe_v1.py"))
    wrapper = runpy.run_path(str(ROOT / "src/exia/genetic_primus/strict_synthetic_wrapper_v1.py"))
    adapter = runpy.run_path(str(ROOT / "src/exia/genetic_primus/boundary_stub_adapter_v1.py"))
    contract = api["expected_contract"]()
    payloads = {case: adapter["fixture"](api, contract, case) for case in adapter["CASES"]}
    for value in payloads.values():
        wrapper["validate_inputs"](api, contract, value, "normal")
    namespace, environment, denied = probe["load_verified_accounting"](ROOT, OUTPUT)
    previous_results = {(x["case"], x["scenario"]): x for x in outputs["wrapper.json"]["results"]}
    if len(previous_results) != 6:
        raise ValueError("six unique predecessor cases required")
    preserved = 0
    for case in adapter["CASES"]:
        for scenario in ("normal", "stress"):
            actual = wrapper["run_wrapper"](api, contract, payloads[case], scenario, namespace)
            if api["canonical"](actual) != api["canonical"](previous_results[case, scenario]):
                raise AssertionError("registered synthetic wrapper output changed")
            preserved += 1
    if any(event != "socket.gethostname" for event in denied):
        raise PermissionError("unexpected blocked operation")
    roots = [Path(sys.prefix), Path(sys.base_prefix), ROOT / "tools", ROOT / "src/exia/genetic_primus"]
    explicit = [("sys_executable_observed", Path(sys.executable)),
                ("base_executable_observed", Path(getattr(sys, "_base_executable", sys.executable)))]
    if sys.platform == "win32":
        explicit.append(("declared_core_dll_candidate_not_memory_attested",
                         Path(sys.base_prefix) / f"python{sys.version_info.major}{sys.version_info.minor}.dll"))
    fingerprint = observation["snapshot"](api, dict(sys.modules), roots, explicit)
    observation["validate_fingerprint"](api, fingerprint, roots)
    evidence = {"wrapper": {"path": str(PREVIOUS / "wrapper.json"), "byte_sha256": receipt["output_byte_sha256"]["wrapper.json"]},
                "fingerprint": {"path": str(OUTPUT / "dependency_fingerprint.json"), "semantic_sha256": api["digest"](fingerprint)}}
    ready = observation["readiness"](api, fingerprint, outputs["summary.json"], evidence)
    file_paths = set()
    for item in fingerprint["files"]:
        key = item["path"]
        file_paths.add(key)
        converted = {"path": key, "byte_sha256": item["byte_sha256"], "bytes": item["bytes"]}
        if key in bank and bank[key] != converted:
            raise ValueError("observed file conflicts with pre-bound metadata")
        bank[key] = converted
    summary = {"schema": "exia.genetic_primus.dependency_readiness_summary/1",
               **fingerprint["counts"], "synthetic_outputs_preserved": preserved,
               "scoped_engineering_gates_verified": 4, "unclosed_gates": 9,
               "full_runtime_dependency_graph_verified": False, "market_research_launch_ready": False,
               "real_money_ready": False, "original_inventory_files_verified": 1784,
               "new_model_fits": 0, "new_market_strategy_evaluations": 0, "safety": dict(api["SAFETY"])}
    def verify():
        original_verify()
        observation["verify_files"](fingerprint)
        for item in bank.values():
            if item["path"] in file_paths:
                continue
            path = Path(item["path"])
            if path.stat().st_size != item["bytes"] or sha256(path.read_bytes()).hexdigest() != item["byte_sha256"]:
                raise ValueError("readiness input identity changed")
    return api, fingerprint, ready, environment, summary, bank, verify


def main():
    dry = sys.argv[1:] == ["--verify-only"]
    if sys.argv[1:] and not dry:
        raise ValueError("only --verify-only supported")
    if not dry and OUTPUT.exists():
        raise FileExistsError("existing readiness snapshot must not be overwritten")
    api, fingerprint, ready, environment, summary, bank, verify = prepare()
    verify()
    if not dry:
        OUTPUT.mkdir()
        hashes = {}
        for name, value in (("dependency_fingerprint.json", fingerprint), ("readiness.json", ready),
                            ("environment.json", environment), ("summary.json", summary)):
            raw = api["canonical"](value) + b"\n"
            with (OUTPUT / name).open("xb") as stream:
                stream.write(raw)
            hashes[name] = sha256(raw).hexdigest()
        verify()
        completion = {"schema": "exia.genetic_primus.dependency_readiness_completion/1",
                      "receipt_kind": "observed_files_and_engineering_readiness_not_market_launch",
                      "previous_receipt_byte_sha256": PREVIOUS_RECEIPT_SHA,
                      "inputs": sorted(bank.values(), key=lambda x: x["path"]),
                      "output_byte_sha256": hashes, "fingerprint_semantic_sha256": api["digest"](fingerprint),
                      "readiness_semantic_sha256": api["digest"](ready),
                      "inputs_verified_unchanged_before_and_after": True,
                      "full_runtime_dependency_graph_verified": False, "market_research_launch_ready": False,
                      "real_money_ready": False, "safety": dict(api["SAFETY"])}
        with (OUTPUT / "completion_receipt.json").open("xb") as stream:
            stream.write(api["canonical"](completion) + b"\n")
        verify()
    print(f"{'preflight' if dry else 'published'}: {summary['module_entries']} modules, {summary['unique_files']} files, {summary['file_bytes']} bytes")
    print("4 scoped gates verified, 9 unclosed; 6 synthetic outputs preserved; 1784 originals unchanged; no market/training/runtime")


if __name__ == "__main__":
    main()
