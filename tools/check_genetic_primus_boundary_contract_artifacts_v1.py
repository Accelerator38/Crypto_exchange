"""Verify engineering receipt and fixtures, without executing the evaluator."""
from __future__ import annotations

import runpy
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    builder_path = ROOT / "tools/build_genetic_primus_boundary_contract_v1.py"
    builder = runpy.run_path(str(builder_path))
    receipt = builder["strict_json"]((builder["OUTPUT"] / "completion_receipt.json").read_bytes())
    if receipt["receipt_kind"] != "engineering_only_not_financial_launch_receipt":
        raise ValueError("invalid receipt kind")
    for item in receipt["inputs"]:
        path = (ROOT / item["path"]).resolve()
        path.relative_to(ROOT)
        if sha256(path.read_bytes()).hexdigest() != item["byte_sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError("receipt input changed")
    api, contract, artifact, summary, inputs, verify = builder["prepare"]()
    verify()
    if api["canonical"](receipt["inputs"]) != api["canonical"](sorted(inputs.values(), key=lambda item: item["path"])):
        raise ValueError("input bank changed")
    for name, expected in (("fixtures.json", artifact), ("summary.json", summary)):
        raw = (builder["OUTPUT"] / name).read_bytes()
        if sha256(raw).hexdigest() != receipt["output_byte_sha256"][name]:
            raise ValueError("output bytes changed")
        if api["canonical"](builder["strict_json"](raw)) != api["canonical"](expected):
            raise ValueError("output semantics changed")
    if receipt["contract_semantic_sha256"] != api["digest"](contract):
        raise ValueError("contract digest changed")
    if receipt["fixture_semantic_sha256"] != api["digest"](artifact):
        raise ValueError("fixture digest changed")
    if receipt["source_reference_bank_sha256"] != api["digest"](contract["reviewed_source_references"]):
        raise ValueError("reference digest changed")
    if receipt["previous_receipt_byte_sha256"] != builder["PREVIOUS_RECEIPT_SHA"]:
        raise ValueError("predecessor receipt changed")
    if receipt["inputs_verified_unchanged_before_and_after"] is not True:
        raise ValueError("incomplete verification")
    if api["canonical"](receipt["future_launch_requirements"]) != api["canonical"](contract["launch_gate"]):
        raise ValueError("launch requirements changed")
    if receipt["runtime_dependency_graph_verified"] is not False or api["canonical"](receipt["safety"]) != api["canonical"](api["SAFETY"]):
        raise ValueError("authority drift")
    print("boundary artifacts: receipt, explicit parameters and fixture identities verified")
    print("1784 originals unchanged; actual evaluator adapter/launch still unverified and disabled")


if __name__ == "__main__":
    main()
