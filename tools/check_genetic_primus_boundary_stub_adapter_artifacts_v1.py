"""Receipt/identity verification for synthetic adapter outputs only."""
from __future__ import annotations

import json
import runpy
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/boundary_stub_adapter_v1_20261005"


def main():
    receipt = json.loads((OUTPUT / "completion_receipt.json").read_bytes())
    if receipt["receipt_kind"] != "synthetic_adapter_only_not_financial_launch":
        raise ValueError("not a synthetic adapter receipt")
    for item in receipt["inputs"]:
        path = (ROOT / item["path"]).resolve()
        path.relative_to(ROOT)
        if path.stat().st_size != item["bytes"] or sha256(path.read_bytes()).hexdigest() != item["byte_sha256"]:
            raise ValueError("receipt input changed")
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_boundary_stub_adapter_v1.py"))
    api, artifact, sources, summary, inputs, verify = builder["prepare"]()
    verify()
    if api["canonical"](receipt["inputs"]) != api["canonical"](sorted(inputs.values(), key=lambda x: x["path"])):
        raise ValueError("adapter input bank mismatch")
    for name, expected in (("fixtures.json", artifact), ("source_bank.json", sources), ("summary.json", summary)):
        raw = (OUTPUT / name).read_bytes()
        if sha256(raw).hexdigest() != receipt["output_byte_sha256"][name]:
            raise ValueError("adapter output bytes changed")
        if api["canonical"](json.loads(raw)) != api["canonical"](expected):
            raise ValueError("adapter output semantics changed")
    for key, value in (("fixture_semantic_sha256", artifact), ("source_bank_semantic_sha256", sources)):
        if receipt[key] != api["digest"](value):
            raise ValueError("semantic digest mismatch")
    if receipt["previous_receipt_byte_sha256"] != builder["PREVIOUS_RECEIPT_SHA"]:
        raise ValueError("predecessor identity mismatch")
    if receipt["inputs_verified_unchanged_before_and_after"] is not True:
        raise ValueError("incomplete input verification")
    for key in ("real_evaluator_adapter_verified", "runtime_dependency_graph_verified", "financial_launch_completed"):
        if receipt[key] is not False:
            raise ValueError("unearned authority")
    if api["canonical"](receipt["safety"]) != api["canonical"](api["SAFETY"]):
        raise ValueError("safety drift")
    print("stub adapter artifacts: receipt/source bank/ledger identities verified; 1784 originals unchanged")
    print("Terminal records are expectations, not fills; no real evaluator, training or trading authority")


if __name__ == "__main__":
    main()
