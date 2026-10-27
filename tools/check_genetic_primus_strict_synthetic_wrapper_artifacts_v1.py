"""Read-only receipt/output revalidation for the closed synthetic wrapper."""
from __future__ import annotations

import json
import runpy
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/strict_synthetic_wrapper_v1_20261006"


def main():
    receipt = json.loads((OUTPUT / "completion_receipt.json").read_bytes())
    if receipt["receipt_kind"] != "synthetic_closed_interval_wrapper_not_market_launch":
        raise ValueError("unexpected receipt kind")
    for item in receipt["inputs"]:
        path = Path(item["path"]) if item["external_package_file"] else (ROOT / item["path"]).resolve()
        if not item["external_package_file"]:
            path.relative_to(ROOT)
        if path.stat().st_size != item["bytes"] or sha256(path.read_bytes()).hexdigest() != item["byte_sha256"]:
            raise ValueError("wrapper input changed")
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_strict_synthetic_wrapper_v1.py"))
    api, artifact, environment, summary, inputs, verify = builder["prepare"]()
    verify()
    if api["canonical"](receipt["inputs"]) != api["canonical"](sorted(inputs.values(), key=lambda x: x["path"])):
        raise ValueError("wrapper input bank mismatch")
    for name, expected in (("wrapper.json", artifact), ("environment.json", environment), ("summary.json", summary)):
        raw = (OUTPUT / name).read_bytes()
        if sha256(raw).hexdigest() != receipt["output_byte_sha256"][name]:
            raise ValueError("wrapper output bytes changed")
        if api["canonical"](json.loads(raw)) != api["canonical"](expected):
            raise ValueError("wrapper output semantics changed")
    if receipt["wrapper_semantic_sha256"] != api["digest"](artifact) or receipt["environment_semantic_sha256"] != api["digest"](environment):
        raise ValueError("wrapper/environment digest mismatch")
    if receipt["previous_receipt_byte_sha256"] != builder["PREVIOUS_RECEIPT_SHA"]:
        raise ValueError("predecessor changed")
    if receipt["inputs_verified_unchanged_before_and_after"] is not True:
        raise ValueError("incomplete wrapper verification")
    if receipt["runtime_dependency_graph_verified"] is not False or receipt["market_strategy_launch_completed"] is not False:
        raise ValueError("unearned authority")
    if api["canonical"](receipt["safety"]) != api["canonical"](api["SAFETY"]):
        raise ValueError("wrapper safety drift")
    print("strict wrapper artifacts: receipt/inputs/model bindings/trade-linked sidecar verified")
    print("1784 originals unchanged; all 6 accounting outputs identical; no market/training/runtime authority")


if __name__ == "__main__":
    main()
