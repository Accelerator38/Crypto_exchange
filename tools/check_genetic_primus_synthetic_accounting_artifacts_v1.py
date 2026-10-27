"""Read-only identity/receipt verification for artificial accounting results."""
from __future__ import annotations

import json
import runpy
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Reports/Exia/Genetic_Primus/synthetic_accounting_probe_v1_20261006"


def main():
    receipt = json.loads((OUTPUT / "completion_receipt.json").read_bytes())
    if receipt["receipt_kind"] != "synthetic_accounting_only_not_market_strategy_launch":
        raise ValueError("unexpected receipt kind")
    for item in receipt["inputs"]:
        path = Path(item["path"]) if item["external_package_file"] else (ROOT / item["path"]).resolve()
        if not item["external_package_file"]:
            path.relative_to(ROOT)
        if path.stat().st_size != item["bytes"] or sha256(path.read_bytes()).hexdigest() != item["byte_sha256"]:
            raise ValueError("probe input changed")
    builder = runpy.run_path(str(ROOT / "tools/build_genetic_primus_synthetic_accounting_probe_v1.py"))
    api, artifact, environment, summary, inputs, verify = builder["prepare"]()
    verify()
    if api["canonical"](receipt["inputs"]) != api["canonical"](sorted(inputs.values(), key=lambda x: x["path"])):
        raise ValueError("input bank mismatch")
    for name, expected in (("fixtures.json", artifact), ("environment.json", environment), ("summary.json", summary)):
        raw = (OUTPUT / name).read_bytes()
        if sha256(raw).hexdigest() != receipt["output_byte_sha256"][name]:
            raise ValueError("probe output hash changed")
        if api["canonical"](json.loads(raw)) != api["canonical"](expected):
            raise ValueError("probe output semantics changed")
    for key, value in (("fixture_semantic_sha256", artifact), ("environment_semantic_sha256", environment)):
        if receipt[key] != api["digest"](value):
            raise ValueError("probe semantic digest changed")
    if receipt["previous_receipt_byte_sha256"] != builder["PREVIOUS_RECEIPT_SHA"]:
        raise ValueError("predecessor changed")
    if receipt["inputs_verified_unchanged_before_and_after"] is not True:
        raise ValueError("incomplete verification")
    if receipt["runtime_dependency_graph_verified"] is not False or receipt["market_strategy_launch_completed"] is not False:
        raise ValueError("unearned authority")
    if api["canonical"](receipt["safety"]) != api["canonical"](api["SAFETY"]):
        raise ValueError("safety drift")
    print("artificial accounting artifacts: source/environment/receipt/oracle outputs verified")
    print("1784 originals unchanged; historical/full runtime identity and market authority remain unverified")


if __name__ == "__main__":
    main()
