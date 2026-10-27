"""Independently reconstruct metadata links and verify receipt/input/output bytes."""

from hashlib import sha256
import json
from pathlib import Path
import runpy


def main():
    root = Path(__file__).resolve().parents[1]
    builder_path = root / "tools/build_genetic_primus_result_context_links_v1.py"
    module_path = root / "src/exia/genetic_primus/result_context_links_v1.py"
    output = root / "Reports/Exia/Genetic_Primus/result_context_links_v1_20261005"
    receipt = json.loads((output / "completion_receipt.json").read_bytes())
    assert receipt["inputs_verified_unchanged_before_and_after"] is True
    assert sha256(builder_path.read_bytes()).hexdigest() == receipt["builder_byte_sha256"]
    assert sha256(module_path.read_bytes()).hexdigest() == receipt["module_byte_sha256"]
    builder = runpy.run_path(str(builder_path))
    api, expected, summary, inputs, verify, _, _ = builder["prepare"]()
    verify()
    for name, value in (("context_links.json", expected), ("summary.json", summary)):
        raw = (output / name).read_bytes()
        assert sha256(raw).hexdigest() == receipt["output_byte_sha256"][name]
        assert api["canonical"](json.loads(raw)) == api["canonical"](value)
    assert api["canonical"](receipt["input_files"]) == api["canonical"](list(inputs.values()))
    assert len(expected["sources"]) == 1784
    assert summary["linked_sources"] == 23 and summary["linked_evaluations"] == 67
    flags = ("credentials_allowed", "orders_enabled", "paper", "live", "execution_allowed", "promotion_authority")
    assert all(expected[flag] is receipt[flag] is False for flag in flags)
    assert expected["cross_source_comparison_authority"] is False
    base_catalog = json.loads((root / builder["CATALOG_DIR"] / "catalog.json").read_bytes())
    base_by_path = {e["source"]["path"]: e for e in base_catalog["records"]}
    for family in expected["families"]:
        assert family["comparison_authority"] is False and family["raw_market_data_hashes_recomputed"] is False
        for binding in family["bindings"]:
            assert inputs[binding["source_path"]]["byte_sha256"] == binding["source_byte_sha256"]
            assert inputs[binding["target_path"]]["byte_sha256"] == binding["target_byte_sha256"]
        for link in family["evaluation_links"]:
            original = base_by_path[link["source_path"]]
            assert original["envelope_sha256"] == link["base_envelope_sha256"]
            evaluation = next(e for e in original["evaluations"] if (e["policy"], e["scenario"]) == (link["policy"], link["scenario"]))
            assert api["digest"](evaluation["metrics"]) == link["base_metrics_semantic_sha256"]
            for field in link["fields"]:
                assert (field["value"] is None) == (field["unknown_reason"] is not None)
                assert field["evidence"] or field["value"] is None
                assert field["unit"] == api["UNITS"][field["name"]]
                for proof in field["evidence"]:
                    assert inputs[proof["path"]]["byte_sha256"] == proof["byte_sha256"]
    print("context artifacts: receipt/provenance/metrics verified; 1784 originals and frozen contract unchanged")
    print("67 linked evaluations; no new comparison, validation, training or trading authority.")


if __name__ == "__main__":
    main()
