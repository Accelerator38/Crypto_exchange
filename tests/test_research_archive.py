from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict[str, object]:
    return json.loads((PROJECT_ROOT / path).read_text(encoding="utf-8"))


def test_archive_has_no_trading_authority() -> None:
    manifest = _json("configs/research_archive/archive_manifest_v1.json")
    hypotheses = _json("configs/research_archive/hypotheses_v1.json")
    safety = manifest["safety"]

    assert safety["legacy_panteon_bitget_live_frozen"] is True
    assert safety["orders_enabled"] is False
    assert safety["promotion_authority"] is False
    assert safety["approved_model_checkpoints"] == 0
    assert safety["approved_live_strategies"] == 0
    assert hypotheses["active_candidates"] == 0
    assert hypotheses["orders_enabled"] is False
    assert hypotheses["promotion_authority"] is False


def test_hypothesis_evidence_paths_exist() -> None:
    hypotheses = _json("configs/research_archive/hypotheses_v1.json")
    for item in hypotheses["hypotheses"]:
        evidence = item.get("evidence")
        if evidence:
            assert (PROJECT_ROOT / evidence).is_file(), item["id"]


def test_archive_dataset_ids_and_paths_are_unique() -> None:
    manifest = _json("configs/research_archive/archive_manifest_v1.json")
    datasets = manifest["retained_local_datasets"]
    ids = [item["id"] for item in datasets]
    paths = [item["path"] for item in datasets]

    assert len(ids) == len(set(ids))
    assert len(paths) == len(set(paths))
