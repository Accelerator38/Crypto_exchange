from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "exia_progress_milestones_v1.json"
BUILDER_PATH = ROOT / "tools" / "build_exia_progress_milestones_v1.py"


def _load_builder():
    spec = importlib.util.spec_from_file_location("exia_progress_milestones_test", BUILDER_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_progress_registry_is_bounded_safe_and_source_backed() -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "exia.progress_milestones.v1"
    assert len(payload["milestones"]) == 8
    assert len({row["milestone_id"] for row in payload["milestones"]}) == 8
    assert set(payload["safety"].values()) == {False}
    for row in payload["milestones"]:
        assert (ROOT / row["report"]).is_file()
        for field in ("metrics", "ledger"):
            if field in row:
                assert (ROOT / row[field]).is_file()


def test_progress_snapshot_keeps_missing_regimes_empty_and_tsv_rectangular(tmp_path: Path) -> None:
    payload = _load_builder().build(CONFIG_PATH, tmp_path)

    assert payload["milestone_count"] == 8
    assert set(payload["safety"].values()) == {False}
    by_id = {row["milestone_id"]: row for row in payload["milestones"]}
    assert by_id["M04_4H_MARKET_QUORUM"]["overall_mean_bps"] > 0
    assert by_id["M04_4H_MARKET_QUORUM"]["overall_lcb_bps"] < 0
    assert by_id["M08_SOL_VALIDATION"]["overall_mean_bps"] > 0
    assert by_id["M08_SOL_VALIDATION"]["median_bps"] < 0
    assert by_id["M08_SOL_VALIDATION"]["overall_lcb_bps"] < 0
    assert by_id["M08_SOL_VALIDATION"]["trend_down_mean_bps"] is None
    assert by_id["M08_SOL_VALIDATION"]["range_neutral_mean_bps"] is None

    with (tmp_path / "milestones.tsv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle, delimiter="\t"))
    assert len(rows) == 9
    assert len({len(row) for row in rows}) == 1
    assert len(rows[0]) >= 20
