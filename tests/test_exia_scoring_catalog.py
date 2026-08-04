from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from exia.catalog import build_catalog
from exia.contracts import ExperimentManifest
from exia.scoring import moving_block_lcb, summarize_scope


def test_block_lcb_is_deterministic_and_clusters_days() -> None:
    timestamps = pd.Series(
        [
            int(pd.Timestamp("2026-01-01T01:00:00Z").timestamp() * 1000),
            int(pd.Timestamp("2026-01-01T02:00:00Z").timestamp() * 1000),
            int(pd.Timestamp("2026-01-02T01:00:00Z").timestamp() * 1000),
            int(pd.Timestamp("2026-01-02T02:00:00Z").timestamp() * 1000),
        ]
    )
    values = pd.Series([10.0, 10.0, 20.0, 20.0])

    first = moving_block_lcb(values, timestamps, samples=200, seed=7)
    second = moving_block_lcb(values, timestamps, samples=200, seed=7)

    assert first == second
    assert first is not None and first > 0


def test_scope_summary_is_insufficient_without_ten_trades() -> None:
    ledger = pd.DataFrame(
        {
            "net_bps": [20.0] * 9,
            "fills": [2] * 9,
            "signal_timestamp": [
                int(pd.Timestamp(f"2026-01-{day:02d}T00:00:00Z").timestamp() * 1000)
                for day in range(1, 10)
            ],
        }
    )
    summary = summarize_scope(ledger, seed_key="small")

    assert summary["closed_trades"] == 9
    assert summary["status"] == "INSUFFICIENT"
    assert summary["mean_net_bps"] == 20.0


def _write_experiment(path: Path) -> None:
    digest = "a" * 64
    manifest = {
        "schema_version": "exia.experiment_manifest.v1",
        "experiment_id": "exia_test_001",
        "candidate_id": "candidate_a",
        "candidate_spec_sha256": digest,
        "strategy_sha256": digest,
        "taxonomy_sha256": digest,
        "dataset_sha256": digest,
        "freqtrade_revision": "b" * 40,
        "experiment_code_sha256": digest,
        "runtime_config_sha256": digest,
        "historical_config_sha256": digest,
        "timeranges": {"development": "20220101-20240101"},
        "costs": {"base": 0.0004, "stress": 0.0008},
        "command": ["python", "tools/run_exia_experiment_v1.py"],
        "orders_enabled": False,
        "promotion_authority": False,
    }
    metrics = {
        "experiment_id": "exia_test_001",
        "candidate_id": "candidate_a",
        "status": "ENGINEERING_PASS_NO_TRADES",
        "orders_enabled": False,
        "promotion_authority": False,
        "catalog_rows": [
            {
                "experiment_id": "exia_test_001",
                "candidate_id": "candidate_a",
                "candidate_version": "1.0.0",
                "family": "engineering_foundation",
                "dataset_sha256": digest,
                "taxonomy_sha256": digest,
                "split": "development",
                "cost_scenario": "base",
                "scope": "state",
                "scope_value": "TREND_UP",
                "closed_trades": 0,
                "fills": 0,
                "mean_net_bps": None,
                "block_lcb_net_bps": None,
                "baseline_differential_lcb_bps": None,
                "max_drawdown": None,
                "status": "INSUFFICIENT",
            }
        ],
    }
    path.mkdir(parents=True)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_catalog_is_rebuilt_from_immutable_experiments(tmp_path: Path) -> None:
    experiments = tmp_path / "experiments"
    output = tmp_path / "catalog"
    _write_experiment(experiments / "exia_test_001")

    first = build_catalog(experiments_root=experiments, output_dir=output)
    second = build_catalog(experiments_root=experiments, output_dir=output)

    assert first["experiment_count"] == second["experiment_count"] == 1
    assert first["metric_rows"] == second["metric_rows"] == 1
    assert (output / "metrics.parquet").is_file()
    assert "candidate_a" in (output / "candidate_by_state.md").read_text(
        encoding="utf-8"
    )
    assert first["orders_enabled"] is False
    assert first["promotion_authority"] is False


def test_experiment_id_is_lowercase_and_machine_safe() -> None:
    payload = json.loads(
        (Path(__file__).resolve().parents[1] / "freqtrade_pilot" / "candidates" / "exia_market_mode_foundation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["candidate_id"].islower()

    digest = "a" * 64
    manifest = {
        "schema_version": "exia.experiment_manifest.v1",
        "experiment_id": "candidate_20260804t153450z_deadbeef",
        "candidate_id": "candidate",
        "candidate_spec_sha256": digest,
        "strategy_sha256": digest,
        "taxonomy_sha256": digest,
        "dataset_sha256": digest,
        "freqtrade_revision": "b" * 40,
        "experiment_code_sha256": digest,
        "runtime_config_sha256": digest,
        "historical_config_sha256": digest,
        "timeranges": {"development": "20220101-20240101"},
        "costs": {"base": 0.0004, "stress": 0.0008},
        "command": ["python", "runner.py"],
        "orders_enabled": False,
        "promotion_authority": False,
    }
    assert ExperimentManifest.from_mapping(manifest).experiment_id.endswith(
        "deadbeef"
    )


def test_generated_exia_catalog_is_fail_closed_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    index_path = root / "Reports" / "Exia" / "catalog" / "index.json"
    if not index_path.exists():
        return
    index = json.loads(index_path.read_text(encoding="utf-8"))

    assert index["experiment_count"] >= 1
    assert index["metric_rows"] >= 1
    assert index["paper_allowed"] is False
    assert index["live_allowed"] is False
    assert index["orders_enabled"] is False
    assert index["promotion_authority"] is False
    assert all(
        item["status"]
        in {
            "ENGINEERING_PASS_NO_TRADES",
            "DEVELOPMENT_SCORED",
            "SCORED_RETROSPECTIVE",
        }
        for item in index["experiments"]
    )
