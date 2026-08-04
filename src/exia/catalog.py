from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import ExperimentManifest


CATALOG_COLUMNS = (
    "experiment_id",
    "candidate_id",
    "candidate_version",
    "family",
    "dataset_sha256",
    "taxonomy_sha256",
    "split",
    "cost_scenario",
    "scope",
    "scope_value",
    "closed_trades",
    "fills",
    "mean_net_bps",
    "block_lcb_net_bps",
    "baseline_differential_lcb_bps",
    "max_drawdown",
    "status",
)


def _load_experiment(path: Path) -> tuple[ExperimentManifest, dict[str, Any]]:
    manifest_path = path / "manifest.json"
    metrics_path = path / "metrics.json"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise ValueError(f"incomplete experiment folder: {path}")
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    manifest = ExperimentManifest.from_mapping(manifest_payload)
    if metrics.get("experiment_id") != manifest.experiment_id:
        raise ValueError(f"experiment id mismatch: {path}")
    if metrics.get("candidate_id") != manifest.candidate_id:
        raise ValueError(f"candidate id mismatch: {path}")
    if metrics.get("orders_enabled") is not False or metrics.get(
        "promotion_authority"
    ) is not False:
        raise ValueError(f"authoritative safety flag in experiment: {path}")
    return manifest, metrics


def _render_state_table(frame: pd.DataFrame) -> str:
    lines = [
        "# Exia candidate by market state",
        "",
        "Offline research catalog only. It cannot authorize paper/live.",
        "",
        "| Candidate | Version | Split | Cost | State | Trades | Mean bps | Block LCB | Status |",
        "|---|---|---|---|---|---:|---:|---:|---|",
    ]
    states = frame.loc[frame["scope"] == "state"].copy()
    states = states.sort_values(
        ["candidate_id", "candidate_version", "split", "cost_scenario", "scope_value"]
    )
    for row in states.itertuples(index=False):
        mean = "n/a" if pd.isna(row.mean_net_bps) else f"{row.mean_net_bps:.2f}"
        lcb = (
            "n/a"
            if pd.isna(row.block_lcb_net_bps)
            else f"{row.block_lcb_net_bps:.2f}"
        )
        lines.append(
            f"| {row.candidate_id} | {row.candidate_version} | {row.split} | "
            f"{row.cost_scenario} | {row.scope_value} | {row.closed_trades} | "
            f"{mean} | {lcb} | {row.status} |"
        )
    return "\n".join(lines) + "\n"


def build_catalog(*, experiments_root: Path, output_dir: Path) -> dict[str, Any]:
    experiment_summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    if experiments_root.is_dir():
        for path in sorted(item for item in experiments_root.iterdir() if item.is_dir()):
            manifest, metrics = _load_experiment(path)
            experiment_summaries.append(
                {
                    "experiment_id": manifest.experiment_id,
                    "candidate_id": manifest.candidate_id,
                    "status": metrics["status"],
                    "dataset_sha256": manifest.dataset_sha256,
                    "taxonomy_sha256": manifest.taxonomy_sha256,
                    "path": path.name,
                }
            )
            for row in metrics["catalog_rows"]:
                missing = [column for column in CATALOG_COLUMNS if column not in row]
                if missing:
                    raise ValueError(
                        f"catalog row missing columns in {path}: {missing}"
                    )
                rows.append({column: row[column] for column in CATALOG_COLUMNS})

    frame = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "metrics.parquet", index=False, compression="zstd")
    report = {
        "schema_version": "exia.research_catalog.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "experiments": experiment_summaries,
        "experiment_count": len(experiment_summaries),
        "metric_rows": len(frame),
        "comparison_contract": (
            "compare only equal dataset_sha256, taxonomy_sha256, split, and "
            "cost_scenario"
        ),
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    (output_dir / "index.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "candidate_by_state.md").write_text(
        _render_state_table(frame),
        encoding="utf-8",
    )
    return report
