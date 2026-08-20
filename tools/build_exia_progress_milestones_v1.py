from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "exia_progress_milestones_v1.json"
DEFAULT_OUTPUT = ROOT / "Reports" / "Exia" / "build_progress_tracker_v1"
STATE_MAP = {
    "TREND_UP": "trend_up_mean_bps",
    "bullish": "trend_up_mean_bps",
    "TREND_DOWN": "trend_down_mean_bps",
    "bearish": "trend_down_mean_bps",
    "RANGE": "range_neutral_mean_bps",
    "neutral": "range_neutral_mean_bps",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _base_row(item: dict[str, Any], report_path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "milestone_id": item["milestone_id"],
        "short_label": item["short_label"],
        "build_datetime": report["generated_at"],
        "timeframe": item["timeframe"],
        "window": item["window"],
        "candidate": item["candidate_id"] if "candidate_id" in item else item["id_value"],
        "overall_mean_bps": None,
        "trend_up_mean_bps": None,
        "trend_down_mean_bps": None,
        "range_neutral_mean_bps": None,
        "overall_lcb_bps": None,
        "familywise_lcb_bps": None,
        "median_bps": None,
        "observations": 0,
        "gate_pass": False,
        "status": report.get("verdict", report.get("status", "UNKNOWN")),
        "selection_criterion": item["selection_criterion"],
        "main_difference": item["main_difference"],
        "decision": item["decision"],
        "source_report": str(report_path.relative_to(ROOT)),
        "source_report_sha256": _sha256(report_path),
    }


def _legacy(item: dict[str, Any]) -> dict[str, Any]:
    report_path = ROOT / item["report"]
    report = _load_json(report_path)
    candidate = next(
        row for row in report["candidates"] if row["candidate_id"] == item["candidate_id"]
    )
    row = _base_row(item, report_path, report)
    row.update(
        {
            "overall_mean_bps": candidate["mean_net_bps"],
            "overall_lcb_bps": candidate["block_lcb_net_bps"],
            "observations": candidate["closed_trades"],
            "gate_pass": bool(candidate.get("passes_alpha_gate", False)),
        }
    )
    for state in candidate["market_state_results"]:
        if state["cost_scenario"] != "stress" or state["market_state"] not in STATE_MAP:
            continue
        row[STATE_MAP[state["market_state"]]] = state["mean_net_bps"]
    return row


def _ledger(item: dict[str, Any]) -> dict[str, Any]:
    report_path = ROOT / item["report"]
    report = _load_json(report_path)
    metrics_path = ROOT / item["metrics"]
    ledger_path = ROOT / item["ledger"]
    metrics = pd.read_parquet(metrics_path)
    selected = metrics.loc[
        metrics[item["id_column"]].astype(str).eq(str(item["id_value"]))
        & metrics["window"].astype(str).eq(str(item["window"]))
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one metric row for {item['milestone_id']}")
    metric = selected.iloc[0]
    ledger = pd.read_parquet(ledger_path)
    selected_ledger = ledger.loc[
        ledger[item["id_column"]].astype(str).eq(str(item["id_value"]))
        & ledger["window"].astype(str).eq(str(item["window"]))
    ]
    row = _base_row(item, report_path, report)
    row.update(
        {
            "overall_mean_bps": metric["mean_stress_net_bps"],
            "overall_lcb_bps": metric["stress_lcb_bps"],
            "familywise_lcb_bps": metric.get("familywise_stress_lcb_bps"),
            "median_bps": metric.get("median_stress_net_bps"),
            "observations": int(metric[item["count_column"]]),
            "gate_pass": bool(metric.get("gate_pass", False)),
            "source_metrics": str(metrics_path.relative_to(ROOT)),
            "source_metrics_sha256": _sha256(metrics_path),
            "source_ledger": str(ledger_path.relative_to(ROOT)),
            "source_ledger_sha256": _sha256(ledger_path),
        }
    )
    if "market_regime" in selected_ledger:
        grouped = selected_ledger.groupby("market_regime", sort=True)["stress_net_bps"].mean()
        for state, value in grouped.items():
            if str(state) in STATE_MAP:
                row[STATE_MAP[str(state)]] = float(value)
    return row


def build(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    if config.get("schema_version") != "exia.progress_milestones.v1":
        raise ValueError("unexpected progress milestone schema")
    if set(config.get("safety", {}).values()) != {False}:
        raise ValueError("progress tracker cannot grant trading authority")
    rows = []
    for item in config["milestones"]:
        if item["source_type"] == "legacy_family":
            rows.append(_legacy(item))
        elif item["source_type"] in {"trade_ledger", "forward_labels"}:
            rows.append(_ledger(item))
        else:
            raise ValueError(f"unknown milestone source type: {item['source_type']}")
    payload = {
        "schema_version": "exia.progress_tracker_snapshot.v1",
        "tracker_id": config["tracker_id"],
        "metric": config["metric"],
        "config_sha256": _sha256(config_path),
        "milestone_count": len(rows),
        "regime_normalization": {
            "trend_up": ["TREND_UP", "bullish"],
            "trend_down": ["TREND_DOWN", "bearish"],
            "range_neutral": ["RANGE", "neutral"],
        },
        "milestones": rows,
        "safety": dict(config["safety"]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "milestones.json"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    columns = list(rows[0])
    for row in rows[1:]:
        columns.extend(key for key in row if key not in columns)
    required_tail = ["source_report", "source_report_sha256"]
    columns = [key for key in columns if key not in required_tail] + required_tail
    tsv_path = output_dir / "milestones.tsv"
    with tsv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"milestones": len(rows), "output": str(json_path)}, ensure_ascii=False))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Exia progress milestone snapshot")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
