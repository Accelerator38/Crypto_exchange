from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from .dataset import load_feature_tape
from .simulator import CostModel, SimulationResult, simulate_targets
from .strategies import build_signal


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def validate_registry(registry: Mapping[str, object]) -> None:
    if registry.get("orders_enabled") is not False:
        raise ValueError("registry orders_enabled must be false")
    if registry.get("promotion_authority") is not False:
        raise ValueError("registry promotion_authority must be false")
    strategies = registry.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        raise ValueError("registry strategies must be a nonempty list")
    budget = registry.get("trial_budget", {})
    maximum = int(budget.get("maximum_strategies", 10))
    maximum_family = int(budget.get("maximum_variants_per_family", 3))
    if len(strategies) > maximum or maximum > 10:
        raise ValueError("strategy budget exceeds 10")
    ids = [str(row["id"]) for row in strategies]
    if len(ids) != len(set(ids)):
        raise ValueError("strategy ids must be unique")
    family_counts = Counter(str(row["family"]) for row in strategies)
    if family_counts and max(family_counts.values()) > maximum_family:
        raise ValueError("family variant budget exceeded")
    if int(budget.get("registered_trials", -1)) != len(strategies):
        raise ValueError("registered_trials does not match strategy count")


def _cost_model(row: Mapping[str, object]) -> CostModel:
    return CostModel(
        fee_bps_per_fill=float(row["fee_bps_per_fill"]),
        slippage_bps_per_fill=float(row["slippage_bps_per_fill"]),
    )


def _screening_pass(
    *,
    strategy_id: str,
    results: Mapping[str, Mapping[str, Mapping[str, object]]],
    gate: Mapping[str, object],
    long_only_oos_return: float,
) -> tuple[bool, list[str]]:
    if strategy_id in {"flat_v1", "long_only_v1"}:
        return False, ["baseline_not_candidate"]
    reasons: list[str] = []
    minimum = int(gate["min_closed_trades_per_validation_split"])
    for split in ("validation", "oos"):
        base = results[split]["base"]
        stress = results[split]["stress"]
        if int(base["closed_trades"]) < minimum:
            reasons.append(f"{split}:closed_trades_below_{minimum}")
        if float(base["mean_net_bps"]) <= 0:
            reasons.append(f"{split}:nonpositive_mean")
        if base["lcb_95_net_bps"] is None or float(base["lcb_95_net_bps"]) <= 0:
            reasons.append(f"{split}:nonpositive_lcb")
        adjusted = base["multiple_testing_lcb_95_net_bps"]
        if adjusted is None or float(adjusted) <= 0:
            reasons.append(f"{split}:nonpositive_adjusted_lcb")
        if float(stress["mean_net_bps"]) <= 0:
            reasons.append(f"{split}:nonpositive_stress_mean")
        if float(base["max_drawdown"]) > float(gate["max_drawdown"]):
            reasons.append(f"{split}:drawdown")
    candidate_oos = float(results["oos"]["base"]["total_net_return"])
    if candidate_oos <= long_only_oos_return:
        reasons.append("oos:does_not_beat_long_only")
    return not reasons, reasons


def _write_markdown(report: Mapping[str, object], path: Path) -> None:
    lines = [
        "# Simple research reset batch",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This is screening-only evidence. It cannot authorize paper or live trading.",
        "",
        "| Strategy | OOS trades | OOS net bps | OOS LCB | Stress net bps | Max DD | Pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["strategies"]:
        oos = row["splits"]["oos"]
        base = oos["base"]
        stress = oos["stress"]
        lcb = base["lcb_95_net_bps"]
        lcb_text = "n/a" if lcb is None else f"{float(lcb):.3f}"
        lines.append(
            "| {id} | {trades} | {mean:.3f} | {lcb} | {stress:.3f} | "
            "{dd:.3%} | {passed} |".format(
                id=row["id"],
                trades=base["closed_trades"],
                mean=float(base["mean_net_bps"]),
                lcb=lcb_text,
                stress=float(stress["mean_net_bps"]),
                dd=float(base["max_drawdown"]),
                passed="PASS" if row["screening_pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            f"Survivors: `{', '.join(report['survivors']) or 'none'}`",
            "",
            "`orders_enabled=false`; `promotion_authority=false`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_batch(
    *,
    registry_path: Path,
    project_root: Path,
    output_dir: Path,
) -> dict[str, object]:
    batch_started = time.monotonic()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    validate_registry(registry)
    tape_path = project_root / str(registry["dataset"]["feature_tape"])
    manifest_path = tape_path.with_suffix(".manifest.json")
    tape_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tape_manifest.get("orders_enabled") is not False:
        raise ValueError("feature tape orders_enabled must be false")
    expected_dataset_sha = str(registry["dataset"]["source_manifest_sha256"])
    actual_dataset_sha = str(
        (tape_manifest.get("source_integrity") or {}).get("dataset_sha256")
    )
    if actual_dataset_sha != expected_dataset_sha:
        raise ValueError(
            "feature tape source dataset SHA mismatch: "
            f"expected {expected_dataset_sha}, got {actual_dataset_sha}"
        )
    frame = load_feature_tape(tape_path)
    strategies = registry["strategies"]
    trial_count = len(strategies)
    costs = {
        name: _cost_model(row) for name, row in registry["costs"].items()
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_dir = output_dir / "ledgers"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    decision_tape = frame.loc[:, ["timestamp", "symbol"]].copy()
    rows: list[dict[str, object]] = []
    for strategy in strategies:
        strategy_started = time.monotonic()
        strategy_id = str(strategy["id"])
        signal = build_signal(frame, strategy)
        decision_tape[strategy_id] = signal
        split_results: dict[str, dict[str, dict[str, object]]] = {}
        for split in registry["splits"]:
            split_id = str(split["id"])
            split_results[split_id] = {}
            for cost_name, cost_model in costs.items():
                result: SimulationResult = simulate_targets(
                    frame,
                    signal,
                    start_timestamp=_timestamp(str(split["start"])),
                    end_timestamp=_timestamp(str(split["end"])),
                    costs=cost_model,
                    trial_count=trial_count,
                )
                split_results[split_id][cost_name] = result.metrics
                result.ledger.to_parquet(
                    ledger_dir / f"{strategy_id}_{split_id}_{cost_name}.parquet",
                    index=False,
                    compression="zstd",
                )
        rows.append(
            {
                "id": strategy_id,
                "family": strategy["family"],
                "kind": strategy["kind"],
                "params": strategy["params"],
                "splits": split_results,
                "elapsed_sec": round(time.monotonic() - strategy_started, 3),
            }
        )
    decision_path = tape_path.parent / "decision_tape.parquet"
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_tape.to_parquet(decision_path, index=False, compression="zstd")
    long_only = next(row for row in rows if row["id"] == "long_only_v1")
    long_only_return = float(
        long_only["splits"]["oos"]["base"]["total_net_return"]
    )
    survivors: list[str] = []
    for row in rows:
        passed, reasons = _screening_pass(
            strategy_id=str(row["id"]),
            results=row["splits"],
            gate=registry["screening_gate"],
            long_only_oos_return=long_only_return,
        )
        row["screening_pass"] = passed
        row["screening_reasons"] = reasons
        if passed:
            survivors.append(str(row["id"]))
    report: dict[str, object] = {
        "schema_version": "simple_research.batch_report.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_path.relative_to(project_root)).replace(
            "\\", "/"
        ),
        "registry_sha256": _sha256(registry_path),
        "feature_tape_manifest_sha256": _sha256(manifest_path),
        "decision_tape_path": str(decision_path.relative_to(project_root)).replace(
            "\\", "/"
        ),
        "decision_tape_sha256": _sha256(decision_path),
        "strategy_count": len(rows),
        "trial_count_for_adjustment": trial_count,
        "elapsed_sec": round(time.monotonic() - batch_started, 3),
        "strategies": rows,
        "survivors": survivors,
        "screening_complete": True,
        "paper_or_live_authorized": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    json_path = output_dir / "batch_report.json"
    markdown_path = output_dir / "batch_report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _write_markdown(report, markdown_path)
    return report
