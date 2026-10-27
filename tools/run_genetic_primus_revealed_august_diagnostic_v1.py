"""Bounded offline diagnostic on already-revealed August 2026 data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exia.genetic_primus.prospective_contract_v1 import file_sha256
from exia.genetic_primus.prospective_evaluator_v1 import evaluate_panel_bounds
from exia.genetic_primus.prospective_manifest_v1 import (
    build_manifest,
    load_verified_snapshot,
    save_manifest_exclusive,
)
from exia.genetic_primus_statistics.hac_adapter_v1 import assess_paired_daily


CONFIG = ROOT / "configs/genetic_primus_revealed_august_diagnostic_v1.json"
REGISTRY = ROOT / "configs/genetic_primus_objective_registry_v6.json"


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError("explicit UTC timestamp required")
    return parsed.astimezone(timezone.utc)


def _origins(config: dict) -> tuple[tuple[int, int], ...]:
    outer = config["outer"]
    start, end = _utc(outer["start_utc"]), _utc(outer["end_exclusive_utc"])
    horizon = timedelta(days=int(outer["deployment_horizon_days"]))
    cadence = timedelta(days=int(outer["retrain_cadence_days"]))
    if horizon != cadence or outer["overlap_allowed"] or (end - start) % cadence:
        raise ValueError("diagnostic origins must be complete and non-overlapping")
    result = tuple(
        (int((start + i * cadence).timestamp() * 1000), int((start + (i + 1) * horizon).timestamp() * 1000))
        for i in range(int((end - start) / cadence))
    )
    if len(result) != int(outer["origin_count"]) or len(result) != 10:
        raise ValueError("exactly ten registered August origins required")
    return result


def _load() -> tuple[dict, Path]:
    if Path(sys.executable).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve():
        raise ValueError("only project .venv Python is authorized")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if config["schema_version"] != "exia.genetic_primus.revealed_3d_diagnostic/1":
        raise ValueError("wrong diagnostic schema")
    if config.get("data_already_revealed") is not True or config.get("selection_use") != (
        "engineering_diagnostic_only_not_validation_not_tuning_not_promotion"
    ):
        raise ValueError("revealed-data limitations are not explicit")
    if any(config["safety"].values()) or not all(config["measurement"].values()):
        raise ValueError("diagnostic safety/measurement lock violated")
    if file_sha256(ROOT / config["source_prospective_contract"]) != config["source_prospective_contract_sha256"]:
        raise ValueError("source prospective contract changed")
    if [item["candidate_id"] for item in config["candidates"]] != [
        "NoTrade", "EMA_TrendConsensus", "GA_overlay_f7a4ae11"
    ] or config["candidates"][2]["genome"] != [0, 0, 1, 1, -1]:
        raise ValueError("candidate set changed")
    _origins(config)
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    entries = [item for item in registry["objectives"] if item["objective_id"] == config["objective_id"]]
    if len(entries) != 1 or entries[0]["config_sha256"] != file_sha256(CONFIG):
        raise ValueError("diagnostic is not pinned by registry v6")
    if file_sha256(ROOT / registry["previous_registry_path"]) != registry["previous_registry_sha256"]:
        raise ValueError("registry ancestry changed")
    output = (ROOT / config["result_path"]).resolve()
    if not output.is_relative_to(ROOT / "Reports/Exia/Genetic_Primus"):
        raise ValueError("invalid diagnostic result root")
    return config, output


def _save_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def prepare(_: argparse.Namespace) -> None:
    config, output = _load()
    paths = []
    for entry in config["data_commitment"]["files"]:
        path = ROOT / entry["path"]
        if file_sha256(path) != entry["sha256"]:
            raise ValueError("registered data file SHA changed")
        paths.append(path)
    manifest = build_manifest(
        paths, root=ROOT, contract_path=CONFIG, contract=config,
        created_at=datetime.now(timezone.utc),
    )
    target = output / "data_manifest.json"
    save_manifest_exclusive(target, manifest)
    _save_new(output / "pre_run_receipt.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": file_sha256(CONFIG),
        "registry_sha256": file_sha256(REGISTRY),
        "manifest_sha256": file_sha256(target),
        "snapshot_sha256": manifest["snapshot_sha256"],
        "data_already_revealed": True,
        "selection_use": config["selection_use"],
        "safety": config["safety"],
    })
    print(json.dumps({"status": "REVEALED_DIAGNOSTIC_PREPARED_NO_EVALUATION",
                      "manifest_sha256": file_sha256(target),
                      "snapshot_sha256": manifest["snapshot_sha256"]}, separators=(",", ":")))


def evaluate(args: argparse.Namespace) -> None:
    config, output = _load()
    origins = _origins(config)
    if args.origin not in range(len(origins)):
        raise ValueError("unregistered origin")
    manifest_path = output / "data_manifest.json"
    manifest, panel = load_verified_snapshot(
        manifest_path, root=ROOT, contract_path=CONFIG, contract=config,
    )
    start_ms, end_ms = origins[args.origin]
    if manifest["metadata"]["coverage_start"] > int(_utc(config["training"]["initial_history_start_utc"]).timestamp() * 1000):
        raise ValueError("training history is incomplete")
    if manifest["metadata"]["coverage_end_exclusive"] < end_ms:
        raise ValueError("origin data is incomplete")
    target = output / f"origin_{args.origin + 1:02d}.json"
    started = output / f"origin_{args.origin + 1:02d}_started.json"
    if target.exists() or started.exists():
        raise FileExistsError("origin was already attempted")
    _save_new(started, {"created_at_utc": datetime.now(timezone.utc).isoformat(),
                        "origin_index": args.origin, "config_sha256": file_sha256(CONFIG),
                        "manifest_sha256": file_sha256(manifest_path), "safety": config["safety"]})
    result = evaluate_panel_bounds(
        panel, config, manifest, args.origin, start_ms=start_ms, end_ms=end_ms,
    )
    result.update({"data_already_revealed": True, "promotion_authority": False,
                   "config_sha256": file_sha256(CONFIG), "manifest_sha256": file_sha256(manifest_path)})
    _save_new(target, result)
    normal = result["evaluations"]["normal"]["candidates"]
    print(json.dumps({"origin": args.origin + 1, "status": "DIAGNOSTIC_ONLY",
                      "ema_return": normal["EMA_TrendConsensus"]["metrics"]["net_return"],
                      "overlay_return": normal["GA_overlay_f7a4ae11"]["metrics"]["net_return"],
                      "result_sha256": file_sha256(target)}, separators=(",", ":")))


def _compound(values: np.ndarray) -> float:
    return float(np.prod(1.0 + values) - 1.0)


def _max_drawdown(values: np.ndarray) -> float:
    equity = np.concatenate(([1.0], np.cumprod(1.0 + values)))
    return float(np.max(1.0 - equity / np.maximum.accumulate(equity)))


def finalize(_: argparse.Namespace) -> None:
    config, output = _load()
    origins = _origins(config)
    results = [json.loads((output / f"origin_{i + 1:02d}.json").read_text(encoding="utf-8")) for i in range(10)]
    if any(item["origin_start"] != origins[i][0] or item["origin_end_exclusive"] != origins[i][1]
           for i, item in enumerate(results)):
        raise ValueError("saved origin bounds changed")
    timestamps = np.arange(origins[0][0] + 86_400_000, origins[-1][1] + 1, 86_400_000, dtype=np.int64)
    summary: dict[str, object] = {"schema_version": "exia.genetic_primus.revealed_3d_summary/1",
                                  "data_already_revealed": True, "promotion_authority": False,
                                  "origins": 10, "daily_observations": 30, "costs": {}}
    for cost_label in ("normal", "stress"):
        daily = {candidate: np.concatenate([
            np.asarray(item["evaluations"][cost_label]["candidates"][candidate]["daily_returns"], dtype=float)
            for item in results
        ]) for candidate in ("NoTrade", "EMA_TrendConsensus", "GA_overlay_f7a4ae11")}
        trades = {candidate: sum(
            int(item["evaluations"][cost_label]["candidates"][candidate]["metrics"]["trades"])
            for item in results
        ) for candidate in daily}
        positive_vs_ema = np.mean([
            sum(item["evaluations"][cost_label]["paired_overlay_minus_ema"]) > 0 for item in results
        ])
        positive_vs_no_trade = np.mean([
            sum(item["evaluations"][cost_label]["paired_overlay_minus_no_trade"]) > 0 for item in results
        ])
        summary["costs"][cost_label] = {
            "candidates": {candidate: {"compounded_return": _compound(values),
                                         "max_drawdown": _max_drawdown(values),
                                         "trades": trades[candidate]} for candidate, values in daily.items()},
            "overlay_minus_ema_hac": assess_paired_daily(
                daily["GA_overlay_f7a4ae11"], daily["EMA_TrendConsensus"], timestamps
            ).mapping(),
            "overlay_minus_no_trade_hac": assess_paired_daily(
                daily["GA_overlay_f7a4ae11"], daily["NoTrade"], timestamps
            ).mapping(),
            "positive_origin_fraction_vs_ema": float(positive_vs_ema),
            "positive_origin_fraction_vs_no_trade": float(positive_vs_no_trade),
        }
    summary["decision"] = "REVEALED_ENGINEERING_DIAGNOSTIC_ONLY_NO_TUNING_NO_PROMOTION"
    target = output / "summary.json"
    _save_new(target, summary)
    print(json.dumps({"status": summary["decision"], "summary_sha256": file_sha256(target)}, separators=(",", ":")))


def complete(_: argparse.Namespace) -> None:
    config, output = _load()
    manifest_path = output / "data_manifest.json"
    summary_path = output / "summary.json"
    artifacts = {
        "data_manifest.json": file_sha256(manifest_path),
        "pre_run_receipt.json": file_sha256(output / "pre_run_receipt.json"),
        "summary.json": file_sha256(summary_path),
    }
    for index in range(10):
        for suffix in ("_started.json", ".json"):
            name = f"origin_{index + 1:02d}{suffix}"
            path = output / name
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("config_sha256") != file_sha256(CONFIG):
                raise ValueError("origin artifact contract mismatch")
            if any(payload.get("safety", {}).values()):
                raise ValueError("origin artifact safety mismatch")
            if suffix == ".json" and (
                payload.get("search_evaluations") != 0
                or payload.get("data_already_revealed") is not True
                or payload.get("promotion_authority") is not False
            ):
                raise ValueError("origin diagnostic status mismatch")
            artifacts[name] = file_sha256(path)
    receipt = {
        "schema_version": "exia.genetic_primus.revealed_3d_completion/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "REVEALED_ENGINEERING_DIAGNOSTIC_ONLY_NO_TUNING_NO_PROMOTION",
        "config_sha256": file_sha256(CONFIG),
        "registry_sha256": file_sha256(REGISTRY),
        "manifest_sha256": file_sha256(manifest_path),
        "artifact_sha256": dict(sorted(artifacts.items())),
        "artifact_count": len(artifacts),
        "safety": config["safety"],
    }
    target = output / "completion.json"
    _save_new(target, receipt)
    print(json.dumps({"status": "COMPLETE_DIAGNOSTIC_ONLY", "artifact_count": len(artifacts),
                      "completion_sha256": file_sha256(target)}, separators=(",", ":")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare")
    command.set_defaults(handler=prepare)
    command = commands.add_parser("origin")
    command.add_argument("--origin", type=int, required=True)
    command.set_defaults(handler=evaluate)
    command = commands.add_parser("finalize")
    command.set_defaults(handler=finalize)
    command = commands.add_parser("complete")
    command.set_defaults(handler=complete)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
