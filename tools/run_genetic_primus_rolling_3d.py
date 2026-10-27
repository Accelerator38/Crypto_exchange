from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.contracts import sha256_file  # noqa: E402
from exia.genetic_primus.contracts_v4 import load_v4_config  # noqa: E402
from exia.genetic_primus.data import load_panel  # noqa: E402
from exia.genetic_primus.features import build_feature_frame  # noqa: E402
from exia.genetic_primus.labels import build_forward_labels  # noqa: E402
from exia.genetic_primus.measurement import load_measurement_protocol  # noqa: E402
from exia.genetic_primus.objective_registry import (  # noqa: E402
    load_immutable_objective_registry,
)
from exia.genetic_primus.policy_v4 import V4NetworkShape  # noqa: E402
from exia.genetic_primus.promotion_v4 import (  # noqa: E402
    evaluate_outer_promotion_gate,
    load_promotion_protocol,
)
from exia.genetic_primus.rolling import (  # noqa: E402
    RollingProtocol,
    evaluate_rolling,
    load_rolling_protocol,
    summarize_outer_deployment_evidence,
    validate_rolling_output_directory,
)
from exia.genetic_primus.rolling_v4 import bounded_v4_origin_trainer  # noqa: E402


DEFAULT_V4_CONFIG = ROOT / "configs/genetic_primus_v4.json"
DEFAULT_MEASUREMENT = ROOT / "configs/genetic_primus_measurement_3d_v1.json"
DEFAULT_ROLLING = ROOT / "configs/genetic_primus_rolling_3d_v1.json"
DEFAULT_PROMOTION = ROOT / "configs/genetic_primus_promotion_gate_v1.json"


def _utc_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("engineering timestamps must be timezone-aware")
    return parsed


def build_revealed_engineering_override(
    base: RollingProtocol, *, validation_start: str, validation_end: str
) -> RollingProtocol:
    if base.payload.get("protocol_id") != "genetic_primus_overlay_3d_v6":
        raise ValueError("revealed engineering override is restricted to V6")
    if base.payload.get("sealed_interval", {}).get("status") != "UNOPENED":
        raise ValueError("base V6 sealed interval must remain unopened")
    start = _utc_datetime(validation_start)
    end = _utc_datetime(validation_end)
    sealed_start = _utc_datetime(str(base.payload["validation_start"]))
    if not start < end <= sealed_start:
        raise ValueError("engineering interval must end before the sealed interval")

    payload = json.loads(json.dumps(base.payload))
    payload["validation_start"] = validation_start
    payload["validation_end"] = validation_end
    payload["sealed_interval"] = {
        "status": "REVEALED_ENGINEERING",
        "parameter_changes_after_open_forbidden": True,
        "minimum_outer_origins": 1,
    }
    payload.pop("objective_registry", None)
    payload["engineering_override"] = {
        "non_evidentiary": True,
        "base_protocol_sha256": base.sha256,
        "sealed_interval_touched": False,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return RollingProtocol(
        source_path=base.source_path,
        payload=payload,
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def _artifact_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(
    *,
    v4_config_path: Path,
    measurement_path: Path,
    rolling_path: Path,
    promotion_path: Path,
    output_dir: Path,
    max_origins: int,
    rolling_override=None,
    resume_partial: bool = False,
) -> dict[str, object]:
    validate_rolling_output_directory(output_dir, resume_partial=resume_partial)
    if max_origins <= 0:
        raise ValueError("max_origins must be positive")
    config = load_v4_config(v4_config_path, root=ROOT)
    measurement = load_measurement_protocol(measurement_path)
    rolling = rolling_override if rolling_override is not None else load_rolling_protocol(rolling_path)
    objective_registry = None
    global_multiple_testing = None
    registry_contract = rolling.payload.get("objective_registry")
    if registry_contract is not None:
        objective_registry = load_immutable_objective_registry(
            ROOT / str(registry_contract["registry_path"]), workspace=ROOT
        )
        if str(registry_contract["objective_id"]) not in objective_registry["objective_ids"]:
            raise ValueError("rolling objective is absent from immutable registry")
        multiple_testing_path = ROOT / str(registry_contract["evidence_path"])
        if multiple_testing_path.is_file():
            global_multiple_testing = json.loads(multiple_testing_path.read_text(encoding="utf-8"))
            if global_multiple_testing.get("objective_registry_sha256") != objective_registry[
                "registry_sha256"
            ]:
                raise ValueError("global multiple-testing evidence registry mismatch")
    promotion = load_promotion_protocol(promotion_path)
    training = rolling.payload["genetic_training"]
    architecture = next(
        item for item in config.architectures if item["id"] == training["architecture_id"]
    )
    shape = V4NetworkShape(
        inputs=int(architecture["inputs"]),
        hidden=tuple(int(value) for value in architecture["hidden"]),
        outputs=(
            1
            if training.get("output_mode") == "binary_allow"
            else int(architecture["outputs"])
        ),
        maximum_active_features=(
            int(training["maximum_active_features"])
            if "maximum_active_features" in training
            else None
        ),
    )
    panel = load_panel(config, root=ROOT)
    features = build_feature_frame(panel)
    labels = build_forward_labels(
        panel,
        horizons_bars=tuple(config.payload["labels"]["horizons_bars"]),
        round_trip_cost_bps=float(config.costs["base_round_trip"]),
        minimum_edge_bps=float(config.payload["labels"]["minimum_edge_bps"]),
        minimum_votes=int(config.payload["labels"]["minimum_votes"]),
    )
    trainer = bounded_v4_origin_trainer(
        rolling_protocol=rolling,
        measurement_protocol=measurement,
        shape=shape,
        fitness_config=config.fitness,
        round_trip_cost_bps=float(config.costs["base_round_trip"]),
        promotion_protocol=promotion,
    )
    output_dir.mkdir(parents=True, exist_ok=resume_partial)
    result = evaluate_rolling(
        features=features,
        labels=labels,
        protocol=rolling,
        trainer=trainer,
        round_trip_cost_bps=float(config.costs["base_round_trip"]),
        stress_round_trip_cost_bps=(
            float(rolling.payload["promotion_evidence"]["stress_round_trip_cost_bps"])
            if "stress_round_trip_cost_bps"
            in rolling.payload.get("promotion_evidence", {})
            else None
        ),
        output_dir=output_dir,
        max_origins=max_origins,
        resume_partial=resume_partial,
    )
    protocol_snapshot = output_dir / "rolling_protocol_snapshot.json"
    protocol_snapshot.write_text(
        json.dumps(rolling.payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    artifacts = {
        "origins": output_dir / "origins.parquet",
        "models": output_dir / "models.parquet",
        "decisions": output_dir / "decisions.parquet",
        "trades": output_dir / "trades.parquet",
        "portfolio": output_dir / "portfolio.parquet",
        "benchmarks": output_dir / "benchmarks.parquet",
        "benchmark_portfolio": output_dir / "benchmark_portfolio.parquet",
    }
    frames = {
        "origins": result.origins,
        "models": result.models,
        "decisions": result.decisions,
        "trades": result.trades,
        "portfolio": result.portfolio,
        "benchmarks": result.benchmarks,
        "benchmark_portfolio": result.benchmark_portfolio,
    }
    for name, path in artifacts.items():
        frames[name].to_parquet(path, index=False)
    artifact_sha256 = {name: _artifact_sha(path) for name, path in artifacts.items()}
    artifact_sha256["rolling_protocol_snapshot"] = _artifact_sha(protocol_snapshot)
    outer_evidence = None
    outer_promotion_gate = None
    evidence_config = rolling.payload.get("promotion_evidence")
    if evidence_config is not None:
        evidence_config = dict(evidence_config)
        if registry_contract is not None:
            evidence_config.update(
                {
                    "dsr_required": bool(registry_contract["dsr_required"]),
                    "pbo_required": bool(registry_contract["pbo_required"]),
                    "minimum_dsr_probability": float(
                        registry_contract["minimum_dsr_probability"]
                    ),
                    "maximum_pbo": float(registry_contract["maximum_pbo"]),
                    "global_multiple_testing": global_multiple_testing,
                }
            )
        outer_evidence = summarize_outer_deployment_evidence(
            result.benchmarks,
            minimum_completed_origins=int(
                evidence_config["minimum_completed_outer_origins"]
            ),
            minimum_positive_fraction=float(
                evidence_config["minimum_positive_outer_slice_fraction"]
            ),
        )
        evidence_path = output_dir / "outer_deployment_evidence.json"
        evidence_path.write_text(
            json.dumps(outer_evidence, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        artifact_sha256["outer_deployment_evidence"] = _artifact_sha(evidence_path)
        if "outer_bootstrap_samples" in evidence_config:
            decision = evaluate_outer_promotion_gate(
                benchmarks=result.benchmarks,
                origins=result.models,
                config=evidence_config,
                benchmark_portfolio=result.benchmark_portfolio,
            )
            outer_promotion_gate = dict(decision.metrics)
            gate_path = output_dir / "outer_promotion_gate.json"
            gate_path.write_text(
                json.dumps(outer_promotion_gate, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            artifact_sha256["outer_promotion_gate"] = _artifact_sha(gate_path)
    manifest = {
        "schema_version": "exia.genetic_primus.rolling_3d_run/1",
        "status": "OFFLINE_ROLLING_3D_ENGINEERING",
        "completed_origins": int(len(result.origins)),
        "decisions": int(len(result.decisions)),
        "trades": int(len(result.trades)),
        "benchmark_rows": int(len(result.benchmarks)),
        "v4_config_sha256": sha256_file(v4_config_path),
        "measurement_protocol_sha256": measurement.sha256,
        "rolling_protocol_sha256": rolling.sha256,
        "objective_registry_sha256": (
            objective_registry["registry_sha256"] if objective_registry else None
        ),
        "objective_registry_count": (
            objective_registry["objective_count"] if objective_registry else None
        ),
        "promotion_protocol_sha256": promotion.sha256,
        "experiment_matrix_sha256": rolling.payload.get("experiment_matrix_sha256"),
        "experiment_cell_id": rolling.payload.get("experiment_cell_id"),
        "dataset_sha256": config.dataset["dataset_sha256"],
        "dataset_manifest_file_sha256": sha256_file(ROOT / str(config.dataset["manifest"])),
        "artifact_sha256": artifact_sha256,
        "outer_deployment_evidence": outer_evidence,
        "outer_promotion_gate": outer_promotion_gate,
        "position_boundary_policy": rolling.payload["position_boundary_policy"],
        "credentials_allowed": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline three-day Genetic Primus rolling evaluator")
    parser.add_argument("--v4-config", type=Path, default=DEFAULT_V4_CONFIG)
    parser.add_argument("--measurement-config", type=Path, default=DEFAULT_MEASUREMENT)
    parser.add_argument("--rolling-config", type=Path, default=DEFAULT_ROLLING)
    parser.add_argument("--promotion-config", type=Path, default=DEFAULT_PROMOTION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-origins", type=int, default=1)
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--engineering-validation-start")
    parser.add_argument("--engineering-validation-end")
    args = parser.parse_args()
    engineering_args = (
        args.engineering_validation_start is not None,
        args.engineering_validation_end is not None,
    )
    if any(engineering_args) and not all(engineering_args):
        parser.error("both engineering validation timestamps are required")
    rolling_override = None
    if all(engineering_args):
        if args.max_origins != 1:
            parser.error("engineering override requires --max-origins 1")
        base = load_rolling_protocol(args.rolling_config.resolve())
        rolling_override = build_revealed_engineering_override(
            base,
            validation_start=args.engineering_validation_start,
            validation_end=args.engineering_validation_end,
        )
    result = run(
        v4_config_path=args.v4_config.resolve(),
        measurement_path=args.measurement_config.resolve(),
        rolling_path=args.rolling_config.resolve(),
        promotion_path=args.promotion_config.resolve(),
        output_dir=args.output_dir.resolve(),
        max_origins=args.max_origins,
        rolling_override=rolling_override,
        resume_partial=args.resume_partial,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "completed_origins": result["completed_origins"],
                "decisions": result["decisions"],
                "trades": result["trades"],
                "orders_enabled": result["orders_enabled"],
            }
        )
    )


if __name__ == "__main__":
    main()
