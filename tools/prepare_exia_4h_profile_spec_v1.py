from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_SPEC = ROOT / "configs" / "exia_timebase_audit_v4_full_minute.json"
DEFAULT_MAIN_INDEX = (
    ROOT / "Retrodate" / "exia_full8_timeframes_v1_20220101_20260714" / "index.json"
)
DEFAULT_PRIMARY_PROFILE = ROOT / "configs" / "exia_primary_timeframe_v1.json"
DEFAULT_OUTPUT = ROOT / "configs" / "exia_4h_profile_audit_v1.json"


SIMPLE_RESEARCH_4H_PARAMS: dict[str, dict[str, Any]] = {
    "SR_Flat": {},
    "SR_LongOnly": {},
    "SR_EMATrend12_48": {"fast": 12, "slow": 48, "threshold_bps": 12.0},
    "SR_EMATrend24_96": {"fast": 24, "slow": 96, "threshold_bps": 16.0},
    "SR_Donchian20": {"window": 20},
    "SR_Donchian55": {"window": 55},
    "SR_MeanReversion24": {"window": 24, "entry_z": 2.5, "exit_z": 0.5},
    "SR_VolCompressionBreakout": {
        "compression_threshold": 0.65,
        "fast_span": 12,
        "slow_span": 48,
        "short_vol_window": 24,
        "long_vol_window": 96,
        "short_vol_min_periods": 12,
        "long_vol_min_periods": 48,
        "breakout_window": 20,
    },
    "SR_RegimePullback12_48": {"fast": 12, "slow": 48, "pullback_bps": 80.0},
    "SR_CandleMomentum3": {"bars": 3, "threshold_bps": 100.0},
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _derived_4h(index_path: Path) -> dict[str, Any]:
    index = load_json(index_path)
    item = index.get("timeframes", {}).get("240")
    if not isinstance(item, dict):
        raise ValueError(f"4h timeframe is absent from {index_path}")
    return copy.deepcopy(item)


def _panel(
    *,
    panel_id: str,
    panel_manifest_path: Path,
    timeframe_index_path: Path,
    windows: list[dict[str, str]],
) -> dict[str, Any]:
    manifest = load_json(panel_manifest_path)
    if manifest.get("validation", {}).get("passed") is not True:
        raise ValueError(f"panel validation failed: {panel_manifest_path}")
    return {
        "panel_id": panel_id,
        "manifest": str(panel_manifest_path.relative_to(ROOT)),
        "dataset_sha256": str(manifest["dataset_sha256"]),
        "source_timeframe_minutes": 1,
        "timeframes_minutes": [240],
        "windows": windows,
        "derived_timeframe_manifests": {"240": _derived_4h(timeframe_index_path)},
    }


def build_spec(
    *,
    base_spec_path: Path,
    main_index_path: Path,
    primary_profile_path: Path = DEFAULT_PRIMARY_PROFILE,
    holdout_panel_manifest: Path | None = None,
    holdout_index_path: Path | None = None,
) -> dict[str, Any]:
    base = load_json(base_spec_path)
    primary = load_json(primary_profile_path)
    if int(primary.get("timeframe_minutes", 0)) != 240:
        raise ValueError("primary Exia profile must be sealed to 4h")
    if set(primary.get("safety", {}).values()) != {False}:
        raise ValueError("primary Exia profile must not grant trading authority")
    research_gate = primary["research_gate"]
    source_panel = copy.deepcopy(base["panels"][0])
    agents = copy.deepcopy(base["agents"])
    for component in agents:
        component["modes"] = ["fixed_profile"]
        if component.get("engine", "act") == "simple_research":
            agent_id = str(component["agent_id"])
            if agent_id not in SIMPLE_RESEARCH_4H_PARAMS:
                raise ValueError(f"missing fixed 4h parameters: {agent_id}")
            component["params"] = copy.deepcopy(SIMPLE_RESEARCH_4H_PARAMS[agent_id])
            component["parameter_base_minutes"] = 240

    panels = [
        _panel(
            panel_id="full8_4h_20220101_20260714",
            panel_manifest_path=ROOT / str(source_panel["manifest"]),
            timeframe_index_path=main_index_path,
            windows=copy.deepcopy(source_panel["windows"]),
        )
    ]
    if (holdout_panel_manifest is None) != (holdout_index_path is None):
        raise ValueError("holdout panel and timeframe index must be provided together")
    if holdout_panel_manifest is not None and holdout_index_path is not None:
        holdout_manifest = load_json(holdout_panel_manifest)
        first = pd.Timestamp(str(holdout_manifest["requested_start_date"]), tz="UTC")
        end = pd.Timestamp(str(holdout_manifest["requested_end_date"]), tz="UTC") + pd.Timedelta(
            days=1
        )
        panels.append(
            _panel(
                panel_id="untouched_4h_20260715_20260817",
                panel_manifest_path=holdout_panel_manifest,
                timeframe_index_path=holdout_index_path,
                windows=[
                    {
                        "name": "prospective_holdout",
                        "start": first.isoformat(),
                        "end": end.isoformat(),
                    }
                ],
            )
        )

    return {
        "schema_version": "exia.timebase_audit.v1",
        "audit_id": "exia_all_runtime_components_fixed_4h_profile_v1",
        "purpose": (
            "Evaluate every comparable runtime component under one explicit 4h-bar "
            "profile, with no profile selection from the post-2026-07-14 holdout."
        ),
        "action_normalization": base["action_normalization"],
        "safety": {
            "paper_allowed": False,
            "live_allowed": False,
            "orders_enabled": False,
            "promotion_authority": False,
        },
        "runtime_profile_id": str(primary["profile_id"]),
        "selection_policy": {
            "primary_timeframe_minutes": 240,
            "profile_family_size": 1,
            "profile_selected_before_holdout": True,
            "holdout_not_used_for_parameter_selection": True,
            "baseline_audit": "Reports/Exia/timebase_audit_v4_full_minute",
        },
        "costs_bps": copy.deepcopy(primary["costs_bps"]),
        "bootstrap": copy.deepcopy(base["bootstrap"]),
        "stability_gate": {
            **copy.deepcopy(base["stability_gate"]),
            "minimum_trades_per_timeframe": int(research_gate["minimum_closed_trades"]),
            "supported_split_minimum_trades": int(
                research_gate["minimum_split_closed_trades"]
            ),
            "require_positive_stress_mean": bool(
                research_gate["require_positive_stress_mean"]
            ),
            "require_positive_stress_lcb": bool(
                research_gate["require_positive_stress_lcb"]
            ),
            "minimum_adjacent_passing_timeframes": 1,
        },
        "panels": panels,
        "modes": ["fixed_profile"],
        "agents": agents,
        "equivalent_components": copy.deepcopy(base.get("equivalent_components", [])),
        "excluded_components": copy.deepcopy(base.get("excluded_components", [])),
        "stability_panel_id": "full8_4h_20220101_20260714",
        "holdout_panel_id": (
            "untouched_4h_20260715_20260817" if len(panels) == 2 else None
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the sealed Exia 4h profile audit")
    parser.add_argument("--base-spec", type=Path, default=DEFAULT_BASE_SPEC)
    parser.add_argument("--main-timeframe-index", type=Path, default=DEFAULT_MAIN_INDEX)
    parser.add_argument("--primary-profile", type=Path, default=DEFAULT_PRIMARY_PROFILE)
    parser.add_argument("--holdout-panel-manifest", type=Path)
    parser.add_argument("--holdout-timeframe-index", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = build_spec(
        base_spec_path=args.base_spec.resolve(),
        main_index_path=args.main_timeframe_index.resolve(),
        primary_profile_path=args.primary_profile.resolve(),
        holdout_panel_manifest=(
            None if args.holdout_panel_manifest is None else args.holdout_panel_manifest.resolve()
        ),
        holdout_index_path=(
            None if args.holdout_timeframe_index is None else args.holdout_timeframe_index.resolve()
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(spec, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "components": len(spec["agents"]),
                "panels": len(spec["panels"]),
                "runtime_profile_id": spec["runtime_profile_id"],
                "orders_enabled": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
