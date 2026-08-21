from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "configs" / "exia_4h_profile_audit_v1.json"
DEFAULT_OUTPUT = ROOT / "configs" / "exia_4h_regime_discovery_v1.json"
VARIANTS = (
    ("FAST067", 0.67),
    ("SLOW150", 1.50),
    ("SLOW200", 2.00),
)


NEW_AGENTS: tuple[dict[str, Any], ...] = (
    {
        "agent_id": "NEW_KeltnerBreakout",
        "strategy_kind": "keltner_breakout",
        "temporal_fields": ["ema_span", "atr_window"],
        "params": {"ema_span": 20, "atr_window": 14, "atr_multiplier": 1.8},
        "hypothesis": "ATR-scaled channel breakouts adapt the entry distance to 4h volatility.",
    },
    {
        "agent_id": "NEW_DualThrust",
        "strategy_kind": "dual_thrust",
        "temporal_fields": ["window"],
        "params": {"window": 12, "coefficient": 0.55},
        "hypothesis": "Prior-range expansion from the current open captures directional impulse.",
    },
    {
        "agent_id": "NEW_RSIFailureSwing",
        "strategy_kind": "rsi_failure_swing",
        "temporal_fields": ["window"],
        "params": {"window": 14, "lower": 30.0, "upper": 70.0},
        "hypothesis": "Re-entry from an RSI extreme is more robust than trading the extreme itself.",
    },
    {
        "agent_id": "NEW_VolumePriceDivergence",
        "strategy_kind": "volume_price_divergence",
        "temporal_fields": ["lookback"],
        "params": {"lookback": 12, "price_threshold_bps": 80.0},
        "hypothesis": "Price and signed-volume divergence identifies exhausted directional moves.",
    },
    {
        "agent_id": "NEW_EfficiencyRatioTrend",
        "strategy_kind": "efficiency_ratio_trend",
        "temporal_fields": ["window"],
        "params": {"window": 16, "minimum_efficiency": 0.35, "minimum_move_bps": 50.0},
        "hypothesis": "Kaufman-style path efficiency separates persistent trends from noisy moves.",
    },
    {
        "agent_id": "NEW_VolatilityRegimeSwitch",
        "strategy_kind": "volatility_regime_switch",
        "temporal_fields": ["short_window", "long_window", "trend_lookback", "z_window"],
        "params": {
            "short_window": 6,
            "long_window": 30,
            "trend_lookback": 6,
            "z_window": 20,
            "z_entry": 1.5,
        },
        "hypothesis": "Momentum in high volatility and reversion in low volatility share one causal policy.",
    },
    {
        "agent_id": "NEW_RollingVWAPReversion",
        "strategy_kind": "rolling_vwap_reversion",
        "temporal_fields": ["window"],
        "params": {"window": 24, "entry_z": 1.8},
        "hypothesis": "Volume-weighted fair-value deviations revert on liquid 4h bars.",
    },
    {
        "agent_id": "NEW_CandleStructureBreakout",
        "strategy_kind": "candle_structure_breakout",
        "temporal_fields": ["volume_window"],
        "params": {
            "volume_window": 20,
            "volume_multiplier": 1.25,
            "body_fraction": 0.65,
            "close_location": 0.80,
        },
        "hypothesis": "Large directional bodies closing near an extreme with volume confirm continuation.",
    },
    {
        "agent_id": "NEW_CrossSectionalResidualMomentum",
        "strategy_kind": "cross_sectional_residual_momentum",
        "temporal_fields": ["lookback"],
        "params": {"lookback": 12, "rank_threshold": 0.75},
        "hypothesis": "Market-demeaned full8 momentum retains relative continuation information.",
    },
    {
        "agent_id": "NEW_CrossSectionalShortReversal",
        "strategy_kind": "cross_sectional_short_term_reversal",
        "temporal_fields": ["lookback"],
        "params": {"lookback": 2, "rank_threshold": 0.75},
        "hypothesis": "Extreme short-horizon full8 relative moves partially reverse.",
    },
)


def _profile_overrides() -> dict[str, dict[str, Any]]:
    import sys

    runtime = ROOT / "src" / "panteon_runtime"
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    from timeframe_profiles import FOUR_HOUR_CLASS_OVERRIDES  # noqa: PLC0415

    return FOUR_HOUR_CLASS_OVERRIDES


def _scaled_temporal_values(values: dict[str, Any], multiplier: float) -> dict[str, int]:
    return {
        field: max(1, int(math.ceil(int(value) * multiplier)))
        for field, value in values.items()
    }


def _runtime_variants(component: dict[str, Any]) -> list[dict[str, Any]]:
    profile = _profile_overrides()
    class_name = str(component["class_name"])
    class_profile = profile.get(class_name, {})
    temporal = {
        field: class_profile[field]
        for field in component.get("temporal_fields", [])
        if field in class_profile
    }
    if len(temporal) != len(component.get("temporal_fields", [])):
        missing = sorted(set(component.get("temporal_fields", [])) - set(temporal))
        raise ValueError(f"missing 4h profile temporal fields for {class_name}: {missing}")
    variants = []
    for variant_id, multiplier in VARIANTS:
        variant = copy.deepcopy(component)
        variant["agent_id"] = f"{component['agent_id']}__{variant_id}"
        variant["base_agent_id"] = component["agent_id"]
        variant["variant_id"] = variant_id
        variant["post_profile_overrides"] = _scaled_temporal_values(temporal, multiplier)
        variants.append(variant)
    return variants


def _strategy_variants(component: dict[str, Any]) -> list[dict[str, Any]]:
    temporal = {
        field: component["params"][field] for field in component.get("temporal_fields", [])
    }
    variants = []
    for variant_id, multiplier in VARIANTS:
        variant = copy.deepcopy(component)
        variant["agent_id"] = f"{component['agent_id']}__{variant_id}"
        variant["base_agent_id"] = component["agent_id"]
        variant["variant_id"] = variant_id
        variant["params"].update(_scaled_temporal_values(temporal, multiplier))
        variants.append(variant)
    return variants


def build_spec(base_path: Path = DEFAULT_BASE) -> dict[str, Any]:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    result = copy.deepcopy(base)
    result["audit_id"] = "exia_4h_all_agent_regime_discovery_v1"
    result["purpose"] = (
        "Rank the maximum comparable set of current, varied and new agents on completed 4h "
        "Bitget full8 bars, attributing trades to market regimes without an external NoTrade gate."
    )
    result["bootstrap"]["samples"] = 2000
    result["selection_policy"] = {
        "experiment_kind": "diagnostic_regime_discovery",
        "primary_timeframe_minutes": 240,
        "external_regime_gate": False,
        "no_trade_by_market_regime": False,
        "ranking_is_promotion_authority": False,
        "l2_history_available": False,
    }
    result["regime_attribution"] = {
        "label_only": True,
        "labels": ["bullish", "bearish", "neutral"],
        "market_lookback": 12,
        "market_threshold_bps": 100.0,
        "market_breadth": 0.625,
        "market_minimum_symbols": 6,
        "market_confirmation_bars": 2,
        "trade_regime_timestamp": "entry_timestamp",
    }
    current = list(result["agents"])
    expanded: list[dict[str, Any]] = []
    for component in current:
        base_component = copy.deepcopy(component)
        base_component["base_agent_id"] = component["agent_id"]
        base_component["variant_id"] = "BASE"
        expanded.append(base_component)
        if component.get("component_type") == "agent" and component.get("engine", "act") == "act":
            expanded.extend(_runtime_variants(component))
        elif component.get("component_type") == "strategy":
            expanded.extend(_strategy_variants(component))
    for definition in NEW_AGENTS:
        expanded.append(
            {
                **copy.deepcopy(definition),
                "base_agent_id": definition["agent_id"],
                "variant_id": "NEW_BASE",
                "component_type": "new_strategy",
                "engine": "simple_research",
                "parameter_base_minutes": 240,
                "modes": ["fixed_profile"],
            }
        )
    result["agents"] = expanded
    result["excluded_components"] = list(result.get("excluded_components", [])) + [
        {
            "component": "NEW_L2OrderBookAgent",
            "reason": (
                "No continuous point-in-time L2 snapshots exist for the common 2022-2026 full8 window; "
                "using OHLCV proxies would mislabel the experiment."
            ),
        }
    ]
    result["experiment_counts"] = {
        "current_components": len(current),
        "current_runtime_agents": sum(row.get("component_type") == "agent" for row in current),
        "current_simple_strategies": sum(row.get("component_type") == "strategy" for row in current),
        "additional_variants": len(expanded) - len(current) - len(NEW_AGENTS),
        "new_agents": len(NEW_AGENTS),
        "total_components": len(expanded),
    }
    validate_discovery_spec(result)
    return result


def validate_discovery_spec(spec: dict[str, Any]) -> None:
    counts = spec["experiment_counts"]
    if counts != {
        "current_components": 43,
        "current_runtime_agents": 22,
        "current_simple_strategies": 8,
        "additional_variants": 90,
        "new_agents": 10,
        "total_components": 143,
    }:
        raise ValueError(f"unexpected experiment counts: {counts}")
    identifiers = [str(row["agent_id"]) for row in spec["agents"]]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("component identifiers must be unique")
    if spec["selection_policy"]["external_regime_gate"] is not False:
        raise ValueError("external market-regime gate must remain disabled")
    if spec["regime_attribution"]["label_only"] is not True:
        raise ValueError("regimes must be attribution labels only")
    if set(spec["safety"].values()) != {False}:
        raise ValueError("discovery experiment cannot grant trading authority")
    if spec["modes"] != ["fixed_profile"]:
        raise ValueError("discovery experiment must use the fixed 4h profile")
    if any(panel["timeframes_minutes"] != [240] for panel in spec["panels"]):
        raise ValueError("all discovery panels must use only completed 4h bars")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Exia 4h all-agent regime discovery")
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base if args.base.is_absolute() else ROOT / args.base
    output = args.output if args.output.is_absolute() else ROOT / args.output
    payload = build_spec(base.resolve())
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["experiment_counts"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
