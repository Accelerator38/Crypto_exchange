from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from panteon_legend.config import LEGEND_EXECUTION_CANDIDATE_ALLOW_LABELS
from panteon_legend.retrotest import build_legend_retrodate_config
from panteon_v2.analysis.soft_allocator import SoftAllocatorPolicy
from panteon_v2.analysis.retrodate_market_runner import run_retrodate_market_benchmark
from panteon_v2.selection.composer import PROFILE_GENETICS_RESEARCH

DEFAULT_RESULTS_ROOT = ROOT / "Results" / "PanteonLegend_experiments_dynamic_memory_20260531_full"
GENETICS_SPECIALISTS_ENV = "PANTEON_V2_GENETICS_SPECIALISTS_MANIFEST"
GENETICS_REGIME_ADAPTIVE_BIAS_ENV = "PANTEON_V2_GENETICS_REGIME_ADAPTIVE_BIAS_MANIFEST"
DEFAULT_GENETICS_SPECIALISTS_MANIFEST = (
    ROOT
    / "Results"
    / "neiro_genetics"
    / "live_active_shadow_20260604"
    / "genetics_specialists_manifest.json"
)
DEFAULT_GENETICS_REGIME_ADAPTIVE_BIAS_MANIFEST = (
    ROOT
    / "Results"
    / "neiro_genetics"
    / "BITGET"
    / "h4h6_regime_adaptive_runtime_20260603"
    / "regime_adaptive_output_bias_manifest_live_shadow_boost_0p5.json"
)
LATEST_OPTIONAL_AGENT_LABELS = (
    "GeneticsGenomeEnsemble",
    "GeneticsCore",
    "GeneticsBest",
    "GeneticsRiskTight",
    "GeneticsRegimeAdaptiveBias",
)
LATEST_GENETICS_FIXED_AGENT_PLAYER_SETS = (
    ("Legend_GeneticsGenome", ("GeneticsGenomeEnsemble",)),
    ("Legend_GeneticsCore", ("GeneticsCore",)),
    ("Legend_GeneticsBest", ("GeneticsBest",)),
    ("Legend_GeneticsRiskTight", ("GeneticsRiskTight",)),
    ("Legend_GeneticsRegimeAdaptiveBias", ("GeneticsRegimeAdaptiveBias",)),
)
LATEST_GENETICS_EXECUTION_ALLOW_LABELS = (
    "GeneticsResearch",
    "Legend_GeneticsGenome",
    "Legend_GeneticsCore",
    "Legend_GeneticsBest",
    "Legend_GeneticsRiskTight",
    "Legend_GeneticsRegimeAdaptiveBias",
)


def configure_default_genetics_manifests() -> None:
    defaults = (
        (GENETICS_SPECIALISTS_ENV, DEFAULT_GENETICS_SPECIALISTS_MANIFEST),
        (
            GENETICS_REGIME_ADAPTIVE_BIAS_ENV,
            DEFAULT_GENETICS_REGIME_ADAPTIVE_BIAS_MANIFEST,
        ),
    )
    for env_name, path in defaults:
        if os.environ.get(env_name):
            continue
        if path.exists():
            os.environ[env_name] = str(path)


def _expanded_allowlist() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            LEGEND_EXECUTION_CANDIDATE_ALLOW_LABELS
            + (
                "Legend_CurrentActorPool",
                "Legend_Bullish",
                "Legend_Consensus",
                "Legend_RotationFlow",
                "Legend_LegacyCore",
                "Legend_BullBreakout",
                "Legend_NeutralValidator",
                "Solo_RichardDennis",
                "Solo_LiveAfterShock",
                "Solo_LiveCrashHunter",
                "Solo_FundingArb",
                "Solo_PlayerFunding",
                "Solo_VolBreakoutHunter",
            )
        )
    )


def _positive_soft_allowlist() -> tuple[str, ...]:
    return (
        "Legend_Consensus",
        "Solo_MomentumScalper",
        "Solo_LiveAfterShock",
        "Solo_LiveOIBreakout",
        "Solo_ResearchValidatorAgent",
        "Solo_LiveTrendFollow",
        "Solo_LiveVolCompress",
        "Solo_LiveMeanRev",
    )


def _positive_core_soft_allowlist() -> tuple[str, ...]:
    return (
        "Solo_MomentumScalper",
        "Solo_LiveAfterShock",
        "Solo_LiveOIBreakout",
        "Solo_ResearchValidatorAgent",
        "Solo_LiveMeanRev",
    )


def _core_actor_symbol_denylist() -> tuple[str, ...]:
    return (
        "Solo_LiveAfterShock|MATIC/USDT",
        "Solo_LiveAfterShock|LINK/USDT",
        "Solo_MomentumScalper|LINK/USDT",
        "Solo_MomentumScalper|AVAX/USDT",
        "Solo_MomentumScalper|XLM/USDT",
        "Solo_LiveAfterShock|NEAR/USDT",
        "Solo_LiveAfterShock|FIL/USDT",
        "Solo_MomentumScalper|DOGE/USDT",
        "Solo_LiveAfterShock|LTC/USDT",
        "Solo_LiveAfterShock|AVAX/USDT",
    )


def _realized_gate_kwargs() -> dict[str, object]:
    return {
        "soft_allocator_realized_gate_enabled": True,
        "soft_allocator_realized_gate_lookback_bars": 720,
        "soft_allocator_realized_gate_min_closed_trades": 12,
        "soft_allocator_realized_gate_max_recent_pnl_usd": -25.0,
    }


def _early_realized_gate_kwargs() -> dict[str, object]:
    return {
        "soft_allocator_realized_gate_enabled": True,
        "soft_allocator_realized_gate_lookback_bars": 720,
        "soft_allocator_realized_gate_min_closed_trades": 5,
        "soft_allocator_realized_gate_max_recent_pnl_usd": -10.0,
    }


def _soft_regime_top1_24_policy(
    *,
    name: str,
    min_closed_trades: int,
) -> SoftAllocatorPolicy:
    return SoftAllocatorPolicy(
        name=name,
        top_k=1,
        cash_reserve_weight=0.0,
        max_weight_per_leader=1.0,
        min_closed_trades=min_closed_trades,
        half_life_bars=1_000_000,
        drawdown_penalty=0.0,
        persistent_loss_max_weight=0.0,
        score_scope="regime",
        rolling_window_bars=24,
    )


def _csv_labels(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        parts = raw.replace(";", ",").split(",")
    else:
        parts = [str(part) for part in raw]
    return tuple(part.strip() for part in parts if part.strip())


def _extend_unique(values: Sequence[str], extra: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in (*values, *extra):
        text = str(value or "").strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _extend_fixed_sets(
    values: Sequence[tuple[str, tuple[str, ...]]],
    extra: Sequence[tuple[str, tuple[str, ...]]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    out: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for label, agents in (*values, *extra):
        clean_label = str(label or "").strip()
        clean_agents = tuple(str(agent or "").strip() for agent in agents if str(agent or "").strip())
        if not clean_label or not clean_agents or clean_label in seen:
            continue
        out.append((clean_label, clean_agents))
        seen.add(clean_label)
    return tuple(out)


def _with_latest_genetics_players(config):
    profiles = list(config.player_profiles or ())
    profile_labels = {str(getattr(profile, "label", "") or "") for profile in profiles}
    if PROFILE_GENETICS_RESEARCH.label not in profile_labels:
        profiles.append(PROFILE_GENETICS_RESEARCH)
    return replace(
        config,
        fixed_agent_player_sets=_extend_fixed_sets(
            config.fixed_agent_player_sets,
            LATEST_GENETICS_FIXED_AGENT_PLAYER_SETS,
        ),
        player_profiles=tuple(profiles),
        soft_allocator_execution_allow_labels=_extend_unique(
            config.soft_allocator_execution_allow_labels,
            LATEST_GENETICS_EXECUTION_ALLOW_LABELS,
        ) if config.soft_allocator_execution_allow_labels else (),
        v3_candidate_allow_labels=_extend_unique(
            config.v3_candidate_allow_labels,
            LATEST_GENETICS_EXECUTION_ALLOW_LABELS,
        ),
    )


def build_config(
    profile: str,
    results_root: Path,
    *,
    include_optional_agents: bool = False,
    optional_agent_labels: Sequence[str] = (),
):
    optional_labels = _csv_labels(optional_agent_labels)
    if include_optional_agents and not optional_labels:
        optional_labels = LATEST_OPTIONAL_AGENT_LABELS
    config = _build_config(
        profile,
        results_root,
        include_optional_agents=include_optional_agents,
        optional_agent_labels=optional_labels,
    )
    if include_optional_agents:
        config = _with_latest_genetics_players(config)
    return config


def _build_config(
    profile: str,
    results_root: Path,
    *,
    include_optional_agents: bool = False,
    optional_agent_labels: Sequence[str] = (),
):
    base = build_legend_retrodate_config(
        results_root=results_root / f"{profile}_full2022_2026",
        data_dir=ROOT / "Retrodate",
        years=(2022, 2023, 2024, 2025, 2026),
        include_optional_agents=include_optional_agents,
        optional_agent_labels=optional_agent_labels,
    )
    if profile == "confirmed_current_gate_expanded":
        return replace(
            base,
            use_v3_executable_soft_top1_score=False,
            use_v3_executable_soft_confirmed_score=True,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_expanded_allowlist(),
        )
    if profile == "soft_regime_top3_w144_m20_cap45_cash15":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="regime_top3_w144_m20_cap45_cash15",
                top_k=3,
                cash_reserve_weight=0.15,
                max_weight_per_leader=0.45,
                min_closed_trades=20,
                half_life_bars=1_000_000,
                drawdown_penalty=0.0,
                persistent_loss_max_weight=0.0,
                score_scope="regime",
                rolling_window_bars=144,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_expanded_allowlist(),
        )
    if profile == "soft_regime_top3_w144_m20_cap45_cash15_only":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="regime_top3_w144_m20_cap45_cash15",
                top_k=3,
                cash_reserve_weight=0.15,
                max_weight_per_leader=0.45,
                min_closed_trades=20,
                half_life_bars=1_000_000,
                drawdown_penalty=0.0,
                persistent_loss_max_weight=0.0,
                score_scope="regime",
                rolling_window_bars=144,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_expanded_allowlist(),
        )
    if profile == "soft_regime_top1_w24_m50_only":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=_soft_regime_top1_24_policy(
                name="regime_top1_w24_m50",
                min_closed_trades=50,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_expanded_allowlist(),
        )
    if profile == "soft_regime_top1_24_only":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=_soft_regime_top1_24_policy(
                name="soft_regime_top1_24",
                min_closed_trades=20,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_expanded_allowlist(),
        )
    if profile == "soft_regime_top1_24_confirmed_only":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=_soft_regime_top1_24_policy(
                name="soft_regime_top1_24_confirmed",
                min_closed_trades=50,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_expanded_allowlist(),
        )
    if profile == "soft_regime_top3_w144_m20_cap45_cash15_only_positive":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="regime_top3_w144_m20_cap45_cash15_positive",
                top_k=3,
                cash_reserve_weight=0.15,
                max_weight_per_leader=0.45,
                min_closed_trades=20,
                half_life_bars=1_000_000,
                drawdown_penalty=0.0,
                persistent_loss_max_weight=0.0,
                score_scope="regime",
                rolling_window_bars=144,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_positive_soft_allowlist(),
            soft_allocator_execution_allow_labels=_positive_soft_allowlist(),
        )
    if profile == "soft_regime_top3_w144_m20_cap45_cash15_only_core":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="regime_top3_w144_m20_cap45_cash15_core",
                top_k=3,
                cash_reserve_weight=0.15,
                max_weight_per_leader=0.45,
                min_closed_trades=20,
                half_life_bars=1_000_000,
                drawdown_penalty=0.0,
                persistent_loss_max_weight=0.0,
                score_scope="regime",
                rolling_window_bars=144,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_positive_core_soft_allowlist(),
            soft_allocator_execution_allow_labels=_positive_core_soft_allowlist(),
        )
    if profile == "soft_regime_top3_w144_m20_cap45_cash15_only_core_symbol_gate":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="regime_top3_w144_m20_cap45_cash15_core_symbol_gate",
                top_k=3,
                cash_reserve_weight=0.15,
                max_weight_per_leader=0.45,
                min_closed_trades=20,
                half_life_bars=1_000_000,
                drawdown_penalty=0.0,
                persistent_loss_max_weight=0.0,
                score_scope="regime",
                rolling_window_bars=144,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_positive_core_soft_allowlist(),
            soft_allocator_execution_allow_labels=_positive_core_soft_allowlist(),
            soft_allocator_execution_deny_actor_symbols=_core_actor_symbol_denylist(),
        )
    if profile == "soft_regime_top3_w144_m20_cap45_cash15_only_core_symbol_gate_dyn":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="regime_top3_w144_m20_cap45_cash15_core_symbol_gate_dyn",
                top_k=3,
                cash_reserve_weight=0.15,
                max_weight_per_leader=0.45,
                min_closed_trades=20,
                half_life_bars=1_000_000,
                drawdown_penalty=0.0,
                persistent_loss_max_weight=0.0,
                score_scope="regime",
                rolling_window_bars=144,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_positive_core_soft_allowlist(),
            soft_allocator_execution_allow_labels=_positive_core_soft_allowlist(),
            soft_allocator_execution_deny_actor_symbols=_core_actor_symbol_denylist(),
            **_realized_gate_kwargs(),
        )
    if profile == "soft_regime_top3_w144_m20_cap45_cash15_only_core_dyn":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="regime_top3_w144_m20_cap45_cash15_core_dyn",
                top_k=3,
                cash_reserve_weight=0.15,
                max_weight_per_leader=0.45,
                min_closed_trades=20,
                half_life_bars=1_000_000,
                drawdown_penalty=0.0,
                persistent_loss_max_weight=0.0,
                score_scope="regime",
                rolling_window_bars=144,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_positive_core_soft_allowlist(),
            soft_allocator_execution_allow_labels=_positive_core_soft_allowlist(),
            **_realized_gate_kwargs(),
        )
    if profile == "soft_regime_top3_w144_m20_cap45_cash15_only_core_dyn_early":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="regime_top3_w144_m20_cap45_cash15_core_dyn_early",
                top_k=3,
                cash_reserve_weight=0.15,
                max_weight_per_leader=0.45,
                min_closed_trades=20,
                half_life_bars=1_000_000,
                drawdown_penalty=0.0,
                persistent_loss_max_weight=0.0,
                score_scope="regime",
                rolling_window_bars=144,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_positive_core_soft_allowlist(),
            soft_allocator_execution_allow_labels=_positive_core_soft_allowlist(),
            **_early_realized_gate_kwargs(),
        )
    if profile == "soft_regime_top2_w72_m10_cap55_cash10_only_core":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="regime_top2_w72_m10_cap55_cash10_core",
                top_k=2,
                cash_reserve_weight=0.10,
                max_weight_per_leader=0.55,
                min_closed_trades=10,
                half_life_bars=1_000_000,
                drawdown_penalty=0.0,
                persistent_loss_max_weight=0.0,
                score_scope="regime",
                rolling_window_bars=72,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_positive_core_soft_allowlist(),
            soft_allocator_execution_allow_labels=_positive_core_soft_allowlist(),
        )
    if profile == "soft_regime_top1_w48_m8_cap100_cash0_only_core":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="regime_top1_w48_m8_cap100_cash0_core",
                top_k=1,
                cash_reserve_weight=0.0,
                max_weight_per_leader=1.0,
                min_closed_trades=8,
                half_life_bars=1_000_000,
                drawdown_penalty=0.0,
                persistent_loss_max_weight=0.0,
                score_scope="regime",
                rolling_window_bars=48,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_positive_core_soft_allowlist(),
            soft_allocator_execution_allow_labels=_positive_core_soft_allowlist(),
        )
    if profile == "soft_regime_top3_w144_m20_cap45_cash15_only_positive_dyn":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="regime_top3_w144_m20_cap45_cash15_positive_dyn",
                top_k=3,
                cash_reserve_weight=0.15,
                max_weight_per_leader=0.45,
                min_closed_trades=20,
                half_life_bars=1_000_000,
                drawdown_penalty=0.0,
                persistent_loss_max_weight=0.0,
                score_scope="regime",
                rolling_window_bars=144,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_positive_soft_allowlist(),
            soft_allocator_execution_allow_labels=_positive_soft_allowlist(),
            **_realized_gate_kwargs(),
        )
    if profile == "soft_top3_decayed_only":
        return replace(
            base,
            soft_allocator_execution_enabled=True,
            soft_allocator_execution_soft_only=True,
            soft_allocator_execution_policy=SoftAllocatorPolicy(
                name="soft_top3_decayed",
                top_k=3,
                cash_reserve_weight=0.15,
                max_weight_per_leader=0.45,
                min_closed_trades=20,
                half_life_bars=2160,
                drawdown_penalty=0.15,
            ),
            max_new_opens_per_bar=3,
            current_actionable_candidate_layer_enabled=True,
            v3_candidate_allow_labels=_expanded_allowlist(),
        )
    if profile == "baseline":
        return base
    raise ValueError(f"unknown profile: {profile}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(
            "confirmed_current_gate_expanded",
            "soft_regime_top3_w144_m20_cap45_cash15",
            "soft_regime_top3_w144_m20_cap45_cash15_only",
            "soft_regime_top1_w24_m50_only",
            "soft_regime_top1_24_only",
            "soft_regime_top1_24_confirmed_only",
            "soft_regime_top3_w144_m20_cap45_cash15_only_positive",
            "soft_regime_top3_w144_m20_cap45_cash15_only_core",
            "soft_regime_top3_w144_m20_cap45_cash15_only_core_symbol_gate",
            "soft_regime_top3_w144_m20_cap45_cash15_only_core_symbol_gate_dyn",
            "soft_regime_top3_w144_m20_cap45_cash15_only_core_dyn",
            "soft_regime_top3_w144_m20_cap45_cash15_only_core_dyn_early",
            "soft_regime_top2_w72_m10_cap55_cash10_only_core",
            "soft_regime_top1_w48_m8_cap100_cash0_only_core",
            "soft_regime_top3_w144_m20_cap45_cash15_only_positive_dyn",
            "soft_top3_decayed_only",
            "baseline",
        ),
        default="confirmed_current_gate_expanded",
    )
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--max-bars", type=int, default=None)
    parser.add_argument("--shadow-parallel-workers", type=int, default=1)
    parser.add_argument("--include-optional-agents", action="store_true")
    parser.add_argument("--optional-agent-labels", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    configure_default_genetics_manifests()
    optional_labels = _csv_labels(args.optional_agent_labels)
    config = build_config(
        args.profile,
        args.results_root,
        include_optional_agents=args.include_optional_agents,
        optional_agent_labels=optional_labels,
    )
    if args.max_bars is not None:
        config = replace(config, max_bars=max(1, int(args.max_bars)))
    if args.shadow_parallel_workers > 1:
        config = replace(
            config,
            shadow_parallel_workers=max(1, int(args.shadow_parallel_workers)),
        )
    if args.dry_run:
        print(
            {
                "results_root": str(config.results_root),
                "include_optional_agents": config.include_optional_agents,
                "optional_agent_labels": list(config.optional_agent_labels),
                "player_profile_labels": [
                    str(getattr(profile, "label", "") or "")
                    for profile in config.player_profiles
                ],
                "fixed_agent_player_sets": [
                    {"label": label, "agents": list(agents)}
                    for label, agents in config.fixed_agent_player_sets
                ],
                "soft_allocator_execution_allow_labels": list(
                    config.soft_allocator_execution_allow_labels
                ),
                "v3_candidate_allow_labels": list(config.v3_candidate_allow_labels),
                "soft_allocator_execution_policy": (
                    getattr(config.soft_allocator_execution_policy, "name", "")
                    if config.soft_allocator_execution_policy is not None
                    else ""
                ),
                "max_bars": config.max_bars,
                "shadow_parallel_workers": config.shadow_parallel_workers,
            },
            flush=True,
        )
        return 0
    print(f"START {args.profile} results_root={config.results_root}", flush=True)
    started = time.time()
    summary = run_retrodate_market_benchmark(config)
    print(f"DONE seconds={time.time() - started:.1f}", flush=True)
    print(f"output_dir={summary.output_dir}", flush=True)
    print(f"analysis_report={summary.report_path}", flush=True)
    print(f"run_summary={summary.summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
