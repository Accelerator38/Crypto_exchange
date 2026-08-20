from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROFILE_TOOL = ROOT / "tools" / "prepare_exia_4h_profile_spec_v1.py"
AUDIT_TOOL = ROOT / "tools" / "run_exia_timebase_audit_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_four_hour_profile_covers_every_runtime_temporal_field():
    profile = _load(
        ROOT / "src" / "panteon_runtime" / "timeframe_profiles.py",
        "timeframe_profiles_contract_test",
    )
    base = json.loads(
        (ROOT / "configs" / "exia_timebase_audit_v4_full_minute.json").read_text(
            encoding="utf-8"
        )
    )
    for component in base["agents"]:
        if component.get("engine", "act") != "act" or not component["temporal_fields"]:
            continue
        class_name = str(component["class_name"])
        configured = set(profile.FOUR_HOUR_CLASS_OVERRIDES.get(class_name, {}))
        assert set(component["temporal_fields"]) <= configured, component["agent_id"]


def test_profile_updates_nested_agents_and_preserves_bomberman_confirmation():
    profile = _load(
        ROOT / "src" / "panteon_runtime" / "timeframe_profiles.py",
        "timeframe_profiles_nested_test",
    )
    Bomberman = type(
        "Bomberman",
        (),
        {"__module__": "panteon_agents", "BOP_PERIOD": 30, "MRC_PERIOD": 50},
    )
    PlayerBomberman = type(
        "PlayerBomberman",
        (),
        {
            "__module__": "panteon_agents",
            "__init__": lambda self: (
                setattr(self, "_b1", Bomberman()),
                setattr(self, "_b2", Bomberman()),
            )[-1],
        },
    )
    player = PlayerBomberman()
    report = profile.apply_timeframe_profile(player, profile.FOUR_HOUR_PROFILE_ID)
    assert player._b1.BOP_PERIOD == 12
    assert player._b1.MRC_PERIOD == 36
    assert player._b2.BOP_THRESH == pytest.approx(0.70)
    assert player._b2.DONCHIAN_PERIOD == 30
    assert player._b2.MRC_PERIOD == 48
    assert report["objects_updated"] == 2


def test_profiled_factory_rejects_non_4h_runtime_and_records_contract():
    profile = _load(
        ROOT / "src" / "panteon_runtime" / "timeframe_profiles.py",
        "timeframe_profiles_factory_test",
    )

    class Dummy:
        pass

    with pytest.raises(ValueError, match="requires completed 4h bars"):
        profile.build_profiled_component(
            Dummy,
            profile_id=profile.FOUR_HOUR_PROFILE_ID,
            timeframe="1m",
        )

    component = profile.build_profiled_component(
        Dummy,
        profile_id=profile.FOUR_HOUR_PROFILE_ID,
        timeframe="4h",
    )
    assert component._timeframe_profile_report["profile_id"] == "exia_4h_v1"


def test_fixed_profile_is_sealed_to_4h_and_applied_to_actual_agent():
    audit = _load(AUDIT_TOOL, "run_exia_timebase_audit_fixed_profile_test")
    base = json.loads(
        (ROOT / "configs" / "exia_timebase_audit_v4_full_minute.json").read_text(
            encoding="utf-8"
        )
    )
    component = next(row for row in base["agents"] if row["agent_id"] == "MomentumScalper")
    agent, parameters, limited = audit.build_agent(
        component,
        timeframe_minutes=240,
        mode="fixed_profile",
        runtime_profile_id="exia_4h_v1",
    )
    assert agent.CHECK_INT == 1
    assert agent.EMA_F == 6
    assert agent.EMA_M == 18
    assert agent.EMA_S == 42
    assert parameters["__runtime_profile__"]["profile_id"] == "exia_4h_v1"
    assert limited == []
    with pytest.raises(ValueError, match="only be applied to 4h"):
        audit.build_agent(
            component,
            timeframe_minutes=60,
            mode="fixed_profile",
            runtime_profile_id="exia_4h_v1",
        )


def test_panteon_market_context_uses_replay_bar_session():
    audit = _load(AUDIT_TOOL, "run_exia_timebase_audit_replay_clock_test")
    base = json.loads(
        (ROOT / "configs" / "exia_timebase_audit_v4_full_minute.json").read_text(
            encoding="utf-8"
        )
    )
    component = next(row for row in base["agents"] if row["agent_id"] == "PanteonResearch")
    agent, _, _ = audit.build_agent(
        component,
        timeframe_minutes=240,
        mode="fixed_profile",
        runtime_profile_id="exia_4h_v1",
    )
    agent.set_replay_timestamp_ms(1_704_081_600_000)  # 2024-01-01 04:00:00 UTC
    context = agent._build_market_context(month=1)

    assert context["session_utc"] == "asia"
    assert context["season"] == "winter"


def test_generated_spec_has_one_profile_mode_and_no_authority():
    generator = _load(PROFILE_TOOL, "prepare_exia_4h_profile_spec_contract_test")
    result = generator.build_spec(
        base_spec_path=generator.DEFAULT_BASE_SPEC,
        main_index_path=generator.DEFAULT_MAIN_INDEX,
    )
    assert result["modes"] == ["fixed_profile"]
    assert result["runtime_profile_id"] == "exia_4h_v1"
    assert result["panels"][0]["timeframes_minutes"] == [240]
    assert result["stability_gate"]["minimum_adjacent_passing_timeframes"] == 1
    assert all(component["modes"] == ["fixed_profile"] for component in result["agents"])
    assert set(result["safety"].values()) == {False}
    candle = next(row for row in result["agents"] if row["agent_id"] == "SR_CandleMomentum3")
    assert candle["params"]["threshold_bps"] == 100.0


def test_primary_profile_is_the_single_source_for_timeframe_costs_and_gate():
    generator = _load(PROFILE_TOOL, "prepare_exia_4h_primary_contract_test")
    primary = json.loads(generator.DEFAULT_PRIMARY_PROFILE.read_text(encoding="utf-8"))
    result = generator.build_spec(
        base_spec_path=generator.DEFAULT_BASE_SPEC,
        main_index_path=generator.DEFAULT_MAIN_INDEX,
        primary_profile_path=generator.DEFAULT_PRIMARY_PROFILE,
    )
    assert result["runtime_profile_id"] == primary["profile_id"]
    assert result["costs_bps"] == primary["costs_bps"]
    assert result["stability_gate"]["minimum_trades_per_timeframe"] == primary[
        "research_gate"
    ]["minimum_closed_trades"]
    assert result["stability_gate"]["supported_split_minimum_trades"] == primary[
        "research_gate"
    ]["minimum_split_closed_trades"]
