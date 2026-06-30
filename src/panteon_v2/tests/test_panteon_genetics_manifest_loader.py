from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from panteon_v2.domain.types import Action
from panteon_v2.selection import AgentRegistry
from panteon_v2.shadow import make_market_snapshot


class _FakeGeneticsAgent:
    def __init__(self, genome):
        self.genome = np.asarray(genome, dtype=np.float32)

    def act(self, prices, volumes, **_kwargs):
        return {symbol: int(self.genome[0]) for symbol in prices}


class _DefaultingFakeGeneticsAgent:
    def __init__(self, genome=None):
        if genome is None:
            genome = np.asarray([0], dtype=np.float32)
        self.genome = np.asarray(genome, dtype=np.float32)

    def act(self, prices, volumes, **_kwargs):
        return {symbol: int(self.genome[0]) for symbol in prices}


class _ConfigurableFakeGeneticsAgent(_FakeGeneticsAgent):
    def configure_regime_adaptive_output_bias(self, regime_open_bias, *, enabled=True):
        self._regime_adaptive_output_bias_map = {
            str(key): float(value)
            for key, value in dict(regime_open_bias).items()
        }
        self._regime_adaptive_output_bias_enabled = bool(enabled)
        return self


def _install_fake_crypto_genetics(monkeypatch):
    module = types.ModuleType("crypto_genetics")
    module.GeneticsAgent = _FakeGeneticsAgent
    module.GENOME_SIZE = 1
    monkeypatch.setitem(sys.modules, "crypto_genetics", module)
    return module


def _write_genome(path: Path, action: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray([action], dtype=np.float32))
    return path


def test_regime_router_routes_crash_candidate_when_manifest_has_crash_and_confident(tmp_path):
    from panteon_v2.app.agent_bootstrap import build_genetics_regime_router_adapter

    results_root = tmp_path / "Results" / "neiro_genetics"
    run_dir = results_root / "run"
    baseline_path = _write_genome(run_dir / "warm_start_genome.npy", 3)
    crash_path = _write_genome(run_dir / "crash_holdout_winner.npy", 4)
    manifest_path = run_dir / "selection_router.json"
    manifest_path.write_text(
        json.dumps(
            {
                "selected_is_baseline": False,
                "selected_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                    "crash": str(crash_path),
                },
                "baseline_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                    "crash": str(baseline_path),
                },
                "validation": {
                    "mean_ret": 1.1,
                    "min_ret": 0.5,
                    "positive_period_pct": 100.0,
                },
                "baseline_validation": {
                    "mean_ret": 1.0,
                    "min_ret": 0.5,
                    "positive_period_pct": 100.0,
                },
            }
        ),
        encoding="utf-8",
    )

    adapter = build_genetics_regime_router_adapter(
        manifest_path,
        _FakeGeneticsAgent,
        results_root=results_root,
        min_regime_confidence=0.70,
    )

    market = make_market_snapshot(
        bar=1,
        prices={"BTC": 100.0},
        regime="crash",
        regime_confidence=0.95,
    )

    assert adapter.act(market)["BTC"] == Action.FUT_SHORT_FULL


def test_regime_router_falls_back_to_baseline_when_confidence_is_low(tmp_path):
    from panteon_v2.app.agent_bootstrap import build_genetics_regime_router_adapter

    results_root = tmp_path / "Results" / "neiro_genetics"
    run_dir = results_root / "run"
    baseline_path = _write_genome(run_dir / "warm_start_genome.npy", 3)
    crash_path = _write_genome(run_dir / "crash_holdout_winner.npy", 4)
    manifest_path = run_dir / "selection_router.json"
    manifest_path.write_text(
        json.dumps(
            {
                "selected_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                    "crash": str(crash_path),
                },
                "baseline_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                    "crash": str(baseline_path),
                },
            }
        ),
        encoding="utf-8",
    )

    adapter = build_genetics_regime_router_adapter(
        manifest_path,
        _FakeGeneticsAgent,
        results_root=results_root,
        min_regime_confidence=0.70,
    )

    market = make_market_snapshot(
        bar=1,
        prices={"BTC": 100.0},
        regime="crash",
        regime_confidence=0.30,
    )

    assert adapter.act(market)["BTC"] == Action.FUT_LONG_FULL


def test_manifest_specialists_register_shadow_only_from_neiro_genetics_manifest(
    tmp_path,
    monkeypatch,
):
    from panteon_v2.app import agent_bootstrap

    _install_fake_crypto_genetics(monkeypatch)
    results_root = tmp_path / "Results" / "neiro_genetics"
    run_dir = results_root / "specialists"
    paths = {
        "GeneticsBest": _write_genome(run_dir / "overall.npy", 3),
        "GeneticsCrash": _write_genome(run_dir / "crash.npy", 4),
        "GeneticsRiskTight": _write_genome(run_dir / "risk_tight.npy", 2),
    }
    manifest_path = run_dir / "genetics_specialists_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "specialist_genome_map": {
                    label: str(path) for label, path in paths.items()
                },
                "paper_trading_eligible": True,
                "live_trading_eligible": False,
            }
        ),
        encoding="utf-8",
    )
    labels = tuple(paths)

    with monkeypatch.context() as mp:
        mp.setenv("PANTEON_V2_GENETICS_SPECIALISTS_MANIFEST", str(manifest_path))
        mp.setattr(agent_bootstrap, "_ensure_paths", lambda: None)
        registry = AgentRegistry()
        registered = agent_bootstrap.register_optional_agents(
            registry,
            optional_agent_labels=labels,
            skip_on_error=False,
        )

    assert registered == list(labels)
    for label in labels:
        adapter = registry.get(label)
        assert adapter.shadow_only is True
        assert adapter.paper_trading_eligible is True
        assert adapter.live_trading_eligible is False
        assert Path(adapter.source_genome_path).resolve() == paths[label].resolve()


def test_manifest_specialists_register_risk_tight_shadow_candidate(
    tmp_path,
    monkeypatch,
):
    from panteon_v2.app import agent_bootstrap

    _install_fake_crypto_genetics(monkeypatch)
    results_root = tmp_path / "Results" / "neiro_genetics"
    run_dir = results_root / "risk_tight"
    genome_path = _write_genome(run_dir / "risk_tight.npy", 3)
    manifest_path = run_dir / "genetics_specialists_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "specialist_genome_map": {
                    "GeneticsRiskTight": str(genome_path),
                },
                "paper_trading_eligible": False,
                "live_trading_eligible": False,
                "shadow_only": True,
            }
        ),
        encoding="utf-8",
    )

    with monkeypatch.context() as mp:
        mp.setenv("PANTEON_V2_GENETICS_SPECIALISTS_MANIFEST", str(manifest_path))
        mp.setattr(agent_bootstrap, "_ensure_paths", lambda: None)
        registry = AgentRegistry()
        registered = agent_bootstrap.register_optional_agents(
            registry,
            optional_agent_labels=("GeneticsRiskTight",),
            skip_on_error=False,
        )

    assert registered == ["GeneticsRiskTight"]
    adapter = registry.get("GeneticsRiskTight")
    assert adapter.shadow_only is True
    assert adapter.paper_trading_eligible is False
    assert adapter.live_trading_eligible is False
    assert Path(adapter.source_genome_path).resolve() == genome_path.resolve()


def test_core_registration_uses_explicit_genome_source(tmp_path, monkeypatch):
    from panteon_v2.app import agent_bootstrap

    module = _install_fake_crypto_genetics(monkeypatch)
    module.GeneticsAgent = _DefaultingFakeGeneticsAgent
    genome_path = _write_genome(
        tmp_path
        / "Results"
        / "neiro_genetics"
        / "core"
        / "best_genome.npy",
        4,
    )

    with monkeypatch.context() as mp:
        mp.setenv("PANTEON_V2_GENETICS_CORE_GENOME", str(genome_path))
        mp.setattr(agent_bootstrap, "_ensure_paths", lambda: None)
        registry = AgentRegistry()
        registered = agent_bootstrap.register_optional_agents(
            registry,
            optional_agent_labels=("GeneticsCore",),
            skip_on_error=False,
        )

    assert registered == ["GeneticsCore"]
    adapter = registry.get("GeneticsCore")
    assert adapter.v1_agent.genome.tolist() == [4.0]
    assert Path(adapter.source_genome_path).resolve() == genome_path.resolve()
    assert adapter.genetics_signal_source == "explicit_core_genome"


def test_core_registration_requires_real_genome_source(monkeypatch):
    from panteon_v2.app import agent_bootstrap

    module = _install_fake_crypto_genetics(monkeypatch)
    module.GeneticsAgent = _DefaultingFakeGeneticsAgent

    with monkeypatch.context() as mp:
        mp.delenv("PANTEON_V2_GENETICS_CORE_GENOME", raising=False)
        mp.delenv("PANTEON_V2_REQUIRE_REAL_GENETICS_CORE", raising=False)
        mp.setattr(agent_bootstrap, "_ensure_paths", lambda: None)
        mp.setattr(agent_bootstrap, "_resolve_genetics_core_genome_path", lambda: None)
        registry = AgentRegistry()

        with pytest.raises(ImportError, match="GeneticsCore requires a real genome"):
            agent_bootstrap.register_optional_agents(
                registry,
                optional_agent_labels=("GeneticsCore",),
                skip_on_error=False,
            )


def test_manifest_router_registers_when_manifest_env_is_set_and_label_requested(
    tmp_path,
    monkeypatch,
):
    from panteon_v2.app import agent_bootstrap

    _install_fake_crypto_genetics(monkeypatch)
    results_root = tmp_path / "Results" / "neiro_genetics"
    run_dir = results_root / "router"
    baseline_path = _write_genome(run_dir / "warm_start.npy", 3)
    crash_path = _write_genome(run_dir / "crash.npy", 4)
    manifest_path = run_dir / "genetics_specialists_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "selected_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                    "crash": str(crash_path),
                },
                "baseline_regime_map": {
                    "bearish": str(baseline_path),
                    "neutral": str(baseline_path),
                    "bullish": str(baseline_path),
                    "crash": str(baseline_path),
                },
                "paper_trading_eligible": False,
                "live_trading_eligible": False,
            }
        ),
        encoding="utf-8",
    )

    with monkeypatch.context() as mp:
        mp.setenv("PANTEON_V2_GENETICS_ROUTER_MANIFEST", str(manifest_path))
        mp.delenv("PANTEON_V2_LOAD_GENETICS_ROUTER", raising=False)
        mp.setattr(agent_bootstrap, "_ensure_paths", lambda: None)
        registry = AgentRegistry()
        registered = agent_bootstrap.register_optional_agents(
            registry,
            optional_agent_labels=("GeneticsRegimeRouter",),
            skip_on_error=False,
        )

    assert registered == ["GeneticsRegimeRouter"]
    adapter = registry.get("GeneticsRegimeRouter")
    assert adapter.shadow_only is True
    assert adapter.live_trading_eligible is False
    assert Path(adapter.selection_manifest_path).resolve() == manifest_path.resolve()


def test_regime_adaptive_bias_manifest_registers_shadow_only_configured_agent(
    tmp_path,
    monkeypatch,
):
    from panteon_v2.app import agent_bootstrap

    module = _install_fake_crypto_genetics(monkeypatch)
    module.GeneticsAgent = _ConfigurableFakeGeneticsAgent
    results_root = tmp_path / "Results" / "neiro_genetics"
    run_dir = results_root / "regime_adaptive"
    source_path = _write_genome(run_dir / "source.npy", 3)
    manifest_path = run_dir / "regime_adaptive_output_bias_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_genome": str(source_path),
                "regime_open_bias": {
                    "bearish": 0.75,
                    "neutral": 0.95,
                    "default": 0.75,
                },
                "promotion_eligible": True,
                "paper_trading_eligible": True,
                "live_trading_eligible": False,
            }
        ),
        encoding="utf-8",
    )

    with monkeypatch.context() as mp:
        mp.setenv("PANTEON_V2_GENETICS_REGIME_ADAPTIVE_BIAS_MANIFEST", str(manifest_path))
        mp.setattr(agent_bootstrap, "_ensure_paths", lambda: None)
        registered = agent_bootstrap.register_optional_agents(
            AgentRegistry(),
            optional_agent_labels=("GeneticsRegimeAdaptiveBias",),
            skip_on_error=False,
        )

    registry = AgentRegistry()
    with monkeypatch.context() as mp:
        mp.setenv("PANTEON_V2_GENETICS_REGIME_ADAPTIVE_BIAS_MANIFEST", str(manifest_path))
        mp.setattr(agent_bootstrap, "_ensure_paths", lambda: None)
        agent_bootstrap.register_optional_agents(
            registry,
            optional_agent_labels=("GeneticsRegimeAdaptiveBias",),
            skip_on_error=False,
        )

    assert registered == ["GeneticsRegimeAdaptiveBias"]
    adapter = registry.get("GeneticsRegimeAdaptiveBias")
    assert adapter.shadow_only is True
    assert adapter.live_trading_eligible is False
    assert adapter.paper_trading_eligible is True
    assert Path(adapter.source_genome_path).resolve() == source_path.resolve()
    assert adapter.regime_open_bias == {
        "bearish": 0.75,
        "neutral": 0.95,
        "default": 0.75,
    }
    assert adapter.v1_agent._regime_adaptive_output_bias_enabled is True
    assert adapter.v1_agent._regime_adaptive_output_bias_map["neutral"] == 0.95


def test_regime_adaptive_bias_manifest_applies_source_position_state_meta(
    tmp_path,
    monkeypatch,
):
    from panteon_v2.app import agent_bootstrap

    module = _install_fake_crypto_genetics(monkeypatch)
    module.GeneticsAgent = _ConfigurableFakeGeneticsAgent
    results_root = tmp_path / "Results" / "neiro_genetics"
    run_dir = results_root / "regime_adaptive"
    source_path = _write_genome(run_dir / "best_genome.npy", 3)
    (run_dir / "best_genome_meta.json").write_text(
        json.dumps({"position_state_features_enabled": True}),
        encoding="utf-8",
    )
    manifest_path = run_dir / "regime_adaptive_output_bias_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_genome": str(source_path),
                "regime_open_bias": {"default": 0.75},
                "promotion_eligible": True,
            }
        ),
        encoding="utf-8",
    )

    adapter = agent_bootstrap.build_genetics_regime_adaptive_bias_adapter(
        manifest_path,
        module.GeneticsAgent,
        results_root=results_root,
        expected_genome_size=module.GENOME_SIZE,
    )

    assert adapter.v1_agent._position_state_features_enabled is True


def test_regime_adaptive_bias_manifest_accepts_project_relative_results_path(
    tmp_path,
    monkeypatch,
):
    from panteon_v2.app import agent_bootstrap

    module = _install_fake_crypto_genetics(monkeypatch)
    module.GeneticsAgent = _ConfigurableFakeGeneticsAgent
    results_root = tmp_path / "Results" / "neiro_genetics"
    run_dir = results_root / "regime_adaptive"
    source_path = _write_genome(run_dir / "source.npy", 3)
    manifest_path = run_dir / "regime_adaptive_output_bias_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source_genome": "regime_adaptive/source.npy",
                "regime_open_bias": {"default": 0.75},
                "promotion_eligible": True,
            }
        ),
        encoding="utf-8",
    )

    registry = AgentRegistry()
    with monkeypatch.context() as mp:
        mp.chdir(tmp_path)
        mp.setenv(
            "PANTEON_V2_GENETICS_REGIME_ADAPTIVE_BIAS_MANIFEST",
            "Results/neiro_genetics/regime_adaptive/regime_adaptive_output_bias_manifest.json",
        )
        mp.setattr(agent_bootstrap, "_ensure_paths", lambda: None)
        registered = agent_bootstrap.register_optional_agents(
            registry,
            optional_agent_labels=("GeneticsRegimeAdaptiveBias",),
            skip_on_error=False,
        )

    assert registered == ["GeneticsRegimeAdaptiveBias"]
    adapter = registry.get("GeneticsRegimeAdaptiveBias")
    assert Path(adapter.selection_manifest_path).resolve() == manifest_path.resolve()
    assert Path(adapter.source_genome_path).resolve() == source_path.resolve()


def test_manifest_specialist_loader_rejects_genome_outside_neiro_genetics(
    tmp_path,
    monkeypatch,
):
    from panteon_v2.app import agent_bootstrap

    _install_fake_crypto_genetics(monkeypatch)
    results_root = tmp_path / "Results" / "neiro_genetics"
    safe_dir = results_root / "specialists"
    outside_dir = tmp_path / "Results" / "other"
    safe_dir.mkdir(parents=True)
    outside_dir.mkdir(parents=True)
    manifest_path = safe_dir / "genetics_specialists_manifest.json"
    outside_path = _write_genome(outside_dir / "crash.npy", 4)
    manifest_path.write_text(
        json.dumps(
            {
                "specialist_genome_map": {
                    "GeneticsCrash": str(outside_path),
                },
                "paper_trading_eligible": True,
                "live_trading_eligible": False,
            }
        ),
        encoding="utf-8",
    )

    with monkeypatch.context() as mp:
        mp.setenv("PANTEON_V2_GENETICS_SPECIALISTS_MANIFEST", str(manifest_path))
        mp.setattr(agent_bootstrap, "_ensure_paths", lambda: None)
        with pytest.raises(ValueError):
            agent_bootstrap.register_optional_agents(
                AgentRegistry(),
                optional_agent_labels=("GeneticsCrash",),
                skip_on_error=False,
            )


def test_genetics_specialist_clone_preserves_manifest_loaded_genome():
    from panteon_v2.shadow.adapters import GeneticsV2AgentAdapter

    adapter = GeneticsV2AgentAdapter(
        label="GeneticsCrash",
        v1_agent=_DefaultingFakeGeneticsAgent(genome=np.asarray([4], dtype=np.float32)),
    )

    clone = adapter.clone_for_shadow()
    market = make_market_snapshot(
        bar=1,
        prices={"BTC": 100.0},
        regime="crash",
        regime_confidence=0.95,
    )

    assert clone.act(market)["BTC"] == Action.FUT_SHORT_FULL
    assert clone.v1_agent is not adapter.v1_agent
    assert clone.v1_agent.genome.tolist() == [4.0]


def test_genetics_adapter_cache_is_scoped_by_runtime_source():
    from panteon_v2.shadow.adapters import GeneticsV2AgentAdapter

    class SourceAgent:
        def __init__(self, raw_action, source):
            self.raw_action = int(raw_action)
            self.source_genome_path = source

        def act(self, prices, volumes, **_kwargs):
            return {symbol: self.raw_action for symbol in prices}

    GeneticsV2AgentAdapter._act_cache.clear()
    GeneticsV2AgentAdapter._act_trace_cache.clear()
    market = make_market_snapshot(
        bar=1,
        prices={"BTC": 100.0},
        volumes={"BTC": 10.0},
        regime="bearish",
    )
    hold = GeneticsV2AgentAdapter(
        label="GeneticsCore",
        v1_agent=SourceAgent(0, "fallback.npy"),
    )
    short = GeneticsV2AgentAdapter(
        label="GeneticsCore",
        v1_agent=SourceAgent(4, "real.npy"),
    )

    assert hold.act(market)["BTC"] == Action.HOLD
    assert short.act(market)["BTC"] == Action.FUT_SHORT_FULL


def test_genetics_router_clone_preserves_manifest_loaded_regime_genomes():
    from panteon_v2.shadow.adapters import GeneticsRegimeRouterV2AgentAdapter

    adapter = GeneticsRegimeRouterV2AgentAdapter(
        label="GeneticsRegimeRouter",
        baseline_agent=_DefaultingFakeGeneticsAgent(genome=np.asarray([3], dtype=np.float32)),
        regime_agents={
            "crash": _DefaultingFakeGeneticsAgent(genome=np.asarray([4], dtype=np.float32)),
        },
        min_regime_confidence=0.70,
    )

    clone = adapter.clone_for_shadow()
    market = make_market_snapshot(
        bar=1,
        prices={"BTC": 100.0},
        regime="crash",
        regime_confidence=0.95,
    )

    assert clone.act(market)["BTC"] == Action.FUT_SHORT_FULL
    assert clone.baseline_agent is not adapter.baseline_agent
    assert clone.baseline_agent.genome.tolist() == [3.0]
    assert clone.regime_agents[next(iter(clone.regime_agents))].genome.tolist() == [4.0]
