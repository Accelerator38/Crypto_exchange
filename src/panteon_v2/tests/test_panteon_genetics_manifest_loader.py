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
        "GeneticsBullish": _write_genome(run_dir / "bullish.npy", 1),
        "GeneticsBearish": _write_genome(run_dir / "bearish.npy", 4),
        "GeneticsNeutral": _write_genome(run_dir / "neutral.npy", 0),
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
