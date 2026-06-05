from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np


def _load_crypto_genetics():
    root = Path(__file__).resolve().parents[3]
    genetics_dir = root / "Genetics_DL_Agents"
    if str(genetics_dir) not in sys.path:
        sys.path.insert(0, str(genetics_dir))
    return importlib.import_module("crypto_genetics")


def test_genetics_agent_applies_regime_adaptive_open_bias_before_argmax(monkeypatch):
    cg = _load_crypto_genetics()
    agent = cg.GeneticsAgent(genome=np.zeros(cg.GENOME_SIZE, dtype=np.float32))
    agent.configure_regime_adaptive_output_bias(
        {"neutral": 1.0, "bearish": 0.5},
        enabled=True,
    )

    def fake_fwd_np(x, *_weights):
        logits = np.zeros((x.shape[0], cg.N_ACTIONS), dtype=np.float32)
        logits[:, 0] = 0.0
        logits[:, 1] = 0.8
        return logits

    monkeypatch.setattr(cg, "_fwd_np", fake_fwd_np)

    action = {}
    for idx in range(25):
        action = agent.act({"BTC": 100.0 + idx * 0.01}, {"BTC": 1000.0}, month=4)

    assert action == {"BTC": 0}
    assert agent.last_regime_adaptive_output_bias == {
        "enabled": True,
        "regime": "neutral",
        "raw_regime": "unknown",
        "open_output_bias": 1.0,
    }


def test_genetics_v2_adapter_exposes_runtime_bias_trace():
    from panteon_v2.shadow import make_market_snapshot
    from panteon_v2.shadow.adapters import GeneticsV2AgentAdapter

    class LegacyGeneticsAgent:
        def __init__(self):
            self.last_regime_adaptive_output_bias = {}

        def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
            self.last_regime_adaptive_output_bias = {
                "enabled": True,
                "regime": "neutral",
                "raw_regime": "neutral",
                "open_output_bias": 0.95,
            }
            return {"BTC": 3}

    wrapped = GeneticsV2AgentAdapter("GeneticsCore", LegacyGeneticsAgent())
    market = make_market_snapshot(
        bar=1,
        prices={"BTC": 100.0},
        regime="neutral",
    )

    assert wrapped.act(market)["BTC"].is_open
    assert wrapped.last_regime_adaptive_output_bias == {
        "enabled": True,
        "regime": "neutral",
        "raw_regime": "neutral",
        "open_output_bias": 0.95,
    }


def test_shadow_agent_signal_records_regime_adaptive_bias_trace():
    from panteon_v2.app.shadow_tournament import ProductionShadowTournament
    from panteon_v2.domain.types import Action
    from panteon_v2.execution import RiskLimitsConfig
    from panteon_v2.memory import PerformanceMemory
    from panteon_v2.selection import AgentRegistry
    from panteon_v2.shadow import make_market_snapshot

    class RegimeAdaptiveAgent:
        label = "GeneticsRegimeAdaptiveBias"

        def __init__(self):
            self.last_regime_adaptive_output_bias = {}

        def act(self, market):
            self.last_regime_adaptive_output_bias = {
                "enabled": True,
                "regime": "neutral",
                "raw_regime": "neutral",
                "open_output_bias": 0.95,
            }
            return {"BTC": Action.FUT_LONG_FULL}

    registry = AgentRegistry()
    registry.register(RegimeAdaptiveAgent())
    tournament = ProductionShadowTournament(
        registry=registry,
        perf=PerformanceMemory(trade_fraction=1.0),
        risk_config=RiskLimitsConfig(capital_fraction=0.10, min_notional_usd=0.0),
        runtime_event_logs_enabled=False,
    )
    market = make_market_snapshot(
        bar=1,
        prices={"BTC": 100.0},
        regime="neutral",
    )

    tournament.run_bar(market, players=(), balance_usd=10_000.0)

    signal = tournament.last_agent_signals()["GeneticsRegimeAdaptiveBias"][0]
    assert signal.metadata["regime_adaptive_output_bias"] == {
        "enabled": True,
        "regime": "neutral",
        "raw_regime": "neutral",
        "open_output_bias": 0.95,
    }
