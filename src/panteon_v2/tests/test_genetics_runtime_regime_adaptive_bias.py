from __future__ import annotations

import importlib
import sys
from collections import deque
from pathlib import Path

import numpy as np


def _load_crypto_genetics():
    root = Path(__file__).resolve().parents[3]
    genetics_dir = root / "Genetics_DL_Agents"
    if str(genetics_dir) not in sys.path:
        sys.path.insert(0, str(genetics_dir))
    return importlib.import_module("crypto_genetics")


def test_action_contract_metrics_reports_directional_exposure_bias():
    cg = _load_crypto_genetics()

    short_only = np.zeros((1, 3, 2), dtype=np.int32)
    short_only[0, 0, 0] = 6
    balanced = np.zeros((1, 3, 2), dtype=np.int32)
    balanced[0, 0, 0] = 4
    balanced[0, 0, 1] = 6

    short_metrics = cg._action_contract_metrics(short_only)
    balanced_metrics = cg._action_contract_metrics(balanced)

    assert short_metrics["net_direction_biases"][0] == -1.0
    assert balanced_metrics["net_direction_biases"][0] == 0.0
    assert short_metrics["mean_short_slot_rates"][0] > 0.0
    assert balanced_metrics["mean_long_slot_rates"][0] > 0.0


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
    assert {
        key: agent.last_regime_adaptive_output_bias[key]
        for key in ("enabled", "regime", "raw_regime", "open_output_bias")
    } == {
        "enabled": True,
        "regime": "neutral",
        "raw_regime": "unknown",
        "open_output_bias": 1.0,
    }
    assert agent.last_regime_adaptive_output_bias["by_symbol"]["BTC"][
        "action_confidence"
    ] > 0.0
    assert "logit_margin" in agent.last_regime_adaptive_output_bias["by_symbol"]["BTC"]


def test_genetics_agent_negative_regime_bias_boosts_open_logits(monkeypatch):
    cg = _load_crypto_genetics()
    agent = cg.GeneticsAgent(genome=np.zeros(cg.GENOME_SIZE, dtype=np.float32))
    agent.configure_regime_adaptive_output_bias(
        {"neutral": -0.25},
        enabled=True,
    )

    def fake_fwd_np(x, *_weights):
        logits = np.zeros((x.shape[0], cg.N_ACTIONS), dtype=np.float32)
        logits[:, 0] = 0.50
        logits[:, 1] = 0.40
        return logits

    monkeypatch.setattr(cg, "_fwd_np", fake_fwd_np)

    action = {}
    for idx in range(25):
        action = agent.act({"BTC": 100.0 + idx * 0.01}, {"BTC": 1000.0}, month=4)

    assert action == {"BTC": 1}
    assert agent.last_regime_adaptive_output_bias["open_output_bias"] == -0.25


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


def test_genetics_v2_adapter_uses_symbol_local_regime_for_adaptive_bias():
    from panteon_v2.domain.types import Regime
    from panteon_v2.shadow import make_market_snapshot
    from panteon_v2.shadow.adapters import GeneticsV2AgentAdapter

    class LegacyAdaptiveAgent:
        def __init__(self):
            self.calls = 0
            self._regime_adaptive_output_bias_enabled = True
            self._regime_adaptive_output_bias_map = {
                "bearish": 0.90,
                "range_low_vol": 0.70,
                "mixed_rotational": 0.0,
            }
            self.last_regime_adaptive_output_bias = {}

        def clone_for_shadow(self):
            clone = LegacyAdaptiveAgent()
            clone._regime_adaptive_output_bias_map = dict(
                self._regime_adaptive_output_bias_map
            )
            return clone

        def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
            self.calls += 1
            raw_regime = str(
                getattr(self, "_regime_adaptive_output_bias_regime_override", "")
                or "mixed_rotational"
            )
            bias = float(self._regime_adaptive_output_bias_map.get(raw_regime, 0.0))
            self.last_regime_adaptive_output_bias = {
                "enabled": True,
                "regime": raw_regime,
                "raw_regime": raw_regime,
                "open_output_bias": bias,
            }
            return {
                "BTC": 4 if raw_regime == "bearish" else 0,
                "ETH": 3 if raw_regime == "range_low_vol" else 0,
                "SOL": 0,
            }

    legacy = LegacyAdaptiveAgent()
    wrapped = GeneticsV2AgentAdapter("GeneticsRegimeAdaptiveBias", legacy)
    market = make_market_snapshot(
        bar=1,
        prices={"BTC": 100.0, "ETH": 50.0, "SOL": 20.0},
        regime="mixed_rotational",
        regimes_by_symbol={
            "BTC": Regime.BEARISH,
            "ETH": Regime.RANGE_LOW_VOL,
            "SOL": Regime.MIXED_ROTATIONAL,
        },
    )

    actions = wrapped.act(market)

    assert actions["BTC"].is_open
    assert actions["ETH"].is_open
    assert actions["SOL"].is_hold
    assert legacy.calls == 1
    trace = wrapped.last_regime_adaptive_output_bias
    assert trace["by_symbol"]["BTC"]["raw_regime"] == "bearish"
    assert trace["by_symbol"]["ETH"]["raw_regime"] == "range_low_vol"


def test_genetics_v2_adapter_symbol_local_clone_preserves_warm_runtime_state():
    import numpy as np

    from panteon_v2.domain.types import Regime
    from panteon_v2.shadow import make_market_snapshot
    from panteon_v2.shadow.adapters import GeneticsV2AgentAdapter

    class WarmupSensitiveAdaptiveAgent:
        def __init__(self, genome=None):
            self.genome = np.array(genome if genome is not None else [1.0])
            self.t = 0
            self.ph = {"BTC": [99.0, 100.0]}
            self.vh = {"BTC": [1000.0, 1000.0]}
            self._regime_adaptive_output_bias_enabled = True
            self._regime_adaptive_output_bias_map = {"bearish": 0.90}
            self.last_regime_adaptive_output_bias = {}

        def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
            self.t += 1
            raw_regime = str(
                getattr(self, "_regime_adaptive_output_bias_regime_override", "")
                or "mixed_rotational"
            )
            bias = float(self._regime_adaptive_output_bias_map.get(raw_regime, 0.0))
            self.last_regime_adaptive_output_bias = {
                "enabled": True,
                "regime": raw_regime,
                "raw_regime": raw_regime,
                "open_output_bias": bias,
            }
            if self.t < 11:
                return {"BTC": 0}
            return {"BTC": 4 if raw_regime == "bearish" else 0}

    legacy = WarmupSensitiveAdaptiveAgent()
    legacy.t = 10
    wrapped = GeneticsV2AgentAdapter("GeneticsRegimeAdaptiveBias", legacy)
    market = make_market_snapshot(
        bar=1,
        prices={"BTC": 100.0},
        regime="mixed_rotational",
        regimes_by_symbol={"BTC": Regime.BEARISH},
    )

    actions = wrapped.act(market)

    assert actions["BTC"].is_open
    assert legacy.t == 11
    assert wrapped.last_regime_adaptive_output_bias["by_symbol"]["BTC"] == {
        "enabled": True,
        "regime": "bearish",
        "raw_regime": "bearish",
        "open_output_bias": 0.90,
    }


def test_genetics_runtime_clone_keeps_bounded_tail_of_warm_history():
    from panteon_v2.shadow.adapters import _clone_genetics_runtime_agent

    class WarmHistoryAgent:
        def __init__(self, genome=None):
            self.genome = np.array(genome if genome is not None else [1.0, 2.0])
            self.ph = {"BTC": deque(range(42_000), maxlen=42_000)}
            self.vh = {"BTC": deque(range(18_000), maxlen=18_000)}
            self.spot_qty = {"BTC": 0.0}
            self.spot_entry = {"BTC": 0.0}
            self.fut_qty = {"BTC": 0.0}
            self.fut_entry = {"BTC": 0.0}
            self.pos = {"BTC": None}
            self.t = 27_292

    legacy = WarmHistoryAgent()

    clone = _clone_genetics_runtime_agent(legacy)

    assert clone is not legacy
    assert list(clone.ph["BTC"]) == list(range(24_000, 42_000))
    assert list(clone.vh["BTC"]) == list(range(18_000))
    assert clone.ph["BTC"].maxlen == 42_000
    assert clone.vh["BTC"].maxlen == 18_000
    assert clone.ph["BTC"] is not legacy.ph["BTC"]
    assert clone.vh["BTC"] is not legacy.vh["BTC"]


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


def test_shadow_agent_signal_records_symbol_local_adaptive_bias_trace():
    from panteon_v2.domain.types import Regime
    from panteon_v2.execution import RiskLimitsConfig
    from panteon_v2.memory import PerformanceMemory
    from panteon_v2.selection import AgentRegistry
    from panteon_v2.shadow import make_market_snapshot
    from panteon_v2.shadow.adapters import GeneticsV2AgentAdapter
    from panteon_v2.app.shadow_tournament import ProductionShadowTournament

    class LegacyAdaptiveAgent:
        def __init__(self):
            self._regime_adaptive_output_bias_enabled = True
            self._regime_adaptive_output_bias_map = {"bearish": 0.90}
            self.last_regime_adaptive_output_bias = {}

        def clone_for_shadow(self):
            return LegacyAdaptiveAgent()

        def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
            raw_regime = str(
                getattr(self, "_regime_adaptive_output_bias_regime_override", "")
                or "mixed_rotational"
            )
            self.last_regime_adaptive_output_bias = {
                "enabled": True,
                "regime": raw_regime,
                "raw_regime": raw_regime,
                "open_output_bias": 0.90 if raw_regime == "bearish" else 0.0,
            }
            return {"BTC": 4 if raw_regime == "bearish" else 0}

    registry = AgentRegistry()
    registry.register(
        GeneticsV2AgentAdapter(
            "GeneticsRegimeAdaptiveBias",
            LegacyAdaptiveAgent(),
        )
    )
    tournament = ProductionShadowTournament(
        registry=registry,
        perf=PerformanceMemory(trade_fraction=1.0),
        risk_config=RiskLimitsConfig(capital_fraction=0.10, min_notional_usd=0.0),
        runtime_event_logs_enabled=False,
    )
    market = make_market_snapshot(
        bar=1,
        prices={"BTC": 100.0},
        regime="mixed_rotational",
        regimes_by_symbol={"BTC": Regime.BEARISH},
    )

    tournament.run_bar(market, players=(), balance_usd=10_000.0)

    signal = tournament.last_agent_signals()["GeneticsRegimeAdaptiveBias"][0]
    assert signal.metadata["regime_adaptive_output_bias"]["raw_regime"] == "bearish"
