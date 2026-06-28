from __future__ import annotations

from panteon_v2.app import startup
from panteon_v2.domain.types import Action
from panteon_v2.execution.exchange import FakeExchange


class _StartupProbeAgent:
    label = "StartupProbe"

    def act(self, market):
        return {symbol: Action.HOLD for symbol in market.prices}


def test_compact_causal_entry_include_labels_defaults_to_live_signal_sources(monkeypatch):
    monkeypatch.setattr(startup, "_load_exchange_settings", lambda exchange: {})

    labels = startup._resolve_compact_causal_entry_include_labels("MEXC")

    assert "CarryFlowAgentV2" in labels
    assert "MomentumScalper" in labels
    assert "LiveVolCompress" in labels
    assert "LiveCrashHunter" in labels


def test_compact_causal_entry_include_labels_adds_exchange_scoped_settings(monkeypatch):
    monkeypatch.setattr(
        startup,
        "_load_exchange_settings",
        lambda exchange: {
            "bitget_v2_compact_causal_entry_include_labels": "CustomActor,agent:CustomActor",
        },
    )

    labels = startup._resolve_compact_causal_entry_include_labels("BITGET")

    assert labels[-2:] == ("CustomActor", "agent:CustomActor")
    assert "CarryFlowAgentV2" in labels


def test_flash_component_memory_resolves_exchange_scoped_settings(
    monkeypatch,
    tmp_path,
):
    memory_path = tmp_path / "component_memory.jsonl"
    memory_path.write_text(
        (
            '{"actor_label":"LiveVolCompress","symbol":"BTC","regime":"*",'
            '"action":"FUT_SHORT_HALF","bar":3,"closed_trades":2,'
            '"expectancy":0.25,"pnl_lcb":0.20}\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("MEXC_PANTEON_FLASH_COMPONENT_MEMORY_JSONL", raising=False)
    monkeypatch.delenv("PANTEON_FLASH_COMPONENT_MEMORY_JSONL", raising=False)
    monkeypatch.setattr(
        startup,
        "_load_exchange_settings",
        lambda exchange: {
            "mexc_v2_flash_component_memory_jsonl": str(memory_path),
        },
    )

    memory = startup._resolve_flash_component_memory("MEXC")

    assert memory is not None
    stat = memory.best_prior(
        "LiveVolCompress",
        symbol="BTC",
        regime="neutral",
        action="FUT_SHORT_HALF",
        bar=4,
    )
    assert stat is not None
    assert stat.expectancy == 0.25


def test_futures_signal_fixes_resolve_exchange_scoped_settings(monkeypatch):
    monkeypatch.delenv("MEXC_PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED", raising=False)
    monkeypatch.delenv("PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED", raising=False)
    monkeypatch.setattr(
        startup,
        "_load_exchange_settings",
        lambda exchange: {
            "v2_futures_signal_fixes_enabled": "off",
            "mexc_v2_futures_signal_fixes_enabled": "on",
        },
    )

    assert startup._resolve_futures_signal_fixes_enabled("MEXC") is True
    assert startup._resolve_futures_signal_fixes_enabled("BITGET") is False


def test_futures_signal_fixes_resolve_exchange_scoped_env(monkeypatch):
    monkeypatch.setattr(startup, "_load_exchange_settings", lambda exchange: {})
    monkeypatch.setenv("PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED", "off")
    monkeypatch.setenv("BITGET_PANTEON_V2_FUTURES_SIGNAL_FIXES_ENABLED", "on")

    assert startup._resolve_futures_signal_fixes_enabled("MEXC") is False
    assert startup._resolve_futures_signal_fixes_enabled("BITGET") is True


def test_start_production_passes_futures_signal_fixes_to_v1_registration(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, object] = {}

    def register_probe(registry, **kwargs):
        captured.update(kwargs)
        registry.register(_StartupProbeAgent())
        return [_StartupProbeAgent.label]

    monkeypatch.setattr(
        startup,
        "_load_exchange_settings",
        lambda exchange: {"mexc_v2_futures_signal_fixes_enabled": "on"},
    )
    monkeypatch.setattr(
        startup,
        "resolve_exchange",
        lambda *a, **k: FakeExchange(name="MEXC"),
    )
    monkeypatch.setattr(startup, "register_all_v1_agents", register_probe)

    rc = startup.start_production(
        exchange="MEXC",
        mode="paper",
        max_bars=0,
        use_v1_bridge=False,
        results_root=str(tmp_path),
        sleep_between_polls_sec=0.0,
        include_genetics=False,
    )

    assert rc == 0
    assert captured["futures_replay_signal_fixes_enabled"] is True
