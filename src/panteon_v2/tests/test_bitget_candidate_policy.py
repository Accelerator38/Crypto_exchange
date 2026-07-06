from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = ROOT / "tools" / "build_bitget_candidate_policy.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("build_bitget_candidate_policy", TOOL_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_policy_for_single_slice_denies_unselected_contexts():
    tool = _load_tool()
    report = {
        "recommended_slices": [
            {
                "key": "CarryFlowAgentV2|ETH|bearish|SHORT",
                "actor_label": "CarryFlowAgentV2",
                "symbol": "ETH",
                "regime": "bearish",
                "direction": "SHORT",
            }
        ]
    }

    policy = tool.build_policy(
        report,
        selected_key="CarryFlowAgentV2|ETH|bearish|SHORT",
    )

    assert policy["actor"] == "CarryFlowAgentV2"
    assert policy["symbols"] == ["ETH"]
    assert policy["selected_slice_key"] == "CarryFlowAgentV2|ETH|bearish|SHORT"
    assert "agent:CarryFlowAgentV2|*|FUT_LONG_FULL|*" in policy[
        "terminal_deny_context_signal_keys"
    ]
    assert "agent:CarryFlowAgentV2|ETH|FUT_SHORT_FULL|bearish" not in policy[
        "terminal_deny_context_signal_keys"
    ]
    assert policy["policy_sha256"]
