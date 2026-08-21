from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_exia_regime_agent_archive_v1 import validate_spec  # noqa: E402


SPEC = ROOT / "configs" / "research_archive" / "exia_regime_agent_archive_v1.json"


def test_archive_is_fixed_safe_and_regime_balanced() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    validate_spec(spec)
    counts = {
        regime: sum(row["regime"] == regime for row in spec["agents"])
        for regime in ("bullish", "bearish", "neutral")
    }
    assert counts == {"bullish": 3, "bearish": 3, "neutral": 4}
    assert set(spec["safety"].values()) == {False}


def test_every_archived_source_symbol_exists_once() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for row in spec["agents"]:
        source = ROOT / row["source"]["path"]
        tree = ast.parse(source.read_text(encoding="utf-8"))
        matches = [
            node
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == row["source"]["symbol"]
        ]
        assert len(matches) == 1, row["agent_id"]


def test_terminal_neutral_compression_is_reference_only() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    live_vol = next(row for row in spec["agents"] if row["agent_id"] == "LiveVolCompress")
    assert live_vol["archive_role"] == "terminal_negative_reference"
    assert "закрыта" in live_vol["limitations_ru"]
