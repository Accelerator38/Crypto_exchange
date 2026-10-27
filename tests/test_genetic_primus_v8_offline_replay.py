from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from exia.genetic_primus_v8_clean.collector import collect_public_liquidations
from exia.genetic_primus_v8_clean.status import check_run


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/genetic_primus_v8_clean.json"


class ReplayWebSocket:
    def __init__(self, messages: list[str], clock: "StepClock") -> None:
        self.messages = list(messages)
        self.clock = clock

    async def recv(self) -> str:
        if not self.messages:
            raise AssertionError("collector requested a message past the deterministic replay")
        message = self.messages.pop(0)
        self.clock.advance()
        return message


class StepClock:
    def __init__(self, *, step_seconds: float) -> None:
        self.current = 0.0
        self.step_seconds = step_seconds
        self.steps = 0

    def monotonic(self) -> float:
        return self.current

    def advance(self) -> None:
        self.current += self.step_seconds
        self.steps += 1


class ReplayConnect:
    def __init__(self, messages: list[str], clock: StepClock) -> None:
        self.websocket = ReplayWebSocket(messages, clock)

    def __call__(self, *_args: Any, **_kwargs: Any) -> "ReplayConnect":
        return self

    async def __aenter__(self) -> ReplayWebSocket:
        return self.websocket

    async def __aexit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        return None


def _force_order(*, symbol_type: int) -> dict[str, Any]:
    return {
        "e": "forceOrder",
        "E": 1_700_000_000_100,
        "st": symbol_type,
        "o": {
            "s": "LINKUSD_PERP",
            "S": "SELL",
            "T": 1_700_000_000_000,
            "q": "15",
            "ap": "14.25",
            "X": "FILLED",
        },
    }


def test_offline_replay_seals_and_checks_full_root(tmp_path: Path) -> None:
    valid = _force_order(symbol_type=2)
    messages = [
        json.dumps(_force_order(symbol_type=1)),
        json.dumps({"stream": "!forceOrder@arr", "data": valid}),
        json.dumps(valid),
    ]
    run_dir = tmp_path / "root_offline_replay"
    clock = StepClock(step_seconds=0.02)
    replay = ReplayConnect(messages, clock)
    result = asyncio.run(
        collect_public_liquidations(
            contract_path=CONTRACT,
            run_dir=run_dir,
            duration_seconds=0.05,
            connect_fn=replay,
            monotonic_fn=clock.monotonic,
        )
    )
    assert clock.steps == 3
    assert replay.websocket.messages == []
    assert result["events"] == 1
    assert result["link_events"] == 1
    assert result["duplicate_events"] == 1
    assert result["source_gaps"] == 0
    assert result["orders_enabled"] is False

    checked = check_run(run_dir=run_dir, contract_path=CONTRACT, root=ROOT)
    assert checked["result"] == "SUCCESS"
    assert checked["last_kind"] == "session_stop"
    assert checked["blockers"] == []

    records = [
        json.loads(line)
        for line in (run_dir / "liquidation_tape.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    kinds = [record["kind"] for record in records]
    assert kinds.count("source_message") == 3
    assert kinds.count("liquidation") == 1
    liquidation = next(record for record in records if record["kind"] == "liquidation")
    assert liquidation["payload"]["symbol"] == "LINKUSD_PERP"
    assert liquidation["payload"]["symbol_type"] == 2
    assert (run_dir / "source_snapshot" / "manifest.json").is_file()
