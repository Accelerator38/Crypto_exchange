from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


LONG_ACTIONS = ("FUT_LONG_FULL", "FUT_LONG_HALF", "SPOT_BUY_FULL", "SPOT_BUY_HALF")
SHORT_ACTIONS = ("FUT_SHORT_FULL", "FUT_SHORT_HALF")


def _hash_payload(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _deny_opposite_direction(actor: str, direction: str) -> list[str]:
    actions = LONG_ACTIONS if direction == "SHORT" else SHORT_ACTIONS
    return [f"agent:{actor}|*|{action}|*" for action in actions]


def build_policy(report: Mapping[str, Any], *, selected_key: str) -> dict[str, Any]:
    candidates = {
        str(item.get("key")): item
        for item in report.get("recommended_slices", [])
        if isinstance(item, Mapping)
    }
    if selected_key not in candidates:
        raise ValueError(f"selected slice is not recommended: {selected_key}")

    row = candidates[selected_key]
    actor = str(row["actor_label"])
    symbol = str(row["symbol"]).upper()
    regime = str(row["regime"]).lower()
    direction = str(row["direction"]).upper()
    deny_keys = _deny_opposite_direction(actor, direction)
    if actor == "LiveOIBreakout" and regime != "range_low_vol":
        deny_keys.extend([
            "agent:LiveOIBreakout|*|*|range_low_vol",
            "LiveOIBreakout|*|*|range_low_vol",
            "Solo_LiveOIBreakout|*|*|range_low_vol",
            "ensemble:Solo_LiveOIBreakout|*|*|range_low_vol",
        ])
    policy = {
        "exchange": "BITGET",
        "actor": actor,
        "symbols": [symbol],
        "selected_slice_key": selected_key,
        "regime": regime,
        "direction": direction,
        "terminal_deny_context_signal_keys": sorted(set(deny_keys)),
    }
    policy["policy_sha256"] = _hash_payload(policy)
    return policy


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build Bitget candidate policy artifact.")
    parser.add_argument("--slice-report", required=True)
    parser.add_argument("--selected-key", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    report = json.loads(Path(args.slice_report).read_text(encoding="utf-8"))
    policy = build_policy(report, selected_key=args.selected_key)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({"out": str(out), "policy_sha256": policy["policy_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
