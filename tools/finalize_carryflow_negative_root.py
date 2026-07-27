"""Seal a completed CarryFlow evidence root as a terminal negative experiment."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RUNTIME = SRC / "panteon_runtime"
for path in (SRC, RUNTIME):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from panteon_v2.policy.evidence_campaign import (  # noqa: E402
    CarryFlowEvidenceCampaign,
    EvidenceCampaignError,
    build_negative_root_verdict,
    sha256_file,
    validate_root_payload,
    validate_root_verdict_payload,
    write_atomic_json,
)
from panteon_v2.policy.evidence_tape import CarryFlowEvidenceTape  # noqa: E402
from panteon_v2.policy.warmup_seed import CarryFlowWarmupSeed  # noqa: E402


CAMPAIGN_LOCK_NAME = "campaign_lock.json"
ROOT_LOCK_NAME = "root_lock.json"
ROOT_VERDICT_NAME = "root_verdict.json"


def finalize_negative_root(
    *,
    campaign_dir: str | Path,
    root_dir: str | Path,
    replay_summary_path: str | Path,
    recorded_at: datetime | None = None,
) -> dict[str, Any]:
    campaign_path = Path(campaign_dir).resolve()
    selected_root_dir = Path(root_dir).resolve()
    replay_path = Path(replay_summary_path).resolve()
    campaign = CarryFlowEvidenceCampaign.from_json(
        campaign_path / CAMPAIGN_LOCK_NAME
    )

    roots_dir = (campaign_path / "roots").resolve()
    try:
        selected_root_dir.relative_to(roots_dir)
    except ValueError as exc:
        raise EvidenceCampaignError("root is outside the campaign") from exc
    root_locks = sorted(roots_dir.glob(f"*/{ROOT_LOCK_NAME}"))
    if not root_locks or selected_root_dir != root_locks[-1].parent.resolve():
        raise EvidenceCampaignError("only the latest campaign root can be finalized")

    root_payload = validate_root_payload(
        _read_json(selected_root_dir / ROOT_LOCK_NAME),
        campaign=campaign,
    )
    tape_path = selected_root_dir / str(root_payload["evidence_tape_file"])
    seed_path = selected_root_dir / str(root_payload["warmup_seed_file"])
    status_path = selected_root_dir / str(root_payload["collector_status_file"])
    tape = CarryFlowEvidenceTape.from_jsonl(
        tape_path,
        expected_symbols=campaign.symbols,
    )
    seed = CarryFlowWarmupSeed.from_json(
        seed_path,
        expected_symbols=campaign.symbols,
    )
    seed.validate_for_tape(tape)
    collector_status = _read_json(status_path)
    if str(collector_status.get("run_state") or "").lower() in {
        "running",
        "collecting",
        "starting",
    }:
        raise EvidenceCampaignError("collector is still active")

    replay_summary = _read_json(replay_path)
    verdict_path = selected_root_dir / ROOT_VERDICT_NAME
    if verdict_path.exists():
        existing = validate_root_verdict_payload(
            _read_json(verdict_path),
            campaign=campaign,
            root_payload=root_payload,
        )
        if existing["evidence_tape_sha256"] != sha256_file(tape_path):
            raise EvidenceCampaignError("existing verdict tape SHA-256 mismatch")
        if existing["replay_summary_sha256"] != sha256_file(replay_path):
            raise EvidenceCampaignError("existing verdict replay SHA-256 mismatch")
        return existing

    verdict = build_negative_root_verdict(
        campaign=campaign,
        root_payload=root_payload,
        tape_description=tape.describe(),
        evidence_tape_sha256=sha256_file(tape_path),
        replay_summary=replay_summary,
        replay_summary_sha256=sha256_file(replay_path),
        recorded_at=recorded_at or datetime.now(timezone.utc),
    )
    write_atomic_json(verdict_path, verdict)
    return verdict


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise EvidenceCampaignError(f"required file is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvidenceCampaignError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise EvidenceCampaignError(f"JSON object required: {path}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Write an immutable terminal rejection for a completed negative "
            "CarryFlow evidence root."
        )
    )
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--replay-summary", required=True)
    args = parser.parse_args(argv)
    try:
        verdict = finalize_negative_root(
            campaign_dir=args.campaign_dir,
            root_dir=args.root_dir,
            replay_summary_path=args.replay_summary,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "run_state": "failed",
                    "orders_enabled": False,
                    "promotion_authority": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
