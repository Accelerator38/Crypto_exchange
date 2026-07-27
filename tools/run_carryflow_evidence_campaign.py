"""Run one sealed, read-only CarryFlow evidence campaign observation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

from carryflow_policy import get_carryflow_profile  # noqa: E402
from panteon_v2.policy.evidence_campaign import (  # noqa: E402
    CAMPAIGN_STATUS_SCHEMA_VERSION,
    CampaignCollectionDecision,
    CarryFlowEvidenceCampaign,
    EvidenceCampaignError,
    build_campaign_payload,
    build_root_payload,
    decide_campaign_collection,
    sha256_file,
    validate_evidence_extension_report,
    validate_root_payload,
    validate_root_verdict_payload,
    write_atomic_json,
)
from panteon_v2.policy.evidence_tape import CarryFlowEvidenceTape  # noqa: E402
from panteon_v2.policy.manifest import compute_runtime_fingerprint  # noqa: E402
from panteon_v2.policy.warmup_seed import CarryFlowWarmupSeed  # noqa: E402


CAMPAIGN_LOCK_NAME = "campaign_lock.json"
CAMPAIGN_REPORT_NAME = "screening_report.json"
CAMPAIGN_STATUS_NAME = "campaign_status.json"
ROOT_LOCK_NAME = "root_lock.json"
ROOT_VERDICT_NAME = "root_verdict.json"
COLLECTOR_TOOL = ROOT / "tools" / "collect_bitget_carryflow_tape.py"


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise EvidenceCampaignError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def published_source_revision() -> str:
    dirty = _git("status", "--porcelain")
    if dirty:
        raise EvidenceCampaignError(
            "tracked worktree is dirty; seal and publish runtime before collection"
        )
    head = _git("rev-parse", "HEAD").lower()
    upstream = _git("rev-parse", "@{upstream}").lower()
    if head != upstream:
        raise EvidenceCampaignError(
            "local revision is not the published upstream revision"
        )
    return head


def prepare_campaign(
    campaign_dir: str | Path,
    *,
    exact_report: str | Path | None,
    now: datetime,
) -> CarryFlowEvidenceCampaign:
    directory = Path(campaign_dir).resolve()
    lock_path = directory / CAMPAIGN_LOCK_NAME
    if lock_path.is_file():
        campaign = CarryFlowEvidenceCampaign.from_json(lock_path)
        if exact_report is not None:
            supplied_sha = sha256_file(Path(exact_report).resolve())
            if supplied_sha != campaign.payload["screening_report_sha256"]:
                raise EvidenceCampaignError(
                    "supplied exact report differs from sealed campaign report"
                )
        return campaign
    if exact_report is None:
        raise EvidenceCampaignError(
            "--exact-report is required when creating a new campaign"
        )

    report_source = Path(exact_report).resolve()
    try:
        report = json.loads(report_source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvidenceCampaignError(
            f"exact report not found: {report_source}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise EvidenceCampaignError(f"exact report is invalid JSON: {exc}") from exc
    revision = published_source_revision()
    report_sha = sha256_file(report_source)
    payload = build_campaign_payload(
        report=report,
        report_file=CAMPAIGN_REPORT_NAME,
        report_sha256=report_sha,
        source_revision=revision,
        runtime_fingerprint_sha256=compute_runtime_fingerprint(ROOT),
        created_at=now,
    )
    get_carryflow_profile(payload["strategy_config"]["PROFILE_ID"])
    directory.mkdir(parents=True, exist_ok=True)
    _write_atomic_bytes(directory / CAMPAIGN_REPORT_NAME, report_source.read_bytes())
    write_atomic_json(lock_path, payload)
    return CarryFlowEvidenceCampaign.from_json(lock_path)


def verify_campaign_environment(campaign: CarryFlowEvidenceCampaign) -> None:
    revision = published_source_revision()
    if revision != campaign.payload["source_revision"]:
        raise EvidenceCampaignError(
            "published source revision differs from campaign lock"
        )
    fingerprint = compute_runtime_fingerprint(ROOT)
    if fingerprint != campaign.payload["runtime_fingerprint_sha256"]:
        raise EvidenceCampaignError("runtime fingerprint differs from campaign lock")
    report_path = _local_campaign_path(
        campaign.path.parent,
        str(campaign.payload["screening_report_file"]),
    )
    if not report_path.is_file():
        raise EvidenceCampaignError("sealed campaign report copy is missing")
    if sha256_file(report_path) != campaign.payload["screening_report_sha256"]:
        raise EvidenceCampaignError("sealed campaign report copy was modified")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    selected = validate_evidence_extension_report(report)
    if selected["profile_id"] != campaign.profile_id:
        raise EvidenceCampaignError("campaign/report profile mismatch")
    get_carryflow_profile(campaign.profile_id)


def load_campaign_roots(
    campaign: CarryFlowEvidenceCampaign,
) -> list[dict[str, Any]]:
    roots_dir = campaign.path.parent / "roots"
    result: list[dict[str, Any]] = []
    if not roots_dir.is_dir():
        return result
    for lock_path in sorted(roots_dir.glob(f"*/{ROOT_LOCK_NAME}")):
        raw = json.loads(lock_path.read_text(encoding="utf-8"))
        payload = validate_root_payload(raw, campaign=campaign)
        root_dir = lock_path.parent.resolve()
        tape_path = root_dir / payload["evidence_tape_file"]
        seed_path = root_dir / payload["warmup_seed_file"]
        tape = None
        seed = None
        verdict = None
        if tape_path.is_file():
            tape = CarryFlowEvidenceTape.from_jsonl(
                tape_path,
                expected_symbols=campaign.symbols,
            )
            _validate_tape_contract(campaign, payload, tape)
            if not seed_path.is_file():
                raise EvidenceCampaignError(
                    f"campaign root has evidence but no warm-up seed: {root_dir}"
                )
            seed = CarryFlowWarmupSeed.from_json(
                seed_path,
                expected_symbols=campaign.symbols,
            )
            seed.validate_for_tape(tape)
        verdict_path = root_dir / ROOT_VERDICT_NAME
        if verdict_path.is_file():
            if tape is None:
                raise EvidenceCampaignError(
                    f"terminal root has no evidence tape: {root_dir}"
                )
            verdict = validate_root_verdict_payload(
                json.loads(verdict_path.read_text(encoding="utf-8")),
                campaign=campaign,
                root_payload=payload,
            )
            description = tape.describe()
            if verdict["evidence_tape_sha256"] != sha256_file(tape_path):
                raise EvidenceCampaignError("terminal root tape SHA-256 changed")
            if verdict["evidence_head_sha256"] != description["head_sha256"]:
                raise EvidenceCampaignError("terminal root evidence head changed")
            if int(verdict["samples"]) != int(description["samples"]):
                raise EvidenceCampaignError("terminal root sample count changed")
            if int(verdict["complete_samples"]) != int(
                description["complete_samples"]
            ):
                raise EvidenceCampaignError(
                    "terminal root complete-sample count changed"
                )
        result.append(
            {
                "lock_path": lock_path.resolve(),
                "root_dir": root_dir,
                "payload": payload,
                "tape": tape,
                "seed": seed,
                "verdict": verdict,
            }
        )
    indexes = [int(row["payload"]["root_index"]) for row in result]
    if indexes != list(range(1, len(indexes) + 1)):
        raise EvidenceCampaignError("campaign root indexes are not contiguous")
    terminal_indexes = [
        index for index, row in enumerate(result) if row["verdict"] is not None
    ]
    if terminal_indexes and terminal_indexes != [len(result) - 1]:
        raise EvidenceCampaignError("campaign contains a root after terminal verdict")
    return result


def run_campaign_once(
    campaign_dir: str | Path,
    *,
    exact_report: str | Path | None = None,
    now: datetime | None = None,
) -> tuple[dict[str, Any], int]:
    observed_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    campaign = prepare_campaign(
        campaign_dir,
        exact_report=exact_report,
        now=observed_at,
    )
    roots = load_campaign_roots(campaign)
    latest = roots[-1] if roots else None
    latest_tape = latest["tape"] if latest else None
    latest_close = (
        int(latest_tape.samples[-1]["bar_close_timestamp_ms"])
        if latest_tape is not None
        else None
    )
    if latest is not None and latest["verdict"] is not None:
        decision = CampaignCollectionDecision(
            action="terminal",
            reason=str(latest["verdict"]["reason"]),
            current_bar_close_timestamp_ms=int(latest_close or 0),
            next_collection_at="",
        )
        summary = build_campaign_status(
            campaign,
            roots,
            run_state="completed_negative",
            decision=decision,
        )
        write_atomic_json(campaign.path.parent / CAMPAIGN_STATUS_NAME, summary)
        return summary, 0

    verify_campaign_environment(campaign)
    decision = decide_campaign_collection(
        now=observed_at,
        bar_interval_seconds=int(campaign.payload["bar_interval_seconds"]),
        alignment_delay_seconds=int(campaign.payload["alignment_delay_seconds"]),
        max_bar_close_lag_seconds=int(
            campaign.payload["max_bar_close_lag_seconds"]
        ),
        latest_bar_close_timestamp_ms=latest_close,
    )

    if latest is not None and latest_tape is None:
        intended = int(
            latest["payload"]["intended_first_bar_close_timestamp_ms"]
        )
        if decision.action == "new_root":
            if decision.current_bar_close_timestamp_ms == intended:
                decision = type(decision)(
                    action="resume_pending_root",
                    reason="retry_uncollected_root_in_same_window",
                    current_bar_close_timestamp_ms=(
                        decision.current_bar_close_timestamp_ms
                    ),
                    next_collection_at=decision.next_collection_at,
                )
            elif decision.current_bar_close_timestamp_ms > intended:
                decision = type(decision)(
                    action="new_root",
                    reason="real_bar_gap",
                    current_bar_close_timestamp_ms=(
                        decision.current_bar_close_timestamp_ms
                    ),
                    next_collection_at=decision.next_collection_at,
                )

    if decision.action in {"wait", "idle"}:
        summary = build_campaign_status(
            campaign,
            roots,
            run_state=(
                "scheduled_idle" if decision.action == "idle" else "waiting"
            ),
            decision=decision,
        )
        write_atomic_json(campaign.path.parent / CAMPAIGN_STATUS_NAME, summary)
        return summary, 0

    if decision.action == "resume_root":
        if latest is None or latest_tape is None:
            raise EvidenceCampaignError("resume requested without an evidence root")
        selected_root = latest
    elif decision.action == "resume_pending_root":
        if latest is None:
            raise EvidenceCampaignError("pending root disappeared")
        selected_root = latest
    elif decision.action == "new_root":
        selected_root = create_campaign_root(
            campaign,
            roots,
            now=observed_at,
            intended_close_ms=decision.current_bar_close_timestamp_ms,
            start_reason=decision.reason,
        )
        roots.append(selected_root)
    else:
        raise EvidenceCampaignError(f"unsupported collection action: {decision.action}")

    command = collector_command(campaign, selected_root)
    close_ms = decision.current_bar_close_timestamp_ms
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=600,
    )
    (selected_root["root_dir"] / f"collector_stdout_{close_ms}.log").write_text(
        completed.stdout or "",
        encoding="utf-8",
    )
    (selected_root["root_dir"] / f"collector_stderr_{close_ms}.log").write_text(
        completed.stderr or "",
        encoding="utf-8",
    )
    if completed.returncode not in {0, 2}:
        summary = build_campaign_status(
            campaign,
            load_campaign_roots(campaign),
            run_state="failed",
            decision=decision,
            error=(completed.stderr or completed.stdout).strip(),
        )
        write_atomic_json(campaign.path.parent / CAMPAIGN_STATUS_NAME, summary)
        raise EvidenceCampaignError(
            f"read-only collector failed with exit code {completed.returncode}"
        )

    roots = load_campaign_roots(campaign)
    collected_tape = roots[-1]["tape"]
    if collected_tape is None:
        raise EvidenceCampaignError("collector returned without an evidence tape")
    actual_close = int(collected_tape.samples[-1]["bar_close_timestamp_ms"])
    if actual_close != close_ms:
        raise EvidenceCampaignError("collector appended an unexpected bar")
    last_complete = bool(collected_tape.samples[-1]["sample_complete"])
    summary = build_campaign_status(
        campaign,
        roots,
        run_state=("scheduled_idle" if last_complete else "degraded_sample"),
        decision=decision,
    )
    write_atomic_json(campaign.path.parent / CAMPAIGN_STATUS_NAME, summary)
    return summary, 0 if last_complete else 2


def create_campaign_root(
    campaign: CarryFlowEvidenceCampaign,
    roots: Sequence[Mapping[str, Any]],
    *,
    now: datetime,
    intended_close_ms: int,
    start_reason: str,
) -> dict[str, Any]:
    previous = roots[-1] if roots else None
    previous_tape = previous["tape"] if previous else None
    payload = build_root_payload(
        campaign=campaign,
        root_index=len(roots) + 1,
        created_at=now,
        intended_first_bar_close_timestamp_ms=intended_close_ms,
        start_reason=start_reason,
        previous_root_id=(str(previous["payload"]["root_id"]) if previous else ""),
        previous_root_head_sha256=(
            previous_tape.head_sha256 if previous_tape is not None else ""
        ),
    )
    root_dir = (
        campaign.path.parent
        / "roots"
        / f"root_{payload['root_index']:03d}_{now:%Y%m%dT%H%M%SZ}"
    )
    if root_dir.exists():
        raise EvidenceCampaignError(f"campaign root already exists: {root_dir}")
    root_dir.mkdir(parents=True)
    lock_path = root_dir / ROOT_LOCK_NAME
    write_atomic_json(lock_path, payload)
    return {
        "lock_path": lock_path.resolve(),
        "root_dir": root_dir.resolve(),
        "payload": payload,
        "tape": None,
        "seed": None,
    }


def collector_command(
    campaign: CarryFlowEvidenceCampaign,
    root: Mapping[str, Any],
) -> list[str]:
    payload = root["payload"]
    root_dir = Path(root["root_dir"])
    return [
        sys.executable,
        str(COLLECTOR_TOOL),
        "--resume",
        "--collector-run-id",
        str(payload["root_id"]),
        "--output",
        str(root_dir / payload["evidence_tape_file"]),
        "--status-path",
        str(root_dir / payload["collector_status_file"]),
        "--warmup-seed-output",
        str(root_dir / payload["warmup_seed_file"]),
        "--samples",
        "1",
        "--symbols",
        ",".join(campaign.symbols),
        "--bar-interval-sec",
        str(campaign.payload["bar_interval_seconds"]),
        "--max-bar-close-lag-sec",
        str(campaign.payload["max_bar_close_lag_seconds"]),
        "--max-derivatives-age-sec",
        str(campaign.payload["max_derivatives_age_seconds"]),
        "--alignment-delay-sec",
        str(campaign.payload["alignment_delay_seconds"]),
        "--warmup-bars",
        str(campaign.payload["warmup_bars"]),
    ]


def build_campaign_status(
    campaign: CarryFlowEvidenceCampaign,
    roots: Sequence[Mapping[str, Any]],
    *,
    run_state: str,
    decision: Any,
    error: str = "",
) -> dict[str, Any]:
    root_rows = []
    symbol_observations = 0
    complete_symbol_observations = 0
    for root in roots:
        tape = root["tape"]
        row = {
            "root_index": root["payload"]["root_index"],
            "root_id": root["payload"]["root_id"],
            "start_reason": root["payload"]["start_reason"],
            "root_sha256": root["payload"]["root_sha256"],
            "tape_present": tape is not None,
            "samples": 0,
            "complete_samples": 0,
            "context_coverage_pct": 0.0,
            "head_sha256": "",
            "last_bar_close_timestamp_ms": None,
            "terminal": root["verdict"] is not None,
            "verdict": (
                str(root["verdict"]["verdict"])
                if root["verdict"] is not None
                else ""
            ),
            "verdict_reason": (
                str(root["verdict"]["reason"])
                if root["verdict"] is not None
                else ""
            ),
            "continuation_allowed": (
                bool(root["verdict"]["continuation_allowed"])
                if root["verdict"] is not None
                else True
            ),
        }
        if tape is not None:
            description = tape.describe()
            row.update(
                {
                    "samples": description["samples"],
                    "complete_samples": description["complete_samples"],
                    "context_coverage_pct": description["context_coverage_pct"],
                    "head_sha256": description["head_sha256"],
                    "last_bar_close_timestamp_ms": description[
                        "last_bar_close_timestamp_ms"
                    ],
                }
            )
            symbol_observations += int(description["symbol_observations"])
            complete_symbol_observations += int(
                description["complete_symbol_observations"]
            )
        root_rows.append(row)
    samples = sum(int(row["samples"]) for row in root_rows)
    complete_samples = sum(int(row["complete_samples"]) for row in root_rows)
    return {
        "schema_version": CAMPAIGN_STATUS_SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "campaign_id": campaign.payload["campaign_id"],
        "campaign_sha256": campaign.campaign_sha256,
        "profile_id": campaign.profile_id,
        "source_revision": campaign.payload["source_revision"],
        "runtime_fingerprint_sha256": campaign.payload[
            "runtime_fingerprint_sha256"
        ],
        "run_state": run_state,
        "decision": {
            "action": decision.action,
            "reason": decision.reason,
            "current_bar_close_timestamp_ms": (
                decision.current_bar_close_timestamp_ms
            ),
            "next_collection_at": decision.next_collection_at,
        },
        "orders_enabled": False,
        "promotion_authority": False,
        "segments_concatenated": False,
        "root_count": len(root_rows),
        "roots_with_evidence": sum(row["tape_present"] for row in root_rows),
        "samples": samples,
        "complete_samples": complete_samples,
        "incomplete_samples": samples - complete_samples,
        "context_coverage_pct": (
            complete_symbol_observations / symbol_observations * 100.0
            if symbol_observations
            else 0.0
        ),
        "roots": root_rows,
        "error": str(error),
    }


def _validate_tape_contract(
    campaign: CarryFlowEvidenceCampaign,
    root_payload: Mapping[str, Any],
    tape: CarryFlowEvidenceTape,
) -> None:
    if tape.collector_run_id != root_payload["root_id"]:
        raise EvidenceCampaignError("campaign root/tape collector ID mismatch")
    if tape.source_revision != campaign.payload["source_revision"]:
        raise EvidenceCampaignError("campaign root/tape source revision mismatch")
    if tuple(tape.symbols) != campaign.symbols:
        raise EvidenceCampaignError("campaign root/tape symbols mismatch")
    if tape.bar_interval_seconds != campaign.payload["bar_interval_seconds"]:
        raise EvidenceCampaignError("campaign root/tape cadence mismatch")
    if (
        tape.max_bar_close_lag_seconds
        != campaign.payload["max_bar_close_lag_seconds"]
    ):
        raise EvidenceCampaignError("campaign root/tape lag limit mismatch")
    if (
        tape.max_derivatives_age_seconds
        != campaign.payload["max_derivatives_age_seconds"]
    ):
        raise EvidenceCampaignError("campaign root/tape age limit mismatch")
    first_close = int(tape.samples[0]["bar_close_timestamp_ms"])
    if first_close != root_payload["intended_first_bar_close_timestamp_ms"]:
        raise EvidenceCampaignError("campaign root/tape first bar mismatch")


def _local_campaign_path(directory: Path, relative: str) -> Path:
    candidate = (directory / relative).resolve()
    try:
        candidate.relative_to(directory.resolve())
    except ValueError as exc:
        raise EvidenceCampaignError("campaign file escapes its directory") from exc
    return candidate


def _write_atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect at most one read-only hourly observation for a sealed "
            "CarryFlow evidence campaign."
        )
    )
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument(
        "--exact-report",
        help="Required only for the first invocation that seals the campaign.",
    )
    args = parser.parse_args(argv)
    try:
        summary, return_code = run_campaign_once(
            args.campaign_dir,
            exact_report=args.exact_report,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "schema_version": CAMPAIGN_STATUS_SCHEMA_VERSION,
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
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
