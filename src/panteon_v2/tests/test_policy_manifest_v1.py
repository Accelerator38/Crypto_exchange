from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from panteon_v2.policy import (
    CandidateSignal,
    DecisionOutcome,
    DecisionReason,
    Direction,
    ManifestError,
    PolicyExecutorV1,
    PolicyTarget,
    RuntimeContext,
    compute_runtime_fingerprint,
    load_policy_manifest,
    prepare_manifest_payload,
    seal_manifest_payload,
    validate_policy_manifest,
)
from panteon_v2.policy.manifest import (
    RUNTIME_FINGERPRINT_PATHS,
    parse_manifest_payload,
    sha256_file,
)


NOW = datetime(2026, 7, 12, 12, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[3]
COMPILER_PATH = ROOT / "tools" / "compile_policy_manifest_v1.py"


def _iso(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _load_compiler():
    spec = importlib.util.spec_from_file_location("compile_policy_manifest_v1", COMPILER_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evidence_row(tmp_path: Path, policy_id: str, kind: str) -> dict[str, object]:
    evidence_dir = tmp_path / "evidence"
    source_dir = tmp_path / "source_reports"
    evidence_dir.mkdir(exist_ok=True)
    source_dir.mkdir(exist_ok=True)
    artifact = evidence_dir / f"{kind}.json"
    source = source_dir / f"{kind}.json"
    source.write_text(
        json.dumps({"policy_id": policy_id, "kind": kind, "raw": True}),
        encoding="utf-8",
    )
    is_sanity = kind == "sanity"
    is_short = kind == "short_paper_canary"
    row: dict[str, object] = {
        "kind": kind,
        "candidate_key": policy_id,
        "exchange": "BITGET",
        "artifact_path": str(artifact.relative_to(tmp_path)).replace("\\", "/"),
        "artifact_sha256": "AUTO",
        "generated_at": _iso(NOW - timedelta(minutes=20)),
        "data_as_of": _iso(NOW - timedelta(hours=1)),
        "passed": True,
        "signals": 0 if is_sanity else (2 if is_short else 30),
        "orders": 0 if is_sanity else (1 if is_short else 25),
        "fills": 0 if is_sanity else (1 if is_short else 20),
        "closed_trades": 0 if is_sanity else (1 if is_short else 10),
        "expectancy_after_costs_usd": 0.0 if is_sanity else 0.02,
        "expectancy_lcb_usd": 0.0 if is_sanity or is_short else 0.005,
        "max_drawdown_pct": 0.0 if is_sanity else 5.0,
        "mean_cost_bps": 0.0 if is_sanity else 8.0,
        "direction_collapse": False,
        "regime_collapse": False,
    }
    artifact.write_text(
        json.dumps(
            {
                "schema_version": "panteon.policy_evidence.v1",
                "kind": row["kind"],
                "candidate_key": row["candidate_key"],
                "exchange": row["exchange"],
                "generated_at": row["generated_at"],
                "data_as_of": row["data_as_of"],
                "passed": row["passed"],
                "metrics": {
                    key: row[key]
                    for key in (
                        "signals",
                        "orders",
                        "fills",
                        "closed_trades",
                        "expectancy_after_costs_usd",
                        "expectancy_lcb_usd",
                        "max_drawdown_pct",
                        "mean_cost_bps",
                        "direction_collapse",
                        "regime_collapse",
                    )
                },
                "source_artifacts": [
                    {
                        "path": str(source.relative_to(tmp_path)).replace("\\", "/"),
                        "sha256": sha256_file(source),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return row


def _draft(tmp_path: Path, *, target: str = "paper") -> dict[str, object]:
    policy_id = "carryflow-btc-neutral-short-v1"
    kinds = ["validation", "oos", "cost_stress", "sanity"]
    if target == "micro_live":
        kinds.extend(["short_paper_canary", "extended_paper_canary"])
    return {
        "schema_version": "panteon.policy.v1",
        "policy_id": policy_id,
        "target": target,
        "exchange": "BITGET",
        "actor": "CarryFlowAgentV2",
        "actor_config": {
            "ALLOW_LONG": False,
            "ALLOW_SHORT": True,
        },
        "signal_model": {
            "feature": "diagnostic.edge",
            "intercept_bps": 0.0,
            "slope_bps_per_unit": 20.0,
            "lcb_haircut_bps": 2.0,
            "min_feature_value": 0.1,
            "max_expected_move_bps": 100.0,
        },
        "data": {
            "bar_interval_seconds": 3600,
            "cadence_tolerance_seconds": 0,
            "max_bar_close_lag_seconds": 120,
            "max_derivatives_age_seconds": 1200,
            "required_context_coverage_pct": 95.0,
        },
        "created_at": _iso(NOW - timedelta(minutes=5)),
        "expires_at": _iso(NOW + (timedelta(hours=12) if target == "micro_live" else timedelta(days=2))),
        "source_revision": "d863585",
        "runtime_fingerprint_sha256": "a" * 64,
        "costs": {
            "round_trip_fee_bps": 8.0,
            "slippage_bps": 2.0,
            "safety_buffer_bps": 4.0,
        },
        "risk": {
            "capital_fraction": 0.01,
            "max_notional_usd": 10.0,
            "max_open_positions": 1,
            "max_daily_loss_usd": 5.0,
            "stop_loss_pct": 2.0,
            "max_holding_minutes": 180,
            "max_signal_age_seconds": 120,
        },
        "rules": [
            {
                "symbol": "BTC",
                "regime": "neutral",
                "direction": "SHORT",
                "min_expected_move_bps": 15.0,
                "max_spread_bps": 5.0,
                "max_slippage_bps": 4.0,
                "min_regime_confidence": 0.65,
                "risk_mult": 0.5,
            }
        ],
        "evidence": [_evidence_row(tmp_path, policy_id, kind) for kind in kinds],
    }


def _write_manifest(tmp_path: Path, *, target: str = "paper"):
    payload = prepare_manifest_payload(_draft(tmp_path, target=target), project_root=tmp_path)
    sealed = seal_manifest_payload(payload)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(sealed), encoding="utf-8")
    loaded = load_policy_manifest(
        path,
        project_root=tmp_path,
        expected_sha256=sealed["manifest_sha256"],
        now=NOW,
        verify_runtime_fingerprint=False,
        required_target=target,
        required_exchange="BITGET",
    )
    return path, sealed, loaded


def _signal(**overrides) -> CandidateSignal:
    values = {
        "signal_id": "sig-1",
        "bar": 100,
        "actor": "CarryFlowAgentV2",
        "symbol": "BTC/USDT",
        "regime": "neutral",
        "direction": Direction.SHORT,
        "expected_move_bps": 20.0,
        "price": 60_000.0,
        "generated_at": NOW - timedelta(seconds=10),
    }
    values.update(overrides)
    return CandidateSignal(**values)


def _context(**overrides) -> RuntimeContext:
    values = {
        "exchange": "BITGET",
        "mode": PolicyTarget.PAPER,
        "now": NOW,
        "spread_bps": 2.0,
        "estimated_slippage_bps": 2.0,
        "regime_confidence": 0.8,
        "exchange_healthy": True,
        "kill_switch_active": False,
        "open_positions": 0,
        "daily_loss_usd": 0.0,
    }
    values.update(overrides)
    return RuntimeContext(**values)


def test_valid_paper_manifest_requires_all_robust_evidence(tmp_path):
    prepared = prepare_manifest_payload(_draft(tmp_path), project_root=tmp_path)
    manifest = parse_manifest_payload(seal_manifest_payload(prepared))

    validation = validate_policy_manifest(manifest, project_root=tmp_path, now=NOW)

    assert validation.passed is True
    assert validation.reasons == ()


def test_replay_manifest_is_research_only_and_needs_no_promotion_evidence(tmp_path):
    draft = _draft(tmp_path)
    draft["target"] = "replay"
    draft["evidence"] = []
    draft["expires_at"] = _iso(NOW + timedelta(days=14))
    prepared = prepare_manifest_payload(draft, project_root=tmp_path)
    manifest = parse_manifest_payload(seal_manifest_payload(prepared))

    validation = validate_policy_manifest(manifest, project_root=tmp_path, now=NOW)

    assert validation.passed is True
    assert manifest.target == PolicyTarget.REPLAY
    assert manifest.data.bar_interval_seconds == 3600
    assert manifest.signal_model.estimate_bps(1.0) == 18.0
    assert manifest.signal_model.estimate_bps(0.01) is None

    path = tmp_path / "replay.json"
    path.write_text(json.dumps(manifest.as_dict()), encoding="utf-8")
    with pytest.raises(ManifestError) as error:
        load_policy_manifest(
            path,
            project_root=tmp_path,
            expected_sha256=manifest.manifest_sha256,
            now=NOW,
            verify_runtime_fingerprint=False,
            required_target=PolicyTarget.MICRO_LIVE,
        )
    assert error.value.reasons == ("manifest.target_mismatch",)


def test_manifest_requires_explicit_valid_market_data_contract(tmp_path):
    draft = _draft(tmp_path)
    draft.pop("data")
    with pytest.raises(ManifestError, match="manifest.field_missing:data"):
        seal_manifest_payload(draft)

    draft = _draft(tmp_path)
    draft["target"] = "replay"
    draft["evidence"] = []
    draft["data"]["bar_interval_seconds"] = 45
    with pytest.raises(ManifestError, match="bar_interval_seconds_invalid"):
        seal_manifest_payload(draft)


def test_authoritative_load_pins_dirty_worktree_runtime_bytes(tmp_path):
    draft = _draft(tmp_path)
    draft["target"] = "replay"
    draft["evidence"] = []
    draft["runtime_fingerprint_sha256"] = compute_runtime_fingerprint(ROOT)
    sealed = seal_manifest_payload(draft)
    path = tmp_path / "runtime_pinned.json"
    path.write_text(json.dumps(sealed), encoding="utf-8")

    loaded = load_policy_manifest(
        path,
        project_root=ROOT,
        expected_sha256=sealed["manifest_sha256"],
        now=NOW,
    )
    assert loaded.manifest.runtime_fingerprint_sha256 == compute_runtime_fingerprint(ROOT)

    draft["runtime_fingerprint_sha256"] = "0" * 64
    tampered = seal_manifest_payload(draft)
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ManifestError, match="runtime_fingerprint_mismatch"):
        load_policy_manifest(
            path,
            project_root=ROOT,
            expected_sha256=tampered["manifest_sha256"],
            now=NOW,
        )


def test_runtime_fingerprint_covers_live_policy_wiring():
    required = {
        "Start_panteon.py",
        "src/panteon_v2/app/bitget_adapter.py",
        "src/panteon_v2/app/main_loop.py",
        "src/panteon_v2/app/policy_runtime.py",
        "src/panteon_v2/app/startup.py",
        "src/panteon_v2/app/v1_bridge_runner.py",
    }

    assert required.issubset(set(RUNTIME_FINGERPRINT_PATHS))


def test_manifest_hard_blocks_missing_oos_and_zero_costs(tmp_path):
    draft = _draft(tmp_path)
    draft["evidence"] = [
        row for row in draft["evidence"] if row["kind"] != "oos"  # type: ignore[index]
    ]
    validation_row = next(
        row for row in draft["evidence"] if row["kind"] == "validation"  # type: ignore[index]
    )
    validation_row["mean_cost_bps"] = 0.0
    prepared = prepare_manifest_payload(draft, project_root=tmp_path)
    manifest = parse_manifest_payload(seal_manifest_payload(prepared))

    validation = validate_policy_manifest(manifest, project_root=tmp_path, now=NOW)

    assert validation.passed is False
    assert "evidence.oos.missing" in validation.reasons
    assert "evidence.validation.zero_costs" in validation.reasons


def test_direction_collapse_is_a_hard_manifest_failure(tmp_path):
    draft = _draft(tmp_path)
    oos = next(
        row for row in draft["evidence"] if row["kind"] == "oos"  # type: ignore[index]
    )
    oos["direction_collapse"] = True
    prepared = prepare_manifest_payload(draft, project_root=tmp_path)
    manifest = parse_manifest_payload(seal_manifest_payload(prepared))

    validation = validate_policy_manifest(manifest, project_root=tmp_path, now=NOW)

    assert "evidence.oos.direction_collapse" in validation.reasons


def test_micro_live_manifest_requires_both_paper_canaries(tmp_path):
    draft = _draft(tmp_path)
    draft["target"] = "micro_live"
    draft["expires_at"] = _iso(NOW + timedelta(hours=12))
    prepared = prepare_manifest_payload(draft, project_root=tmp_path)
    manifest = parse_manifest_payload(seal_manifest_payload(prepared))

    validation = validate_policy_manifest(manifest, project_root=tmp_path, now=NOW)

    assert "evidence.short_paper_canary.missing" in validation.reasons
    assert "evidence.extended_paper_canary.missing" in validation.reasons


def test_artifact_tampering_and_pin_mismatch_fail_closed(tmp_path):
    path, sealed, _ = _write_manifest(tmp_path)
    validation_artifact = tmp_path / "evidence" / "validation.json"
    validation_artifact.write_text("tampered", encoding="utf-8")

    with pytest.raises(ManifestError) as tamper_error:
        load_policy_manifest(
            path,
            project_root=tmp_path,
            expected_sha256=sealed["manifest_sha256"],
            now=NOW,
            verify_runtime_fingerprint=False,
        )
    assert "evidence.validation.artifact_sha256_mismatch" in tamper_error.value.reasons

    with pytest.raises(ManifestError) as pin_error:
        load_policy_manifest(
            path,
            project_root=tmp_path,
            expected_sha256="0" * 64,
            now=NOW,
            verify_artifacts=False,
            verify_runtime_fingerprint=False,
        )
    assert pin_error.value.reasons == ("manifest.expected_sha256_mismatch",)


def test_receipt_source_report_tampering_fails_closed(tmp_path):
    path, sealed, _ = _write_manifest(tmp_path)
    source = tmp_path / "source_reports" / "oos.json"
    source.write_text("tampered source", encoding="utf-8")

    with pytest.raises(ManifestError) as error:
        load_policy_manifest(
            path,
            project_root=tmp_path,
            expected_sha256=sealed["manifest_sha256"],
            now=NOW,
            verify_runtime_fingerprint=False,
        )

    assert "evidence.oos.source_0_sha256_mismatch" in error.value.reasons


def test_compiler_writes_readiness_but_preserves_manifest_when_blocked(tmp_path):
    compiler = _load_compiler()
    draft_path = tmp_path / "draft.json"
    out = tmp_path / "active.json"
    readiness = tmp_path / "readiness.json"
    draft = _draft(tmp_path)
    draft_path.write_text(json.dumps(draft), encoding="utf-8")

    passed = compiler.compile_manifest(
        draft_path=draft_path,
        output_path=out,
        readiness_path=readiness,
        project_root=tmp_path,
        now=NOW,
    )
    original = out.read_text(encoding="utf-8")
    draft["evidence"] = [
        row for row in draft["evidence"] if row["kind"] != "oos"  # type: ignore[index]
    ]
    draft_path.write_text(json.dumps(draft), encoding="utf-8")

    blocked = compiler.compile_manifest(
        draft_path=draft_path,
        output_path=out,
        readiness_path=readiness,
        project_root=tmp_path,
        now=NOW,
    )

    assert passed["passed"] is True
    assert blocked["passed"] is False
    assert blocked["existing_manifest_unchanged"] is True
    assert "evidence.oos.missing" in blocked["blockers"]
    assert out.read_text(encoding="utf-8") == original


def test_compiler_labels_replay_manifest_as_research_only(tmp_path):
    compiler = _load_compiler()
    draft = _draft(tmp_path)
    draft["target"] = "replay"
    draft["evidence"] = []
    draft["expires_at"] = _iso(NOW + timedelta(days=14))
    draft_path = tmp_path / "replay_draft.json"
    out = tmp_path / "replay_manifest.json"
    readiness = tmp_path / "replay_readiness.json"
    draft_path.write_text(json.dumps(draft), encoding="utf-8")

    result = compiler.compile_manifest(
        draft_path=draft_path,
        output_path=out,
        readiness_path=readiness,
        project_root=tmp_path,
        now=NOW,
    )

    assert result["passed"] is True
    assert result["status"] == "RESEARCH_ONLY"
    assert out.exists()


def test_executor_allows_only_exact_costed_policy_rule(tmp_path):
    _, _, loaded = _write_manifest(tmp_path)
    executor = PolicyExecutorV1(loaded)

    decision = executor.decide(_signal(), _context())

    assert decision.outcome == DecisionOutcome.ALLOW_OPEN
    assert decision.trace.primary_reason == DecisionReason.ALLOWED
    assert decision.risk_mult == 0.5
    assert decision.max_notional_usd == 10.0


@pytest.mark.parametrize(
    ("signal", "context", "reason"),
    [
        (_signal(regime="range_low_vol"), _context(), DecisionReason.REGIME_NOT_ALLOWED),
        (_signal(expected_move_bps=13.9), _context(), DecisionReason.EXPECTED_MOVE_BELOW_COST),
        (_signal(actor="GeneticsCore"), _context(), DecisionReason.ACTOR_NOT_ALLOWED),
        (_signal(), _context(kill_switch_active=True), DecisionReason.KILL_SWITCH),
    ],
)
def test_executor_returns_one_primary_no_trade_reason(tmp_path, signal, context, reason):
    _, _, loaded = _write_manifest(tmp_path)

    decision = PolicyExecutorV1(loaded).decide(signal, context)

    assert decision.outcome == DecisionOutcome.NO_TRADE
    assert decision.trace.primary_reason == reason
    assert decision.trace.checks[-1].passed is False


def test_executor_is_deterministic_and_invalid_manifest_never_trades(tmp_path):
    path, sealed, loaded = _write_manifest(tmp_path)
    executor = PolicyExecutorV1(loaded)

    first = executor.decide(_signal(), _context())
    second = executor.decide(_signal(), _context())
    invalid = PolicyExecutorV1.from_path(
        path,
        project_root=tmp_path,
        expected_sha256="f" * 64,
        now=NOW,
    ).decide(_signal(), _context())

    assert first == second
    assert first.trace.as_dict() == second.trace.as_dict()
    assert invalid.outcome == DecisionOutcome.NO_TRADE
    assert invalid.trace.primary_reason == DecisionReason.MANIFEST_INVALID
    assert "manifest.expected_sha256_mismatch" in invalid.trace.detail
    assert sealed["manifest_sha256"] == loaded.manifest.manifest_sha256


def test_direct_bitget_worker_cannot_bypass_live_preflight(monkeypatch):
    from panteon_v2.app import startup

    monkeypatch.setattr(
        startup,
        "_direct_bitget_live_preflight_failure",
        lambda exchange, mode: "BITGET live preflight failed: policy_manifest_missing",
    )
    monkeypatch.setattr(
        startup,
        "resolve_exchange",
        lambda *args, **kwargs: pytest.fail("exchange must not be resolved"),
    )

    result = startup.start_production(
        exchange="BITGET",
        mode="live_futures",
        enable_policy_runtime=True,
    )

    assert result == 2


def test_direct_bitget_worker_rejects_legacy_live_runtime(monkeypatch):
    from panteon_v2.app import startup

    monkeypatch.setattr(
        startup,
        "_direct_bitget_live_preflight_failure",
        lambda *args, **kwargs: pytest.fail("legacy live must stop before preflight"),
    )
    monkeypatch.setattr(
        startup,
        "resolve_exchange",
        lambda *args, **kwargs: pytest.fail("exchange must not be resolved"),
    )

    result = startup.start_production(exchange="BITGET", mode="live_futures")

    assert result == 2
