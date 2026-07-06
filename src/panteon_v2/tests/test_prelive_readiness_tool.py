from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone


def test_readiness_warns_on_orphan_stale_status_without_live_process(tmp_path):
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    now = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
    settings_path = tmp_path / "settings.txt"
    settings_path.write_text(
        "\n".join(
            [
                "v2_flash_live_real_actor_whitelist = GeneticsCore,LiveOIBreakout",
                "mexc_v2_flash_live_real_actor_whitelist = agent:GeneticsCore",
                "v2_flash_range_low_vol_real_actor_allowlist = Solo_GeneticsCore",
                "v2_flash_genetics_probation_bypass_min_closed_enabled = on",
                "v2_flash_genetics_core_primary_bypass_shadow_confirmation_enabled = on",
                "mexc_v2_genetics_probation_execution_enabled = on",
            ]
        ),
        encoding="utf-8",
    )
    session_dir = tmp_path / "Results" / "MEXC" / "2026-06-25_14-23-38_v2"
    session_dir.mkdir(parents=True)
    (session_dir / "status.json").write_text(
        (
            "{"
            '"timestamp_utc":"'
            + (now - timedelta(hours=49)).isoformat()
            + '","run_state":"running","feed_status":"active",'
            '"pid":999999,"mode":"live_futures"}'
        ),
        encoding="utf-8",
    )

    report = tool.build_readiness_report(
        project_root=tmp_path,
        exchanges=("MEXC",),
        now=now,
        active_pids=set(),
        include_git=False,
        include_live_preflight=False,
        include_exchange_rules=False,
        include_promotion=False,
    )

    blocker_ids = {item["id"] for item in report["blockers"]}
    warning_ids = {item["id"] for item in report["warnings"]}
    assert report["passed"] is False
    assert "settings.genetics_core_live_admission" in blocker_ids
    assert "settings.genetics_bypass_enabled" in blocker_ids
    assert "settings.genetics_probation_execution_enabled" in blocker_ids
    assert "live_status.MEXC.stale_status" not in blocker_ids
    assert "live_status.MEXC.pid_not_running" not in blocker_ids
    assert "live_status.MEXC.stale_inactive_status" in warning_ids
    assert "live_status.MEXC.orphan_running_status" in warning_ids
    stale_warning = next(
        item for item in report["warnings"] if item["id"] == "live_status.MEXC.stale_inactive_status"
    )
    orphan_warning = next(
        item for item in report["warnings"] if item["id"] == "live_status.MEXC.orphan_running_status"
    )
    assert stale_warning["cleanup_hints"] == [
        "Review or archive stale status: Results\\MEXC\\2026-06-25_14-23-38_v2\\status.json",
        "Run a fresh isolated paper canary before re-checking readiness",
    ]
    assert orphan_warning["cleanup_hints"] == [
        "Confirm no live MEXC worker owns pid 999999",
        "Review or archive stale status: Results\\MEXC\\2026-06-25_14-23-38_v2\\status.json",
    ]


def test_readiness_blocks_stale_status_for_running_live_pid(tmp_path):
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    now = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)
    (tmp_path / "settings.txt").write_text(
        "\n".join(
            [
                "v2_flash_live_real_actor_whitelist = LiveOIBreakout",
                "mexc_v2_genetics_probation_execution_enabled = off",
            ]
        ),
        encoding="utf-8",
    )
    session_dir = tmp_path / "Results" / "MEXC" / "2026-06-25_14-23-38_v2"
    session_dir.mkdir(parents=True)
    (session_dir / "status.json").write_text(
        (
            "{"
            '"timestamp_utc":"'
            + (now - timedelta(hours=49)).isoformat()
            + '","run_state":"running","feed_status":"active",'
            '"active_pid":12345,"mode":"live_futures"}'
        ),
        encoding="utf-8",
    )

    report = tool.build_readiness_report(
        project_root=tmp_path,
        exchanges=("MEXC",),
        now=now,
        active_pids={12345},
        include_git=False,
        include_live_preflight=False,
        include_exchange_rules=False,
        include_promotion=False,
    )

    blocker_ids = {item["id"] for item in report["blockers"]}
    assert report["passed"] is False
    assert "live_status.MEXC.stale_status" in blocker_ids


def test_readiness_passes_when_project_settings_are_safe(tmp_path):
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    (tmp_path / "settings.txt").write_text(
        "\n".join(
            [
                "v2_flash_live_real_actor_whitelist = LiveOIBreakout",
                "mexc_v2_flash_live_real_actor_whitelist = agent:LiveOIBreakout",
                "v2_flash_range_low_vol_real_actor_allowlist = CarryFlowAgentV2",
                "v2_flash_genetics_probation_bypass_min_closed_enabled = off",
                "v2_flash_genetics_probation_bypass_trend_gate_enabled = off",
                "v2_flash_genetics_core_primary_bypass_shadow_confirmation_enabled = off",
                "mexc_v2_genetics_probation_execution_enabled = off",
            ]
        ),
        encoding="utf-8",
    )

    report = tool.build_readiness_report(
        project_root=tmp_path,
        exchanges=("MEXC",),
        now=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
        active_pids=set(),
        include_git=False,
        include_live_status=False,
        include_live_preflight=False,
        include_exchange_rules=False,
        include_promotion=False,
    )

    assert report["passed"] is True
    assert report["sections"]["settings"]["passed"] is True
    assert report["blockers"] == []


def test_readiness_blocks_stale_settings_snapshot(tmp_path):
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    settings_path = tmp_path / "settings.txt"
    settings_path.write_text(
        "\n".join(
            [
                "v2_flash_live_real_actor_whitelist = LiveOIBreakout",
                "mexc_v2_genetics_probation_execution_enabled = off",
            ]
        ),
        encoding="utf-8",
    )
    snapshot_path = tmp_path / "settings_snapshot.json"
    snapshot_path.write_text(
        (
            "{"
            '"source_sha256":"stale",'
            '"safety_verdict":{"passed":true,"blockers":[]}'
            "}"
        ),
        encoding="utf-8",
    )

    report = tool.build_readiness_report(
        project_root=tmp_path,
        exchanges=("MEXC",),
        now=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
        active_pids=set(),
        include_git=False,
        include_live_status=False,
        include_live_preflight=False,
        include_exchange_rules=False,
        include_promotion=False,
        settings_snapshot_path=snapshot_path,
    )

    blocker_ids = {item["id"] for item in report["blockers"]}
    assert report["passed"] is False
    assert "settings.snapshot_stale" in blocker_ids


def test_genetics_promotion_is_warning_when_genetics_not_live_path(tmp_path):
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    (tmp_path / "settings.txt").write_text(
        "\n".join(
            [
                "v2_flash_live_real_actor_whitelist = LiveOIBreakout",
                "v2_flash_genetics_core_primary_enabled = off",
                "mexc_v2_genetics_probation_execution_enabled = off",
            ]
        ),
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection_router_fitness_v4.json"
    selection_path.write_text(
        '{"promotion_eligible":false,"promotion_failures":["baseline_selected"]}',
        encoding="utf-8",
    )

    report = tool.build_readiness_report(
        project_root=tmp_path,
        exchanges=("MEXC",),
        now=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
        active_pids=set(),
        include_git=False,
        include_live_status=False,
        include_live_preflight=False,
        include_exchange_rules=False,
        promotion_selection_path=selection_path,
    )

    blocker_ids = {item["id"] for item in report["blockers"]}
    warning_ids = {item["id"] for item in report["warnings"]}
    assert report["sections"]["promotion"]["passed"] is True
    assert report["sections"]["promotion"]["required"] is False
    assert "promotion.not_eligible" not in blocker_ids
    assert "promotion.not_eligible_rnd_only" in warning_ids


def test_genetics_promotion_blocks_when_genetics_live_path_enabled(tmp_path):
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    (tmp_path / "settings.txt").write_text(
        "\n".join(
            [
                "v2_flash_live_real_actor_whitelist = GeneticsCore",
                "v2_flash_genetics_core_primary_enabled = on",
                "mexc_v2_genetics_probation_execution_enabled = off",
            ]
        ),
        encoding="utf-8",
    )
    selection_path = tmp_path / "selection_router_fitness_v4.json"
    selection_path.write_text(
        '{"promotion_eligible":false,"promotion_failures":["baseline_selected"]}',
        encoding="utf-8",
    )

    report = tool.build_readiness_report(
        project_root=tmp_path,
        exchanges=("MEXC",),
        now=datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc),
        active_pids=set(),
        include_git=False,
        include_live_status=False,
        include_live_preflight=False,
        include_exchange_rules=False,
        promotion_selection_path=selection_path,
    )

    blocker_ids = {item["id"] for item in report["blockers"]}
    assert report["sections"]["promotion"]["passed"] is False
    assert report["sections"]["promotion"]["required"] is True
    assert "promotion.not_eligible" in blocker_ids


def test_readiness_markdown_lists_blockers_and_canary_command():
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    markdown = tool.render_markdown(
        {
            "generated_at": "2026-07-01T12:00:00+00:00",
            "passed": False,
            "blockers": [
                {
                    "id": "live_preflight.MEXC.matrix_missing",
                    "message": "MEXC: pre-live matrix is missing",
                    "severity": "blocker",
                }
            ],
            "warnings": [],
            "sections": {
                "runbook": {
                    "commands": {
                        "single_exchange_paper_canary": (
                            "python tools/run_panteon3_single_component_canary.py --exchange MEXC"
                        )
                    }
                }
            },
        }
    )

    assert "live_preflight.MEXC.matrix_missing" in markdown
    assert "run_panteon3_single_component_canary.py" in markdown


def test_runbook_uses_long_isolated_dual_exchange_canary():
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")

    runbook = tool.runbook_section(
        exchanges=("MEXC", "BITGET"),
        symbols=("BTC", "ETH"),
        exchange_rules_path="Reports/PreLive/exchange_futures_rules_latest.json",
    )

    canary = runbook["commands"]["single_exchange_paper_canary"]
    summary = runbook["commands"]["canary_summary"]
    assert "--exchange MEXC --exchange BITGET" in canary
    assert "--max-bars 65" in canary
    assert "--max-idle-polls 120" in canary
    assert "Results/Panteon3SingleComponentCanary_isolated/prelive_liveoibreakout_65bar" in canary
    assert "Results/Panteon3SingleComponentCanary_isolated/prelive_liveoibreakout_65bar" in summary
    assert "--exchange MEXC --exchange BITGET" in summary


def test_readiness_blocks_failed_matrix_canary_and_activation(tmp_path):
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    (tmp_path / "settings.txt").write_text(
        "bitget_v2_genetics_probation_execution_enabled = off\n",
        encoding="utf-8",
    )
    matrix_dir = tmp_path / "Reports" / "Panteon3PreLiveMatrix" / "latest"
    canary_dir = tmp_path / "Reports" / "Panteon3Canary" / "latest"
    matrix_dir.mkdir(parents=True)
    canary_dir.mkdir(parents=True)
    (matrix_dir / "panteon3_pre_live_matrix_summary.json").write_text(
        (
            "{"
            '"promotion_verdict":{"passed":false,"fail_reasons":["min_filled"]},'
            '"candidate":{"filled_signals":7,"closed_trades":3}'
            "}"
        ),
        encoding="utf-8",
    )
    (canary_dir / "panteon3_live_canary_summary.json").write_text(
        (
            "{"
            '"passed":false,'
            '"calibration_only":true,'
            '"execution_smoke":true,'
            '"actor_overrides":{"MOM_MIN":0.0015},'
            '"expectancy_gate_required":true,'
            '"exchanges":{"BITGET":{'
            '"passed":false,'
            '"fail_reasons":["nonpositive_expectancy"],'
            '"signals":14,"orders":4,"fills":4,'
            '"expectancy_after_costs":-0.01'
            "}}"
            "}"
        ),
        encoding="utf-8",
    )
    (canary_dir / "live_oi_breakout_activation_report.json").write_text(
        (
            "{"
            '"hard_blocked":true,'
            '"activation_blockers":["zero_candidate_signals"],'
            '"candidate_signal_count":0'
            "}"
        ),
        encoding="utf-8",
    )

    report = tool.build_readiness_report(
        project_root=tmp_path,
        exchanges=("BITGET",),
        include_git=False,
        include_live_status=False,
        include_live_preflight=False,
        include_exchange_rules=False,
        include_promotion=False,
        include_artifact_doctor=True,
    )

    blocker_ids = {item["id"] for item in report["blockers"]}
    assert report["passed"] is False
    assert report["sections"]["matrix"]["passed"] is False
    assert report["sections"]["canary"]["passed"] is False
    assert report["sections"]["activation"]["passed"] is False
    assert "matrix.failed" in blocker_ids
    assert "canary.failed" in blocker_ids
    assert "canary.calibration_only" in blocker_ids
    assert "canary.execution_smoke" in blocker_ids
    assert "canary.actor_overrides" in blocker_ids
    assert "canary.BITGET.nonpositive_expectancy" in blocker_ids
    assert "activation.hard_blocked" in blocker_ids


def test_readiness_blocks_execution_smoke_canary_even_when_technical_checks_pass(tmp_path):
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    (tmp_path / "settings.txt").write_text(
        "bitget_v2_genetics_probation_execution_enabled = off\n",
        encoding="utf-8",
    )
    canary_dir = tmp_path / "Reports" / "Panteon3Canary" / "latest"
    canary_dir.mkdir(parents=True)
    (canary_dir / "panteon3_live_canary_summary.json").write_text(
        (
            "{"
            '"passed":true,'
            '"execution_smoke":true,'
            '"expectancy_gate_required":false,'
            '"exchanges":{"BITGET":{'
            '"passed":true,'
            '"signals":1,"orders":1,"fills":1,'
            '"expectancy_after_costs":-0.01'
            "}}"
            "}"
        ),
        encoding="utf-8",
    )

    report = tool.build_readiness_report(
        project_root=tmp_path,
        exchanges=("BITGET",),
        include_git=False,
        include_live_status=False,
        include_live_preflight=False,
        include_exchange_rules=False,
        include_promotion=False,
        include_artifact_doctor=True,
        matrix_summary_path=tmp_path / "missing_matrix.json",
        activation_report_path=tmp_path / "missing_activation.json",
    )

    blocker_ids = {item["id"] for item in report["blockers"]}
    assert report["sections"]["canary"]["passed"] is False
    assert report["sections"]["canary"]["execution_smoke"] is True
    assert "canary.execution_smoke" in blocker_ids


def test_readiness_markdown_lists_artifact_source_paths(tmp_path):
    tool = importlib.import_module("tools.build_panteon_prelive_readiness")
    report = {
        "generated_at": "2026-07-03T12:00:00+00:00",
        "passed": False,
        "blockers": [
            {
                "id": "matrix.failed",
                "message": "matrix promotion verdict failed",
                "severity": "blocker",
            }
        ],
        "warnings": [],
        "sections": {
            "matrix": {
                "passed": False,
                "path": str(tmp_path / "matrix.json"),
                "fail_reasons": ["min_filled"],
            },
            "canary": {
                "passed": False,
                "path": str(tmp_path / "canary.json"),
                "terminal_denied_context_signal_keys": [
                    "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol",
                ],
                "exchanges": {"BITGET": {"passed": False}},
            },
            "activation": {
                "passed": False,
                "path": str(tmp_path / "activation.json"),
                "activation_blockers": ["zero_candidate_signals"],
            },
            "promotion": {
                "passed": True,
                "selection_path": str(tmp_path / "selection.json"),
                "promotion_eligible": False,
                "promotion_failures": ["baseline_selected", "oos_holdout_gate"],
            },
        },
    }

    markdown = tool.render_markdown(report)

    assert str(tmp_path / "matrix.json") in markdown
    assert str(tmp_path / "canary.json") in markdown
    assert str(tmp_path / "activation.json") in markdown
    assert "matrix.failed" in markdown
    assert "agent:LiveOIBreakout|BTC|FUT_SHORT_HALF|range_low_vol" in markdown
    assert "baseline_selected" in markdown
    assert "oos_holdout_gate" in markdown
