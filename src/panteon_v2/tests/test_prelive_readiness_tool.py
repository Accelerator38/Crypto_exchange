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
