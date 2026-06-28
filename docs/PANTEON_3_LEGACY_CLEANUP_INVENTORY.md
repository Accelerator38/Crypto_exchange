# Panteon 3 Cleanup Inventory

Date: 2026-06-24
Scope: working tree hygiene and safe legacy cleanup planning before Panteon 3.0 behavior changes.

## Already Cleaned In This Pass

- Removed `.pytest_cache`.
- Removed `__pycache__` directories outside `.venv`.
- Removed generated `tmp/gate_funnel_*` analyzer reports.

These artifacts are reproducible and should not be tracked.

## Keep As Active Panteon 3.0 Work

- `src/panteon_v2/app/main_loop.py`
- `src/panteon_v2/app/output_writer.py`
- `src/panteon_v2/dashboards/png_renderer.py`
- `src/panteon_v2/tests/test_app.py`
- `src/panteon_v2/tests/test_png_renderer.py`
- `src/panteon_v2/tests/test_gate_funnel_analyzer.py`
- `tools/analyze_panteon_gate_funnel.py`
- `docs/superpowers/plans/2026-06-24-panteon-3-activity-execution-edge.md`

Reason: these implement or verify gate funnel diagnostics and NoTrade observability.

## Keep As Active ML/Execution Work

- `Start_ML.py`
- `Start_ML_BITGET.py`
- `src/panteon_runtime/bitget_connector.py`
- `src/panteon_runtime/mexc_connector.py`
- `src/panteon_v2/ml/bitget_adapter.py`
- `src/panteon_v2/ml/dashboard.py`
- `src/panteon_v2/ml/executor.py`
- `src/panteon_v2/ml/runner.py`
- `src/panteon_v2/ml/admission.py`
- `src/panteon_v2/ml/admission_promotion.py`
- `src/panteon_v2/ml/supervisor.py`
- `src/panteon_v2/tests/test_ml_admission_promotion.py`
- `src/panteon_v2/tests/test_ml_executor.py`
- `src/panteon_v2/tests/test_ml_runner.py`
- `src/panteon_v2/tests/test_production_hardening.py`
- `models/entry_edge_h6_admission.json`
- `tools/promote_entry_edge_admission.py`
- `tools/run_entry_edge_nightly_promotion.cmd`
- `tools/install_entry_edge_nightly_promotion_task.ps1`
- `tools/install_ml_live_supervisor_task.ps1`
- `tools/ml_live_watchdog.ps1`
- `tools/run_ml.cmd`
- `tools/run_ml_bitget.cmd`

Reason: these were already present as uncommitted work before this cleanup pass and appear related to fill-confirmed ML/live supervision. Do not revert without review.

## Keep As Evidence/Baseline

- `PANTEON_V3_AUDIT.md`
- `docs/PANTEON_CURRENT_CODE_AND_LOG_ANALYSIS_2026-06-11.md`
- `Results/`
- `Retrodate/`
- `Reports/panteon_live_audit_2026-06-11_15.md`
- `Reports/legacy_vs_current_bot_report_2026-05-31.md`
- `Reports/panteon_flash_profitability_matrix.json`
- `Reports/PanteonLegend_20260531/`
- `Reports/AgentRegimeMap/`
- `Reports/GeneticsAgentAnalysis_20260603/`
- `Reports/GeneticsAgentAnalysis_20260604/`

Reason: these contain audit, replay, live, component, and baseline evidence needed for before/after Panteon 3.0 comparison.

## Archive Candidates

Move to `docs/archive/` or an external artifact store after confirming no scripts still reference them:

- `Reports/PanteonFlashGeneratedSelectedDeny*`
- `Reports/PanteonFlashSelectedDenyProbe*`
- `Reports/PanteonFlashSelectedDenyConfirmationRound3_20260525/`
- `Reports/PanteonFlashCooldown720*`
- `Reports/PanteonFlashRiskFrequencySweep*`
- `Reports/PanteonFlashSymbolOnly*`
- `Reports/PanteonFlashMomentumCap10*`
- `Reports/PanteonFlashPostV4_*`
- `Reports/PanteonFlashRound2TerminalAtomCooldown720*`
- `Reports/PanteonFlashShadowPnlLcb*`
- `Reports/PanteonFlashLcbDenyDiagnostics*`
- `Reports/PanteonFlashContractDryRun/`
- `Reports/PanteonFlashContractSmoke/`
- `Reports/Panteon_Flash_operational_report_2026-05-21.docx`
- `Reports/Panteon_Flash_operational_report_rendered/`

Reason: these look like historical sweep/report artifacts. Keep the latest summarized evidence, but remove repeated experiment directories from the active project surface.

## Move To Legacy Candidates

Move to `legacy/entrypoints/` after verifying launch scripts and docs no longer reference them:

- `Start_DEFAULT.py`
- `Start_DEFAULT_v2.py`
- `Start_MEXC.py`
- `Start_MEXC_v2.py`
- `Start_BITGET.py`
- `Start_BITGET_v2.py`
- `Start_BITGET_legend.py`
- `Retrostart_MEXC.py`
- `Retrostart_BITGET.py`

Reason: `Start_panteon_v3.py`, `Start_ML.py`, and `Start_ML_BITGET.py` are the current entrypoint surface. Old root start scripts increase operator error risk.

## Delete Candidates After Confirmation

- `.tmp_retro_whatif_tests/`
- `tmp/*.pstats`
- `tmp/ml_dashboard_smoke/`
- `tmp/flash_attr_*`
- Remaining non-venv `__pycache__` directories if recreated.
- `.pytest_cache/` if recreated.

Reason: local test/profiling artifacts. They are reproducible and should stay out of source control.

## Do Not Touch In Cleanup

- `.venv/`
- `.env.local`
- `.webui_secret_key`
- `settings.txt`
- `ключ мехс.txt`
- `logs/`
- `state/`
- `panteon_v2_state/`
- `Crypto_exchange — 15_04_2026 - legacy/`
- `Genetics_DL_Agents/Agents/`
- `Genetics_DL_Agents/results/`

Reason: environment, credentials, local runtime state, or large historical material. Deleting these can break local operation or remove non-reproducible data.

## Next Safe Cleanup Steps

1. Run a reference search before moving any entrypoint:
   `rg "Start_MEXC|Start_BITGET|Start_DEFAULT|Retrostart" .`
2. Move old entrypoints to `legacy/entrypoints/` in one patch and update docs/scripts that reference them.
3. Archive repeated May 2026 Flash sweep report directories after extracting one summary index.
4. Keep before/after evidence until Panteon 3.0 walk-forward comparison is complete.
