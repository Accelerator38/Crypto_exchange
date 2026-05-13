# Panteon v2 Operator Visuals And Assets Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make live Panteon v2 operator output show account balance, total asset value, and visual dashboards like the Shadow Player branch.

**Architecture:** Keep v2 as the source of truth, but add a normalized account snapshot on the exchange adapter and a PNG renderer fed by v2 status/leaderboard data. `OutputWriter` writes both machine-readable JSON and operator-visible PNG/HTML/text files on startup and periodic snapshots.

**Tech Stack:** Python unittest, existing Panteon v2 `OutputWriter`, `V1FuturesExchangeAdapter`, matplotlib with `Agg` backend.

---

### Task 1: Account Snapshot In Status And Logs

**Files:**
- Modify: `src/panteon_v2/tests/test_app.py`
- Modify: `src/panteon_v2/app/v1_futures_adapter.py`
- Modify: `src/panteon_v2/app/main_loop.py`
- Modify: `src/panteon_v2/app/bootstrap.py`
- Modify: `src/panteon_v2/app/output_writer.py`

- [ ] Write tests proving `sync_pipeline_balance()` stores `account_snapshot` and uses futures equity for sizing.
- [ ] Write tests proving `status.json` contains `current_balance`, `total_assets_usd`, `futures_equity_usd`, `spot_assets_usd`, and `available_balance_usd`.
- [ ] Write tests proving `trading.log` lines include `balance=` and `assets=`.
- [ ] Implement normalized snapshot reads from `get_full_snapshot()` and `account_assets()`.
- [ ] Update status/log writer to use the latest snapshot.

### Task 2: Shadow-Style PNG Dashboards

**Files:**
- Modify: `src/panteon_v2/tests/test_app.py`
- Create: `src/panteon_v2/dashboards/png_renderer.py`
- Modify: `src/panteon_v2/dashboards/__init__.py`
- Modify: `src/panteon_v2/app/output_writer.py`

- [ ] Write tests proving a fresh v2 output session creates PNG dashboards before the first bar.
- [ ] Render `dashboard_latest.png` with account summary and top leaderboards.
- [ ] Render `shadow_agents_dashboard.png` and `shadow_player_dashboard.png`.
- [ ] Render `agent_regime_dashboard.png` and `player_regime_dashboard.png`.
- [ ] Link PNG files from `dashboard.html`.

### Task 3: Verify And Publish

**Files:**
- Test: `src/panteon_v2/tests/test_app.py`
- Test: `tests/test_dashboard_and_positions.py`

- [ ] Run targeted tests red, then green.
- [ ] Run full v2 suite in the project venv.
- [ ] Run dashboard/positions tests in the project venv.
- [ ] Compile modified files.
- [ ] Commit and push `Panteon_v2`.
