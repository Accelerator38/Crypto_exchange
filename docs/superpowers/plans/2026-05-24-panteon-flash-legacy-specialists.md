# Panteon Flash Legacy Specialists Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the useful legacy alpha sources from `PANTEON_FLASH_AGENT_COMPARISON.md` without widening production exposure by default.

**Architecture:** Register orthogonal legacy agents for discovery/shadow scoring, keep known risky carry-flow output under existing hard policy, and add opt-in experimental wrappers that narrow weak live agents to their documented edge regimes. Extend `ActionFilterAgent` with reusable filters instead of changing each legacy agent.

**Tech Stack:** Python dataclasses, existing v1/v2 adapters, pytest.

---

### Task 1: Add Regression Tests

**Files:**
- Modify: `src/panteon_v2/tests/test_retrodate_market_runner.py`
- Modify: `src/panteon_v2/tests/test_strategy_extensions.py`

- [ ] Add tests for `ActionFilterAgent` min regime confidence, funding-cost aligned opens, and lookback-return gates.
- [ ] Add tests that `CandlePatternAgent` and `CarryFlowAgentV2` are known labels, while `Solo_CarryFlowAgentV2` remains hard-denied by default.
- [ ] Add tests that the new experimental wrapper labels are registered only when their base agents exist.
- [ ] Run focused tests and verify the new tests fail before implementation.

### Task 2: Implement Specialist Filters

**Files:**
- Modify: `src/panteon_v2/app/agent_bootstrap.py`

- [ ] Add `CandlePatternAgent` and `CarryFlowAgentV2` to `KNOWN_V1_AGENTS`.
- [ ] Extend `ActionFilterAgent` with `min_regime_confidence`, `funding_cost_aligned_opens`, `min_lookback_return_pct_by_bars`, and `max_lookback_return_pct_by_bars`.
- [ ] Keep filters open-only, so close actions are not blocked by stale contextual gates.
- [ ] Add `CrashHunterStrict`, `VolBreakoutFundingAware`, `AfterShockRegimeOnly`, `LiveTrendFollowBullOnly`, and `LiveMeanRevNeutralOnly` to `EXPERIMENTAL_FLASH_AGENT_SPECS`.
- [ ] Preserve the default shadow-only policy for experimental labels.

### Task 3: Document Scope

**Files:**
- Modify: `docs/PANTEON_FLASH_AGENT_COMPARISON.md`
- Modify: `docs/PANTEON_FLASH_MECHANISMS.md`

- [ ] Mark implemented items and explain why `ExternalSignalAgent`, partial close, and new delta-neutral agents remain later work.
- [ ] Document the new wrapper filters and their opt-in status.

### Task 4: Verify

**Commands:**
- `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_retrodate_market_runner.py src\panteon_v2\tests\test_strategy_extensions.py -q`
- `.\.venv\Scripts\python.exe -m pytest src\panteon_v2\tests\test_flash_allocator.py src\panteon_v2\tests\test_app.py src\panteon_v2\tests\test_retrodate_market_runner.py src\panteon_v2\tests\test_strategy_extensions.py -q`
- `.\.venv\Scripts\python.exe -m compileall -q src`

- [ ] Run a short experimental shadow retro-smoke with `--enable-experimental-flash-actors --max-bars 1000`.
- [ ] If smoke wiring is clean, report metrics as diagnostic only, not as profitability proof.
