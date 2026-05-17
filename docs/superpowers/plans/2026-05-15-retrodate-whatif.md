# Retrodate session-aware selector what-if

## Goal

Add an offline Retrodate runner for MEXC and BITGET that compares actual Pantheon player selection from JSONL decision logs with a what-if selector score that includes current-session overlay and penalties for stale/current-session underperformance.

## Scope

- Keep live trading paths untouched.
- Read existing `logs/v2_<exchange>_events.jsonl` and latest `Results/<EXCHANGE>/<session>_v2` artifacts.
- Write standard offline artifacts under `Results/Retrodate/<EXCHANGE>/<session>_whatif`.
- Include full candidate lists, actual vs what-if selected player, score deltas, and classification.
- Generate text logs and dashboard artifacts from saved status/leaderboards.

## Verification

- Add focused unit tests before implementation.
- Run targeted retro what-if tests.
- Run both `Retrostart_MEXC.py` and `Retrostart_BITGET.py` offline and inspect generated summaries.
