# Panteon player runtime

Production contract: `Pantheon -> players`. A player is a complete strategy
and is the smallest selectable, attributable and persistent unit. Agents,
ensembles, Flash and Policy V1 remain only in legacy research code and are not
part of the active player-only execution path.

The launcher exposes two modes:

- `multi`: rank players separately for the current market regime and execute
  the best player with proven positive after-cost efficiency;
- `singleton(<player>)`: pin one registered player and disable leader changes.

Every closed bar follows one causal order:

1. synchronize and reconcile exchange state;
2. detect the global and per-symbol market regime;
3. rank players from prior regime-specific and global closed trades;
4. select exactly one real player or `NoTrade`;
5. run all players in shadow so the current bar affects only the next decision;
6. route only the selected player's fresh filled shadow signals through the
   normal risk and execution guards;
7. persist player-by-regime memory in an isolated
   `pantheon_players_v1` snapshot.

Recent closed trades receive more weight than old trades. A player cannot be
selected in `multi` until both its regime score and its global score have the
configured sample and are strictly positive. A leader change waits until the
current owner closes its position; while waiting, `manage_only` permits closes
and rejects new opens.

Useful commands from the project root:

```powershell
# Read-only Bitget Demo credential probe (zero order calls)
python tools/check_bitget_demo_access.py

# Safe Demo launch preview
$env:BITGET_TRADING_MODE = "demo_futures"
python Start_panteon.py --only BITGET --dry-run `
  --trade-regime "singleton(ResearchValidatorAgent)"

# Short replay smoke test
python tools/run_player_efficiency_retro.py --years 2026 --max-bars 500

# Full configured replay
python tools/run_player_efficiency_retro.py --years 2022,2023,2024,2025,2026

# Tests for the active contract
python -m pytest -q src/panteon_v2/tests/test_player_regime.py `
  src/panteon_v2/tests/test_start_panteon.py `
  src/panteon_v2/tests/test_live_preflight.py
```

Live execution stays blocked until a fresh player-only retro artifact and a
fresh player-only Bitget canary both pass preflight. Passing tests proves the
software contract, not profitability.
