# Isolated Freqtrade reset

This directory is the replacement execution spike for Bitget. It is not a
promotion artifact and cannot authorize live trading.

Safety invariants:

- `dry_run=true` is checked into the only config;
- API credentials are empty;
- futures margin is isolated;
- at most one simulated position is allowed;
- `DeterministicPulseStrategy` refuses to start unless dry-run is active;
- the legacy Pantheon Bitget live path remains frozen independently.

Native Windows acceptance uses a dedicated environment:

```powershell
py -3.12 -m venv .venv-freqtrade
.venv-freqtrade\Scripts\python.exe -m pip install -r freqtrade_reset\requirements.lock.txt
.venv\Scripts\python.exe tools\run_freqtrade_reset_acceptance.py
```

For a Linux test host with Docker, run `docker compose config` and then
`docker compose up` from this directory. Keep `dry_run=true`.
