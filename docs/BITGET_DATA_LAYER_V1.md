# Bitget public data layer v1

## Scope

This layer records public Bitget USDT futures market events. It has no exchange
credentials, order client, policy manifest or promotion authority.

The fixed profile `bitget_full8_microstructure_v1` subscribes to:

- public `trade` events;
- public `books5` snapshots;
- public `ticker` snapshots containing bid/ask, mark/index, funding and OI;
- exact BTC, ETH, SOL, BNB, XRP, DOGE, ADA and LINK USDT futures.

The same process periodically reconciles the WebSocket stream against public
Bitget REST facts:

- exact last closed one-minute candle;
- open interest;
- current funding schedule;
- instrument status, fee and order-size rules.

REST failures are quality events. They do not silently replace the WebSocket
value.

The collector has only one algorithmic setting: `--profile-id`. Data directory
and optional run duration are operational controls, not trading parameters.

## Storage contract

Events are first committed to a SQLite WAL database. UTC-aligned 15-minute
segments are then closed, integrity-checked, renamed from `.partial` and sealed
with a SHA-256 manifest. Segment manifests form a hash chain inside a collector
session.

Every event retains:

- Bitget exchange timestamp;
- local UTC receive timestamp;
- local monotonic receive timestamp;
- channel, symbol and event identity;
- book sequence when supplied by Bitget;
- original normalized channel payload.

A process restart creates a new session. It does not destroy older sealed
segments. Actual gaps and non-increasing book sequences remain explicit quality
events and must not be hidden by a downstream dataset.

The data directory has one process lock. On startup the owner first recovers a
committed `.sqlite.partial` left by an unclean shutdown, verifies its embedded
old profile/fingerprint, checkpoints the WAL and seals it as
`recovered_after_unclean_shutdown`. Corrupt or ambiguous orphan state fails
closed instead of being overwritten.

## Run

Install the pinned project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run continuously:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_data_layer.py
```

Run a bounded technical check:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_data_layer.py `
  --duration-seconds 60
```

Validate a completed session:

```powershell
.\.venv\Scripts\python.exe tools\check_bitget_data_layer.py `
  --session-dir Retrodate\bitget_data_v1\sessions\<session-id>
```

The validator checks the session hash, segment hash chain, database SHA,
SQLite integrity, row counts, unsealed partial files and channel/symbol
coverage. It does not grant paper or live trading.

## One-second materialized view

Build one actor-independent view from exactly one sealed session:

```powershell
.\.venv\Scripts\python.exe tools\materialize_bitget_frame_1s.py `
  --session-dir Retrodate\bitget_data_v1\sessions\<session-id> `
  --output-dir Retrodate\bitget_data_v1\datasets\<dataset-id>
```

Validate the dataset and its still-immutable source:

```powershell
.\.venv\Scripts\python.exe tools\check_bitget_frame_1s.py `
  --dataset-dir Retrodate\bitget_data_v1\datasets\<dataset-id>
```

`frame_1s_v1` uses local receive time as data availability time. It contains
top-of-book, spread, microprice, five-level depth and imbalance, mark/index
basis, funding, OI, signed trade flow and a fixed executable-price curve for
`$5/$10/$25/$50/$100`. The fixed ladder is a measured market fact, not a
strategy notional setting.

A silence longer than five seconds starts a new continuity window and clears
as-of state. Sessions are never concatenated by the materializer.

### Active frame contract v2

`frame_1s_v1` used a two-second received-age limit for `books5`. That rule is
retired for prospective evidence: Bitget sends a new `books5` snapshot when
the order book changes, so a quiet book can legitimately have no new message.
Treating message age as book age reduced eligible-bar coverage on quiet symbols
without identifying a stream failure.

`frame_1s_v2` keeps the last complete snapshot valid inside the same continuous
WebSocket window. The existing five-second global receive gap still clears all
as-of state and starts a new continuity window. A separate 60-second emergency
ceiling fails closed if one channel silently stops while the shared connection
continues. Book sequence anomalies remain raw quality events.

The `95%` campaign coverage gate was not reduced. Re-materializing the sealed
12-hour session `20260801T213112Z-b434b442` under v2 increased eligible-bar
coverage from `92.60%` to `99.72%`; its role is development-only, not validation.
The sealed 24-hour development session reaches `99.65%` eligible bars.

## Minute research dataset

Build the fixed downstream feature and label dataset:

```powershell
.\.venv\Scripts\python.exe tools\materialize_bitget_research_v1.py `
  --frame-dataset-dir Retrodate\bitget_data_v1\datasets\<dataset-id> `
  --output-dir Retrodate\bitget_data_v1\research\<research-id>
```

Validate its hashes, SQLite contents and immutable `frame_1s_v1` source:

```powershell
.\.venv\Scripts\python.exe tools\check_bitget_research_v1.py `
  --dataset-dir Retrodate\bitget_data_v1\research\<research-id>
```

The contract has no strategy settings. It always uses:

- one-minute bars with at least 57 complete seconds;
- five- and fifteen-minute forward horizons;
- `$25` executable VWAP from the captured `books5` curve;
- the captured per-symbol Bitget taker fee on entry and exit;
- continuity-preserving windows only.

Features contain returns, realized volatility, signed trade-flow imbalance,
book imbalance, microprice pressure, basis, funding and OI change. Forward
labels contain both mid-price gross movement and net executable movement after
book impact, spread and fees. Entry features never read a future frame.

Run fixed costed controls and microstructure hypotheses:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_microstructure_baselines_v1.py `
  --dataset-dir Retrodate\bitget_data_v1\research\<research-id> `
  --output-json Reports\BitgetData\<run-id>\baseline_report.json `
  --output-md Reports\BitgetData\<run-id>\baseline_report.md
```

The baseline thresholds are fixed in source code. The report uses chronological
60/20/20 development, validation and OOS slices with one non-overlapping global
position. It is diagnostic evidence only. A single 24-hour source session is
hard-blocked by `evidence_duration_below_72h`, even if an individual fixed
baseline happens to be positive.

Run the event-driven triple-barrier study without another intermediate
artifact:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_event_study_v1.py `
  --dataset-dir Retrodate\bitget_data_v1\research\<research-id> `
  --output-json Reports\BitgetData\<run-id>\event_study_report.json `
  --output-md Reports\BitgetData\<run-id>\event_study_report.md
```

This command has no trading knobs. It fixes 15/30/60-minute horizons and
observes barriers only at executable minute-close books. Take-profit is net
`+10 bps`. Stop-loss is the entry-time immediate-roundtrip net value minus
another `20 bps` adverse move; this avoids placing the stop only a few bps away
after the initial spread, slippage and fees. Entry candidates require a fixed
move budget of at least `20 bps`, computed only from entry-time five-minute
return and realized volatility. A position blocks another global entry until
its actual exit.

Future books are label data only. Signals cannot inspect them. Trades crossing
development/validation/OOS boundaries are excluded from split metrics. The
result remains research-only and cannot create a policy manifest.

Test the fixed post-only entry scenario against the immutable raw trade tape:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_maker_event_study_v1.py `
  --dataset-dir Retrodate\bitget_data_v1\research\<research-id> `
  --output-json Reports\BitgetData\<run-id>\maker_event_study_report.json `
  --output-md Reports\BitgetData\<run-id>\maker_event_study_report.md
```

The order contract is fixed: `$25` post-only at the captured signal-minute best
bid/ask, 60-second TTL, then a taker exit. A maker fill requires aggressive
opposite-side raw trades at the limit or better with cumulative notional of at
least `$25`. Signal ranking occurs before fill evidence is known, so an
unfilled best signal cannot be replaced retroactively by a lower-ranked filled
signal.

This is still an optimistic queue model. Trade-through volume proves that
marketable volume existed, but `books5` does not prove the order's queue
position. Therefore `maker_queue_position_unverified` is always a campaign
hard-block. Removing it requires prospective paper orders or a queue-aware
depth feed, not a configuration bypass.

## Multi-day directional discovery

The short microstructure capture is not expanded by concatenating interrupted
sessions. For offline directional discovery, download a separate, validated
Bitget full8 one-minute history snapshot:

```powershell
.\.venv\Scripts\python.exe tools\build_exchange_futures_retrodate.py `
  --exchange BITGET `
  --symbols BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,DOGE/USDT,ADA/USDT,LINK/USDT `
  --start-date 2026-06-15 `
  --end-date 2026-07-28 `
  --timeframe 1m `
  --source-api bitget-v3 `
  --limit 100 `
  --parallel-workers 8 `
  --output-dir Retrodate\bitget_1m_discovery_v1_20260615_20260728
```

The integrity manifest must say `timeframe=1m`, contain exact full8 symbols and
pass continuity validation. File names alone are not evidence of timeframe;
older local `crypto_1m_*` files built with an hourly manifest remain hourly
data and are rejected by the analyzer.

Run the fixed anchored walk-forward classifier:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_cost_aware_directional_v1.py `
  --dataset-dir Retrodate\bitget_1m_discovery_v1_20260615_20260728 `
  --output-json Reports\BitgetData\cost_aware_directional_20260615_20260728\report.json `
  --output-md Reports\BitgetData\cost_aware_directional_20260615_20260728\report.md
```

This command has no strategy CLI knobs. It fixes a five-minute decision
cadence, 60-minute horizon, 20 bps gross barriers and the measured 8 bps
maker-entry/taker-exit cost. Each fold uses an expanding 14-day-or-longer
training window, then seven validation days and seven untouched OOS days.
Trade labels crossing a split boundary are purged. Validation alone chooses a
probability threshold from the fixed grid; failed validation and any OOS fold
with nonpositive costed expectancy are hard failures. The fixed gate also
rejects arithmetic drawdown above 20%, a symbol producing more than 50% of
trades, or either direction producing less than 10% of trades.

Even a passing result is only eligible for prospective validation against a
sealed microstructure session. Historical candles cannot establish maker fill
or queue position. The command never writes a runtime policy and has
`orders_enabled=false` and `promotion_authority=false`.

Run the predeclared low-turnover event campaign after the broad directional
classifier has been rejected:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_rare_event_discovery_v1.py `
  --dataset-dir Retrodate\bitget_1m_discovery_v1_20260615_20260728 `
  --output-json Reports\BitgetData\rare_event_discovery_20260615_20260728\report.json `
  --output-md Reports\BitgetData\rare_event_discovery_20260615_20260728\report.md
```

This is one fixed five-hypothesis campaign: range breakout with volume,
compression breakout, liquidity-sweep reversal, cross-market shock
continuation and volatility-expansion trend. There are no candidate or
threshold CLI settings. All use a 240-minute horizon, gross `+60/-30 bps`
barriers, `12 bps` taker/taker base costs and `16 bps` cost stress. When both
barriers are touched inside the same minute, the result is the stop loss.

Every candidate must independently pass all anchored validation and OOS folds,
including positive base and stress LCB, trade-count, drawdown, symbol
concentration, direction and a maximum of four trades per day. Testing five
predeclared hypotheses is still research, not promotion evidence. Any survivor
must next be validated on a
sealed prospective microstructure session before a paper policy may exist.

## Prospective precursor campaign

The v1 precursor campaign is retained as immutable diagnostic history. New
evidence must use the v2 frame/research contract and the separate v2 campaign:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_precursor_campaign_v2.py
```

The v2 lock requires `event_snapshot_continuity_bounded_v2`; v1 and v2 research
manifests cannot be mixed. The two sessions available when the v2 lock was
created are development-only. Their current five fixed precursor candidates
all have negative combined costed development expectancy, so do not start a
new 12-hour validation session merely to accumulate duration. First freeze a
candidate that is positive across the independent development sessions.

### Active h120 campaign v3

The active prospective experiment is deliberately reduced to one candidate:

- absolute five-minute OI change at least `10 bps`;
- absolute five-minute return at least `10 bps`;
- mean spread no more than `2 bps`;
- direction follows the five-minute return;
- fixed `120m` holding period and one global position slot.

Its two development sessions contain `15` closed trades / `30` fills with
mean net `+20.65 bps`, cost-stress LCB `+2.47 bps`, maximum drawdown
`50.77 bps`, 40% maximum symbol share and both LONG/SHORT trades. These are
development-selected results and do not authorize paper or live trading.

Inspect the immutable active campaign:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_precursor_campaign_v3.py
```

Run exactly one prospective 12-hour validation session:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_precursor_evidence_session_v3.py
```

There are no candidate, threshold, horizon, symbol or trading arguments. If
the first validation session has fewer than three closed trades or nonpositive
expectancy after the additional four-bps stress, the campaign returns
`rejected_early` and no further continuous collection should be started.

Freeze and update the precursor campaign with one command:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_precursor_campaign_v1.py `
  --research-root Retrodate\bitget_data_v1\research `
  --campaign-dir Reports\BitgetData\precursor_campaign_v1
```

The first run writes `campaign_lock.json`. Its hash covers the five fixed
candidate definitions, thresholds, 15-minute horizon, captured executable
taker costs and an additional four-bps cost stress. Changing any candidate
contract invalidates the lock instead of silently restarting selection.

Every research dataset that exists when the lock is created is permanently
development-only. Later sealed source sessions are ordered chronologically:
the first two become validation and the next three or more become OOS. Each
prospective session must independently contain at least 12 hours with at least
95% eligible-bar and one-second-frame coverage. Sessions and continuity windows
are never concatenated, so a machine restart does not discard already sealed
independent evidence.

The fixed candidates test OI/flow/book quorum, absorption reversal, liquidity
pressure, neutral OI build and OI unwind. Signals use only entry-time facts;
outcomes use the existing `$25` executable 15-minute labels. Passing requires
positive validation and OOS expectancy and LCB under both captured costs and
cost stress, at least 20 fills/10 closed trades, bounded drawdown, no symbol or
direction collapse, and positive stress expectancy in at least two thirds of
OOS sessions.

Even a successful campaign only becomes eligible for strict paper canary. It
does not create a runtime policy, enable orders or grant promotion authority.

The previous v2 evidence command is retained for reproducibility only:

```powershell
.\.venv\Scripts\python.exe tools\run_bitget_precursor_evidence_session_v2.py
```

The command has no duration, symbol, strategy or trading arguments. It always
collects the fixed full8 public profile for 12 hours, validates the sealed raw
session, materializes `frame_1s_v2` and research labels, then updates the frozen
v2 campaign. A short, interrupted or non-`duration_complete` collector session
is left as raw evidence and fails before materialization. Its next invocation
recovers any committed partial segment and starts a new independent session.

Progress and the exact failed/completed stage are written under
`Reports\BitgetData\precursor_campaign_v1\session_runs\<run-id>\state.json`.
This state also hard-codes `orders_enabled=false` and
`promotion_authority=false`.

## Safety

`orders_enabled=false` and `promotion_authority=false` are validated in the
profile, session and every segment manifest. This collector must remain
independent from `bitget_connector`, policy execution, Flash and Genetics.

The next data-layer increment is daily compact retention and replay across
multiple independent market days. Raw sessions and `frame_1s_v1` remain
immutable.
