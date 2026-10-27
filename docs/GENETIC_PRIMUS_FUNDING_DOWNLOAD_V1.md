# Public funding history download v1

Explicit user permission: download historical Bitget funding for the fixed
Aug 1–Sep 30 2026 interval via public REST. No WS, credentials, order/runtime,
account calls, instruments calls, mark-price downloads or financial evaluations.
Safety fields remain false: artifacts never grant future launch authority.

Use the official V2 GET history-fund-rate endpoint, productType USDT-FUTURES,
one known full8 symbol, pageSize100/pageNo. A CLI fetch performs **one** request,
timeout5 seconds, response cap128 KiB, zero retries, no redirects/cookies/auth.
Do not run it in background or loop many requests in a shell. Inspect each result
before requesting the next page. All output files are created exclusively.

The source does not accept start/end parameters; responses can contain outside-
interval rows needed to bracket coverage. Preserve those raw responses, retain only
Aug–Sep events in the final event bank, and record the latest actually seen funding
time for any future prospective seal. Do not fake future coverage if the provider
clock or latest event precedes the requested interval.

Raw JSON and page receipts are SHA256 pinned. Strict envelopes, timestamp types,
symbols, descending unique records, decimal rates, and bounded bytes are checked.
Rates are fractions per settlement: `bps = fraction * 10000`. Positive and negative
rates are retained exactly as decimal strings; no float rounding or sign filtering.
Matching duplicates across mutable API pages are deduplicated and reported;
conflicting rates or pagination without progress fail closed. No zero-fill gaps.

For the requested interval verify left/right boundary guards and observed 1/2/4/8h
gaps. The complete contiguous REST page enumeration is evidence from the provider,
not an independent historical funding-schedule confirmation. Current instruments
cannot retrospectively prove schedule changes. Settlement marks, registered funding
valuation/cash policy, normal/stress costs and trade policy are still separate
dependencies; funding alone is not a completed financial replay.

Commands use only `E:\Crypto_exchange\.venv\Scripts\python.exe -X utf8`:

1. `tools/download_genetic_primus_funding_v1.py check` — 15 pure synthetic tests.
2. `... prepare --output-dir Retrodate/NAME` — new immutable plan, no HTTP.
3. `... fetch --output-dir Retrodate/NAME --symbol BTCUSDT --page 1` — one GET.
4. `... inspect --output-dir Retrodate/NAME --symbol BTCUSDT` — local page audit.
5. Continue one page per invocation until start boundary is reached.
6. `... seal --output-dir Retrodate/NAME` — local only; requires all eight covered.
7. `... status --output-dir Retrodate/NAME` — preserve a blocked/partial report if
   HTTP or actual date/coverage prevents finishing. Never call such a package complete.

The older funding snapshot and rolling preflight stay immutable. No pytest or
collector is imported; no asynchronous worker, sleep, supervisor, or trading adapter.
