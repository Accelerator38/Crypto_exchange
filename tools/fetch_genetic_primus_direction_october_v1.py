"""Authorized historical-only REST intake. One symbol/3-day batch per command."""
from __future__ import annotations
import argparse
import json
import time
import urllib.request
import urllib.parse
from pathlib import Path
import fetch_genetic_primus_v8_clean_ohlcv_backfill as old
import run_genetic_primus_direction_ternary_hypotheses_v1 as parent

ROOT = parent.ROOT
CONFIG = ROOT / "configs/genetic_primus_direction_october_halfband_v1.json"
OUT = ROOT / "Retrodate/genetic_primus_direction_october_20261001_20261013_v1"
DAY, MINUTE = 86400000, 60000

def pins(): return {parent.rel(p): parent.parent.digest(p) for p in (CONFIG, Path(__file__), Path(old.__file__))}
def config():
    c = parent.read(CONFIG)
    assert all(c[k] is False for k in parent.fixed.SAFETY)
    assert c["intake"]["authorized_public_historical_rest"] and c["intake"]["parallelism"] == 1
    parent.config()
    assert c["test_end_exclusive_ms"]-c["test_start_ms"] == 13*DAY
    return c

def plan():
    c = config(); OUT.mkdir(parents=True, exist_ok=True)
    parent.parent.write(OUT / "plan.json", dict(contract=c, code_pins=pins(), expected_minute_rows=13*1440*8,
        expected_symbol_batches=40, expected_requests=216, user_authorized=True,
        scope="Public historical REST only; model network_allowed remains false", **{k: False for k in parent.fixed.SAFETY}))
    print(json.dumps(dict(batches=40, maximum_requests=216, minute_rows=149760)))

def registered():
    c = config(); p = parent.read(OUT / "plan.json")
    assert p["code_pins"] == pins()
    return c, p

def specs(c, batch, symbol):
    if batch not in range(5) or symbol+"/USDT" not in c["symbols"]: raise ValueError("registered symbol/batch only")
    start = c["test_start_ms"]+batch*3*DAY; end = min(start+3*DAY, c["test_end_exclusive_ms"])
    result = []
    for a in range(start, end, 1000*MINUTE):
        b = min(a+1000*MINUTE, end)
        result.append(("candles", dict(category="USDT-FUTURES", symbol=symbol+"USDT", interval="1m", type="market", startTime=str(a-MINUTE), endTime=str(b-MINUTE), limit=str((b-a)//MINUTE)), a, b))
    result.append(("history-candles", dict(category="USDT-FUTURES", symbol=symbol+"USDT", interval="1m", type="market", startTime=str(end-100*MINUTE), endTime=str(end), limit="100"), end-100*MINUTE, end))
    return result

class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("redirect forbidden for historical data intake")

def fetch(batch, symbol):
    import pandas as pd
    c, plan = registered(); specifications = specs(c, batch, symbol)
    deadline, calls, last = time.monotonic()+20, 0, 0.
    pages = []; raw_pins = {}
    raw_dir = OUT / "raw"; raw_dir.mkdir(exist_ok=True)
    opener = urllib.request.build_opener(NoRedirect())
    for index, (endpoint, params, a, b) in enumerate(specifications):
        path = raw_dir / ("batch_%02d_%s_page_%02d.json" % (batch, symbol, index))
        url = c["intake"]["base"]+endpoint+"?"+urllib.parse.urlencode(params)
        if path.exists():
            record = parent.read(path)
            assert record["url"] == url and record["plan_sha256"] == plan["sha256"]
        else:
            remaining = deadline-time.monotonic()
            if remaining < 1 or calls >= 6: raise TimeoutError("bounded command budget reached; saved pages can resume")
            delay = max(0., .12-(time.monotonic()-last))
            if delay: time.sleep(delay)
            with opener.open(urllib.request.Request(url, headers={"User-Agent": "GeneticPrimusHistoricalResearch/1"}), timeout=min(8, remaining)) as response:
                payload = response.read(2_000_001)
                if len(payload) > 2_000_000: raise ValueError("response size bound exceeded")
                status = response.status
            calls += 1; last = time.monotonic()
            import hashlib
            record = dict(url=url, response_utf8=payload.decode("utf-8"), response_sha256=hashlib.sha256(payload).hexdigest(),
                plan_sha256=plan["sha256"], http_status=status)
            parent.parent.write(path, record)
        import hashlib
        assert hashlib.sha256(record["response_utf8"].encode("utf-8")).hexdigest() == record["response_sha256"]
        value = json.loads(record["response_utf8"])
        if value.get("code") != "00000": raise ValueError("historical endpoint rejected request: "+str(value.get("code")))
        rows = old.valid_rows(value.get("data"), a, b)
        pages.append(rows); raw_pins[parent.rel(path)] = parent.parent.digest(path)
    rows = sum(pages[:-1], [])
    if rows[-100:] != pages[-1]: raise ValueError("primary/history boundary or values disagree")
    start, end = specifications[0][2], specifications[-1][3]
    rows = old.valid_rows(rows, start, end)
    derived = OUT / "parts"; derived.mkdir(exist_ok=True)
    path = derived / ("batch_%02d_%s.parquet" % (batch, symbol)); receipt = path.with_suffix(".json")
    if receipt.exists():
        saved = parent.read(receipt)
        assert saved["raw_pins"] == raw_pins and saved["parquet_sha256"] == parent.parent.digest(path)
    else:
        if path.exists(): raise ValueError("partial parquet exists; no overwrite")
        frame = pd.DataFrame({"timestamp": [int(r[0]) for r in rows], "symbol": symbol+"/USDT",
            **{name: [float(r[i+1]) for r in rows] for i, name in enumerate(["open", "high", "low", "close", "volume"])}})
        frame.to_parquet(path, index=False)
        parent.parent.write(receipt, dict(symbol=symbol+"/USDT", batch=batch, start_ms=start, end_exclusive_ms=end,
            rows=len(rows), raw_pins=raw_pins, parquet_sha256=parent.parent.digest(path),
            historical_endpoint_tail_parity=True, plan_sha256=plan["sha256"], **{k: False for k in parent.fixed.SAFETY}))
    print(json.dumps(dict(batch=batch, symbol=symbol, minute_rows=len(rows), new_requests=calls, tail_crosscheck=True)))

def audit():
    import pandas as pd
    c, plan = registered(); total = 0; artifact_pins = {}; records = []
    for batch in range(5):
        for coin in c["symbols"]:
            symbol = coin.split('/')[0]; path = OUT / "parts" / ("batch_%02d_%s.parquet" % (batch, symbol))
            receipt = parent.read(path.with_suffix(".json"))
            assert receipt["parquet_sha256"] == parent.parent.digest(path)
            for p, sha in receipt["raw_pins"].items(): assert parent.parent.digest(ROOT / p) == sha
            frame = pd.read_parquet(path)
            assert frame.timestamp.tolist() == list(range(receipt["start_ms"], receipt["end_exclusive_ms"], MINUTE))
            assert frame.symbol.eq(coin).all()
            total += len(frame); records.append(dict(symbol=coin, batch=batch, rows=len(frame)))
            artifact_pins.update(receipt["raw_pins"])
            for p in (path, path.with_suffix(".json")): artifact_pins[parent.rel(p)] = parent.parent.digest(p)
    assert total == 149760
    artifact_pins.update(pins()); artifact_pins[parent.rel(OUT / "plan.json")] = parent.parent.digest(OUT / "plan.json")
    parent.parent.write(OUT / "data_audit.json", dict(total_minute_rows=total, symbol_batches=40, records=records,
        missing_duplicates_or_shifted_bars=0, tail_crosschecks=40, artifact_pins=artifact_pins,
        qualification="Public market OHLCV, not orderbook/fill evidence", **{k: False for k in parent.fixed.SAFETY}))
    print(json.dumps(dict(rows=total, batches=40, gaps=0, crosschecks=40)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("stage", choices=["plan", "fetch", "audit"])
    parser.add_argument("--batch", type=int); parser.add_argument("--symbol")
    args = parser.parse_args()
    if args.stage == "fetch": fetch(args.batch, args.symbol)
    else: globals()[args.stage]()
