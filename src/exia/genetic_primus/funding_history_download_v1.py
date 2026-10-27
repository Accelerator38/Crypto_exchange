"""Pure strict funding-page parsing and bounded interval audit; no network/I/O."""
from __future__ import annotations

from decimal import Decimal, localcontext
import hashlib
import json
import re

SYMBOLS = ("ADAUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "ETHUSDT", "LINKUSDT", "SOLUSDT", "XRPUSDT")
START_MS = 1_785_542_400_000
END_MS = 1_790_812_800_000
HOUR_MS = 3_600_000
MAX_PAGES = 32
PAGE_SIZE = 100
MAX_RESPONSE_BYTES = 131072
ENDPOINT = "https://api.bitget.com/api/v2/mix/market/history-fund-rate"
DOC_URL = "https://www.bitget.com/api-doc/classic/contract/market/Get-History-Funding-Rate"
SAFETY = dict.fromkeys(("credentials_allowed", "orders_enabled", "execution_allowed", "paper", "live",
                       "promotion_authority", "training_allowed", "financial_evaluation_allowed", "network_allowed"), False)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def strict_json(raw):
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_RESPONSE_BYTES:
        raise ValueError("bounded nonempty response bytes required")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON constant")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)


def timestamp(value):
    if type(value) is str and re.fullmatch(r"[0-9]{1,16}", value):
        value = int(value)
    if type(value) is not int or not 0 < value < 2**53:
        raise ValueError("positive integer millisecond timestamp required")
    return value


def decimal_rate(value):
    if type(value) is not str or not re.fullmatch(r"-?[0-9]{1,3}(?:\.[0-9]{1,18})?", value):
        raise ValueError("bounded exact decimal rate string required")
    number = Decimal(value)
    if not -1 < number < 1:
        raise ValueError("funding rate outside strict bounded fraction range")
    with localcontext() as context:
        context.prec = 50
        canonical = "0" if number == 0 else format(number.normalize(), "f")
        bps = "0" if number == 0 else format((number*10000).normalize(), "f")
    return canonical, bps


def parse_page(payload, symbol):
    if symbol not in SYMBOLS or not isinstance(payload, dict) or payload.get("code") != "00000":
        raise ValueError("known symbol and successful Bitget envelope required")
    server_ms = timestamp(payload.get("requestTime"))
    rows = payload.get("data")
    if type(rows) is not list or len(rows) > PAGE_SIZE:
        raise ValueError("bounded v2 funding data list required")
    result = []
    last = None
    for row in rows:
        if type(row) is not dict or row.get("symbol") != symbol:
            raise ValueError("response symbol mismatch")
        t = timestamp(row.get("fundingTime"))
        if t > server_ms or (last is not None and t >= last):
            raise ValueError("settled descending unique history required")
        rate, bps = decimal_rate(row.get("fundingRate"))
        result.append({"symbol": symbol, "settlement_ms": t, "rate_fraction": rate,
                       "rate_bps": bps, "source_rate_string": row["fundingRate"]})
        last = t
    return {"server_request_ms": server_ms, "rows": result, "row_count": len(result),
            "newest_ms": result[0]["settlement_ms"] if result else None,
            "oldest_ms": result[-1]["settlement_ms"] if result else None,
            "content_sha256": identity(result)}


def audit_symbol(pages, symbol):
    if symbol not in SYMBOLS or not 1 <= len(pages) <= MAX_PAGES:
        raise ValueError("bounded known-symbol page bank required")
    events, duplicate_count, issues = {}, 0, []
    oldest_previous = None
    exhausted = False
    server_times = []
    for expected_page, page in enumerate(pages, 1):
        if page["page_no"] != expected_page:
            raise ValueError("contiguous page numbers from one required")
        parsed = parse_page(page["payload"], symbol)
        server_times.append(parsed["server_request_ms"])
        if exhausted:
            raise ValueError("pages after endpoint exhaustion")
        if parsed["oldest_ms"] is not None:
            if oldest_previous is not None and parsed["oldest_ms"] >= oldest_previous:
                raise ValueError("pagination did not move into older history")
            oldest_previous = parsed["oldest_ms"]
        for row in parsed["rows"]:
            t = row["settlement_ms"]
            if t in events:
                if events[t]["rate_fraction"] != row["rate_fraction"]:
                    raise ValueError("conflicting rate for duplicate settlement")
                duplicate_count += 1
            else:
                events[t] = row
        exhausted = parsed["row_count"] < PAGE_SIZE
    ordered = [events[t] for t in sorted(events)]
    interval = [r for r in ordered if START_MS <= r["settlement_ms"] < END_MS]
    left = [r for r in ordered if r["settlement_ms"] <= START_MS]
    right = [r for r in ordered if r["settlement_ms"] >= END_MS]
    boundary_bracketed = bool(left and right)
    if not left:
        issues.append("start_boundary_not_bracketed")
    if not right:
        issues.append("end_boundary_not_bracketed")
    if not interval:
        issues.append("no_requested_interval_records")
    visible = [r for r in ordered if (left[-1]["settlement_ms"] if left else START_MS) <= r["settlement_ms"]
               <= (right[0]["settlement_ms"] if right else END_MS)]
    gap_counts = {}
    for a, b in zip(visible, visible[1:]):
        gap = b["settlement_ms"]-a["settlement_ms"]
        label = str(gap/HOUR_MS)
        gap_counts[label] = gap_counts.get(label, 0)+1
        if gap not in (HOUR_MS, 2*HOUR_MS, 4*HOUR_MS, 8*HOUR_MS):
            issues.append("unexplained_observed_gap")
    if any(r["settlement_ms"] % HOUR_MS for r in visible):
        issues.append("settlement_not_whole_utc_hour")
    if any(b < a for a, b in zip(server_times, server_times[1:])):
        issues.append("server_request_clock_reversed")
    return {"symbol": symbol, "pages": len(pages), "events": interval,
            "requested_rows": len(interval), "observed_rows": len(ordered), "duplicates_equal_rate": duplicate_count,
            "newest_observed_ms": ordered[-1]["settlement_ms"] if ordered else None,
            "oldest_observed_ms": ordered[0]["settlement_ms"] if ordered else None,
            "left_guard_ms": left[-1]["settlement_ms"] if left else None,
            "right_guard_ms": right[0]["settlement_ms"] if right else None,
            "boundary_bracketed": boundary_bracketed,
            "observed_gap_hours": gap_counts, "issues": sorted(set(issues)),
            "api_enumeration_ready": boundary_bracketed and not issues,
            "settlement_schedule_independently_verified": False,
            "settlement_marks_present": False, "financial_replay_ready": False,
            "endpoint_exhausted": exhausted,
            "more_pages_useful": not left and not exhausted and len(pages) < MAX_PAGES,
            "server_latest_predates_requested_interval": bool(ordered and ordered[-1]["settlement_ms"] < START_MS)}
