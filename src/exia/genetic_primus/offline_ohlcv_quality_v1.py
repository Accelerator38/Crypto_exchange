"""Counter-only OHLCV checks. No decoder, clock, package or market imports."""
from __future__ import annotations

import math

FIELDS = ("open", "high", "low", "close", "volume", "source_rows")
COUNTERS = ("rows", "invalid_rows", "ohlc_inconsistent_rows", "zero_volume_rows",
            "source_rows_invalid", "source_rows_not_240") + tuple(
                f"{name}_{kind}" for name in FIELDS[:5]
                for kind in ("null", "non_numeric", "nonfinite", "domain_invalid"))


def empty():
    return dict.fromkeys(COUNTERS, 0)


def summarize(rows):
    """Exact inequalities; no fitted tolerances or price examples are retained."""
    out = empty()
    for row in rows:
        if len(row) != 6:
            raise ValueError("six fields required")
        out["rows"] += 1
        invalid = False
        finite = []
        for name, value in zip(FIELDS[:5], row[:5]):
            kind = None
            if value is None:
                kind = "null"
            elif type(value) not in (int, float):
                kind = "non_numeric"
            else:
                try:
                    if not math.isfinite(value):
                        kind = "nonfinite"
                except OverflowError:
                    kind = "nonfinite"
            finite.append(kind is None)
            if kind:
                out[f"{name}_{kind}"] += 1
                invalid = True
            elif (value < 0 if name == "volume" else value <= 0):
                out[f"{name}_domain_invalid"] += 1
                invalid = True
        if all(finite[:4]):
            o, h, l, c = row[:4]
            if h < max(o, l, c) or l > min(o, h, c):
                out["ohlc_inconsistent_rows"] += 1
                invalid = True
        if finite[4] and row[4] == 0:
            out["zero_volume_rows"] += 1
        source = row[5]
        if type(source) is not int or source <= 0:
            out["source_rows_invalid"] += 1
            invalid = True
        elif source != 240:
            out["source_rows_not_240"] += 1
            invalid = True
        out["invalid_rows"] += int(invalid)
    return out


def combine(summaries):
    out = empty()
    for item in summaries:
        if set(item) != set(COUNTERS):
            raise ValueError("counter schema mismatch")
        for key in COUNTERS:
            value = item[key]
            if type(value) is not int or value < 0:
                raise ValueError("nonnegative integer required")
            out[key] += value
    if out["invalid_rows"] > out["rows"]:
        raise ValueError("invalid rows exceed rows")
    return out
