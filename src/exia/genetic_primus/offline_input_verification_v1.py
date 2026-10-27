"""Pure timestamp/symbol presence checks. No prices, model, network or decoder."""
from __future__ import annotations

from datetime import datetime, timezone

BAR_MS = 14_400_000
MAX_ROWS = 200_000
MAX_TIMESTAMPS = 20_000


def schema_layout(names, core):
    if type(names) is not list or type(core) is not list or len(core) != 7:
        raise ValueError("column lists required")
    if any(type(name) is not str for name in names) or len(set(names)) != len(names):
        raise ValueError("unique physical columns required")
    if not set(core).issubset(names) or set(names) - set(core) - {"source_rows"}:
        raise ValueError("unregistered physical columns/missing core")
    return {"physical_columns": list(names), "normalized_core_columns": list(core),
            "nonfeature_extra_columns": [name for name in names if name not in core],
            "extra_values_decoded": False, "extra_columns_are_model_features": False}


def integer(value, minimum=0):
    if type(value) is not int or not minimum <= value <= 2**53 - 1:
        raise ValueError("bounded integer required")
    return value


def utc(value):
    if value is None:
        return None
    return datetime.fromtimestamp(value // 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def metadata(meta):
    required = {"columns", "coverage_end_exclusive", "coverage_start", "row_count", "symbols",
                "timeframe_ms", "timestamp_count"}
    if type(meta) is not dict or set(meta) != required:
        raise ValueError("exact manifest metadata required")
    if type(meta["timeframe_ms"]) is not int or meta["timeframe_ms"] != BAR_MS:
        raise ValueError("only frozen 4h grid allowed")
    start, end = integer(meta["coverage_start"]), integer(meta["coverage_end_exclusive"])
    if start % BAR_MS or end % BAR_MS or end <= start:
        raise ValueError("aligned positive coverage required")
    n = (end - start) // BAR_MS
    if not 1 <= n <= MAX_TIMESTAMPS or integer(meta["timestamp_count"], 1) != n:
        raise ValueError("coverage/timestamp count mismatch")
    symbols = meta["symbols"]
    if type(symbols) is not list or not 1 <= len(symbols) <= 8 or any(type(s) is not str or not 1 <= len(s) <= 24 for s in symbols):
        raise ValueError("bounded reference symbols required")
    if len(set(symbols)) != len(symbols) or integer(meta["row_count"], 1) != n * len(symbols):
        raise ValueError("reference grid/universe count mismatch")
    if meta["columns"] != ["timestamp", "symbol", "open", "high", "low", "close", "volume"]:
        raise ValueError("manifest column basis mismatch")
    return start, end, n, symbols


def key_summary(rows, meta):
    start, end, n, symbols = metadata(meta)
    masks = {s: 0 for s in symbols}
    previous = {}
    counts = {key: 0 for key in ("rows", "valid_key_rows", "null_timestamp", "null_symbol",
                                "invalid_timestamp_type", "unknown_symbol", "off_grid_timestamp",
                                "outside_manifest_coverage", "duplicate_keys", "regressions_per_symbol")}
    for item in rows:
        if type(item) is not tuple or len(item) != 2:
            raise ValueError("only timestamp/symbol pairs allowed")
        ts, symbol = item
        counts["rows"] += 1
        if counts["rows"] > MAX_ROWS:
            raise ValueError("projected row cap exceeded")
        invalid = False
        if ts is None:
            counts["null_timestamp"] += 1
            invalid = True
        elif type(ts) is not int or not 0 <= ts <= 2**53 - 1:
            counts["invalid_timestamp_type"] += 1
            invalid = True
        if symbol is None:
            counts["null_symbol"] += 1
            invalid = True
        elif type(symbol) is not str or symbol not in masks:
            counts["unknown_symbol"] += 1
            invalid = True
        if invalid:
            continue
        if not start <= ts < end:
            counts["outside_manifest_coverage"] += 1
            continue
        if (ts - start) % BAR_MS:
            counts["off_grid_timestamp"] += 1
            continue
        counts["valid_key_rows"] += 1
        if symbol in previous and ts < previous[symbol]:
            counts["regressions_per_symbol"] += 1
        previous[symbol] = ts
        bit = 1 << ((ts - start) // BAR_MS)
        if masks[symbol] & bit:
            counts["duplicate_keys"] += 1
        masks[symbol] |= bit
    details = {}
    union = 0
    for symbol, mask in masks.items():
        union |= mask
        first = (mask & -mask).bit_length() - 1 if mask else None
        last = mask.bit_length() - 1 if mask else None
        details[symbol] = {"unique_keys": mask.bit_count(),
                           "first_timestamp_ms": start + first * BAR_MS if first is not None else None,
                           "last_timestamp_ms": start + last * BAR_MS if last is not None else None,
                           "internal_missing_bars": last - first + 1 - mask.bit_count() if mask else 0}
    first = (union & -union).bit_length() - 1 if union else None
    last = union.bit_length() - 1 if union else None
    unique = sum(mask.bit_count() for mask in masks.values())
    errors = ("null_timestamp", "null_symbol", "invalid_timestamp_type", "unknown_symbol",
              "off_grid_timestamp", "outside_manifest_coverage", "duplicate_keys")
    return {"schema": "exia.genetic_primus.offline_key_summary/1", "counts": counts,
            "unique_keys": unique, "timestamps_in_file": union.bit_count(), "symbols_present": sum(bool(m) for m in masks.values()),
            "first_timestamp_ms": start + first * BAR_MS if first is not None else None,
            "last_timestamp_ms": start + last * BAR_MS if last is not None else None,
            "first_timestamp_utc": utc(start + first * BAR_MS) if first is not None else None,
            "last_timestamp_utc": utc(start + last * BAR_MS) if last is not None else None,
            "per_symbol": details, "presence_hex": {symbol: format(mask, "x") for symbol, mask in masks.items()},
            "grid_basis": "integer_epoch_ms_interpretation_checked_against_legacy_manifest_not_arrival_proof",
            "row_order_is_not_arrival_order": True,
            "key_checks_passed": bool(unique) and not any(counts[key] for key in errors),
            "price_values_read": False, "availability_provenance_verified": False, "fit_allowed": False}


def aggregate(files, meta):
    start, end, n, symbols = metadata(meta)
    if type(files) is not list or not 1 <= len(files) <= 16:
        raise ValueError("bounded file audit list required")
    union = {s: 0 for s in symbols}
    total_rows = total_valid = intra_duplicates = 0
    all_keys_passed = True
    for result in files:
        masks = result["presence_hex"]
        if type(masks) is not dict or set(masks) != set(symbols):
            raise ValueError("presence universe substitution")
        all_keys_passed = all_keys_passed and result["key_checks_passed"] is True
        total_rows += integer(result["counts"]["rows"])
        total_valid += integer(result["counts"]["valid_key_rows"])
        intra_duplicates += integer(result["counts"]["duplicate_keys"])
        unique = 0
        for symbol, encoded in masks.items():
            if type(encoded) is not str or not 1 <= len(encoded) <= (n + 3) // 4 or any(c not in "0123456789abcdef" for c in encoded):
                raise ValueError("bounded canonical presence hex required")
            mask = int(encoded, 16)
            if mask.bit_length() > n or format(mask, "x") != encoded:
                raise ValueError("presence outside coverage/noncanonical")
            unique += mask.bit_count()
            union[symbol] |= mask
        if unique != integer(result["unique_keys"]) or unique + result["counts"]["duplicate_keys"] != result["counts"]["valid_key_rows"]:
            raise ValueError("presence/count substitution")
    unique_total = sum(mask.bit_count() for mask in union.values())
    cross_duplicates = total_valid - intra_duplicates - unique_total
    if cross_duplicates < 0:
        raise ValueError("invalid duplicate arithmetic")
    missing = {s: n - mask.bit_count() for s, mask in union.items()}
    full = (1 << n) - 1
    return {"schema": "exia.genetic_primus.offline_key_aggregate/1", "files": len(files),
            "total_rows": total_rows, "expected_rows": meta["row_count"], "unique_keys": unique_total,
            "expected_timestamps": n, "intra_file_duplicate_keys": intra_duplicates,
            "cross_file_duplicate_keys": cross_duplicates, "missing_keys_per_symbol": missing,
            "missing_keys_total": sum(missing.values()), "all_file_key_checks_passed": all_keys_passed,
            "complete_common_grid": all(mask == full for mask in union.values()),
            "key_structure_verified": all_keys_passed and total_rows == meta["row_count"] and
                                      unique_total == meta["row_count"] and cross_duplicates == 0 and
                                      intra_duplicates == 0 and not any(missing.values()),
            "coverage_start_utc": utc(start), "coverage_end_exclusive_utc": utc(end),
            "arrival_revision_venue_product_provenance_verified": False,
            "ohlcv_price_quality_verified": False, "fit_allowed": False}
