"""Finite, offline direction evaluation. No market access or execution adapter."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import zipfile

import numpy as np

from . import direction_selective_v1 as selective
from . import direction_hypotheses_v1 as estimators

MINUTE = 60_000
DAY = 86_400_000
SAFETY = ("credentials_allowed", "orders_enabled", "paper", "live",
          "runtime_authority", "network_allowed", "promotion_authority")
PRICE_FEATURES = ("return_1m", "return_5m", "return_30m", "volatility_30m",
                  "body_over_range", "log1p_volume")
BOOK_FEATURES = ("spread_bps", "log1p_bid_depth5", "log1p_ask_depth5",
                 "depth5_imbalance", "depth_change", "microprice_offset_bps",
                 "book_age_ms")
REQUIRED_ARRAYS = ("times", "price", "book", "last_trade", "midprice", "bid", "ask",
                   "last_trade_receipt_us", "book_receipt_us", "feature_max_receipt_us")


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_contract(contract):
    if any(contract.get(k) is not False for k in SAFETY):
        raise ValueError("all seven safety permissions must explicitly be false")
    if (contract["horizon_minutes"], contract["decision_minutes"],
            contract["retrain_days"], contract["purge_bars"]) != (1, 1, 3, 7):
        raise ValueError("unregistered timing configuration")
    if contract["targets"] != ["last_trade", "midprice"]:
        raise ValueError("target substitution is forbidden")
    if contract["models"] != ["price_logit", "price_boost", "book_logit", "book_boost"]:
        raise ValueError("unregistered model family")


def load_day(path, contract):
    # A single standardized minute bank must remain small and contain no pickle.
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 2_000_000:
            raise ValueError("day bank exceeds bounded uncompressed size")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(REQUIRED_ARRAYS):
            raise ValueError("day bank schema mismatch")
        bank = {key: archive[key].copy() for key in REQUIRED_ARRAYS}
    return bank, validate_day(bank, contract)


def validate_day(bank, contract):
    check_contract(contract)
    times = np.asarray(bank["times"])
    if times.dtype.kind not in "iu" or times.shape != (1440,):
        raise ValueError("1440 integer receipt-clock minute boundaries required")
    start = int(times[0]) - MINUTE
    if start % DAY or not np.array_equal(times, start + np.arange(1, 1441) * MINUTE):
        raise ValueError("one complete contiguous UTC source day required")
    for key in REQUIRED_ARRAYS:
        value = np.asarray(bank[key])
        shape = (1440, 6) if key == "price" else (1440, 13) if key == "book" else (1440,)
        if value.shape != shape or value.dtype.kind not in "fiub":
            raise ValueError("bad day array: " + key)
    for key in ("last_trade_receipt_us", "book_receipt_us", "feature_max_receipt_us"):
        receipt = bank[key]
        if receipt.dtype.kind not in "iu" or np.any(receipt >= times * 1000):
            raise ValueError("future/exact-boundary receipt: " + key)
        if np.any(receipt < start * 1000):
            raise ValueError("receipt outside this source day")
    if np.any(bank["feature_max_receipt_us"] < np.maximum(
            bank["last_trade_receipt_us"], bank["book_receipt_us"])):
        raise ValueError("feature receipt envelope must cover both inputs")
    if not np.array_equal(bank["price"], bank["book"][:, :6], equal_nan=True):
        raise ValueError("price-only and book models must share identical price features")
    for key in ("last_trade", "midprice", "bid", "ask"):
        if not np.isfinite(bank[key]).all() or np.any(bank[key] <= 0):
            raise ValueError("invalid price: " + key)
    if np.any(bank["bid"] >= bank["ask"]):
        raise ValueError("crossed or locked quote")
    if not np.allclose(bank["midprice"], (bank["bid"] + bank["ask"]) / 2, rtol=1e-12):
        raise ValueError("midprice must equal causal bid/ask midpoint")
    book_age = (times * 1000 - bank["book_receipt_us"]) / 1000
    trade_age = (times * 1000 - bank["last_trade_receipt_us"]) / 1000
    finite = np.isfinite(bank["book"]).all(axis=1)
    if not np.allclose(bank["book"][finite, 12], book_age[finite], atol=1e-6):
        raise ValueError("book age feature does not match receipt")
    spread = (bank["ask"] - bank["bid"]) / bank["midprice"] * 10_000
    if not np.allclose(bank["book"][finite, 6], spread[finite], atol=1e-7):
        raise ValueError("book spread feature does not match bid/ask")
    eligible = finite & (book_age <= contract["maximum_book_age_ms"]) & (
        trade_age <= contract["maximum_trade_age_ms"])
    eligible[:30] = False  # Explicit daily restart; never forward-fill gaps.
    return {"utc_start_ms": start, "minutes": 1440,
            "eligible_minutes": int(eligible.sum()), "eligible": eligible,
            "stale_trade_minutes": int((trade_age > contract["maximum_trade_age_ms"]).sum()),
            "stale_book_minutes": int((book_age > contract["maximum_book_age_ms"]).sum())}


def direction_labels(prices, times, eligible, horizon_minutes=1):
    """Flat remains zero. Labels never jump over gaps or source-day boundaries."""
    p, t, eligible = np.asarray(prices), np.asarray(times), np.asarray(eligible, bool)
    h = int(horizon_minutes)
    if h < 1 or p.shape != t.shape or p.shape != eligible.shape or p.ndim != 1:
        raise ValueError("invalid label arrays")
    labels, valid = np.zeros(len(t), np.int8), np.zeros(len(t), bool)
    if len(t) <= h:
        return labels, valid
    rows = np.arange(len(t) - h)
    # Last boundary at midnight belongs to the source day that just ended.
    same_day = (t[rows] - 1) // DAY == (t[rows + h] - 1) // DAY
    contiguous = t[rows + h] - t[rows] == h * MINUTE
    valid[rows] = eligible[rows] & eligible[rows+h] & same_day & contiguous
    labels[rows] = np.sign(p[rows+h] - p[rows]).astype(np.int8)
    return labels, valid


def split_origin(times, common_valid, origin, contract):
    t = np.asarray(times)
    horizon = contract["horizon_minutes"] * MINUTE
    purge = contract["purge_bars"] * MINUTE
    cal_start = origin - contract["calibration_days"] * DAY
    train = np.flatnonzero(common_valid & (t >= origin-contract["train_days"]*DAY)
                          & (t+horizon < cal_start-purge))
    cal = np.flatnonzero(common_valid & (t >= cal_start) & (t+horizon < origin-purge))
    test = np.flatnonzero(common_valid & (t >= origin)
                         & (t < origin+contract["retrain_days"]*DAY))
    if len(train) > contract["train_time_cap"]:
        rng = np.random.default_rng(contract["selection_seed"] ^ (int(origin) // MINUTE))
        train = np.sort(rng.choice(train, contract["train_time_cap"], replace=False))
    if not len(train) or len(cal) < contract["minimum_calibration_rows"] or not len(test):
        raise ValueError("insufficient causal train/calibration/test observations")
    return train, cal, test


def fold_receipt(times, train, cal, test, origin, contract):
    return {"origin_ms": int(origin), "horizon_minutes": contract["horizon_minutes"],
            "purge_bars": contract["purge_bars"], "retrain_days": contract["retrain_days"],
            "train_label_cutoff_exclusive_ms": int(origin-contract["calibration_days"]*DAY
                                                     -contract["purge_bars"]*MINUTE),
            "calibration_label_cutoff_exclusive_ms": int(origin-contract["purge_bars"]*MINUTE),
            "train_times_sha256": sha256(np.asarray(times[train], dtype="<i8").tobytes()).hexdigest(),
            "calibration_times_sha256": sha256(np.asarray(times[cal], dtype="<i8").tobytes()).hexdigest(),
            "test_times_sha256": sha256(np.asarray(times[test], dtype="<i8").tobytes()).hexdigest(),
            "rows": {"train": len(train), "calibration": len(cal), "test": len(test)}}


def fit_target(kind, features, labels, train, contract, seed):
    nonflat = train[np.asarray(labels)[train] != 0]
    if len(nonflat) < contract["minimum_binary_train_rows"]:
        raise ValueError("too few nonflat training labels")
    # Matched price/book models use exactly the same target-specific training rows.
    model = estimators.fit_binary(kind, features[nonflat], (labels[nonflat] > 0).astype(int),
                                  seed, minimum_leaf=contract["boost"]["min_samples_leaf"])
    model["actual_train_times_index_sha256"] = sha256(np.asarray(nonflat, dtype="<i8").tobytes()).hexdigest()
    model["flat_train_rows_excluded"] = int(len(train)-len(nonflat))
    return model


def controls(times, prices, labels, train, seed):
    majority_up = int(np.count_nonzero(labels[train] > 0)) >= int(np.count_nonzero(labels[train] < 0))
    random_up = (selective.tie_keys(times, "BTCUSDT_random", seed) & 1).astype(bool)
    momentum = np.r_[True, np.diff(prices) >= 0]
    return {"random": random_up, "majority_train_only": np.full(len(times), majority_up),
            "momentum_1m": momentum}


def qualify_entries(entries, contract):
    """Fail closed; metadata is an auditable claim, not an exchange completeness attestation."""
    check_contract(contract)
    if len(entries) > contract["train_days"] + contract["test_days"]:
        raise ValueError("more than the registered 60 days is a different experiment")
    starts, reports = [], []
    for entry in entries:
        if entry["asset"] != contract["asset"] or entry["exchange"] != contract["exchange"]:
            raise ValueError("asset/exchange mismatch")
        path = Path(entry["bank_path"])
        if file_sha(path) != entry["bank_sha256"]:
            raise ValueError("minute bank hash mismatch")
        if set(entry["sources"]) != {"trades", "book"}:
            raise ValueError("matched trades and book source pins required")
        for source in entry["sources"].values():
            if file_sha(source["path"]) != source["sha256"]:
                raise ValueError("raw market source hash mismatch")
        if entry.get("book_mode") not in ("atomic_full_snapshots", "verified_incremental"):
            raise ValueError("unknown snapshot/delta semantics")
        if entry["book_mode"] == "verified_incremental" and entry.get("sequence_verified") is not True:
            raise ValueError("incremental sequence continuity unverified")
        if entry.get("clock") != "local_receipt_strict_before_boundary":
            raise ValueError("receipt-clock provenance required")
        if not entry.get("builder_code_pins"):
            raise ValueError("feature builder provenance required")
        for source_path, expected in entry["builder_code_pins"].items():
            if file_sha(source_path) != expected:
                raise ValueError("feature builder code changed")
        _, report = load_day(path, contract)
        report.pop("eligible")
        if report["eligible_minutes"] < contract["minimum_eligible_minutes_per_day"]:
            raise ValueError("too few fresh jointly eligible minutes")
        starts.append(report["utc_start_ms"])
        reports.append(report)
    if starts != sorted(set(starts)) or any(b-a != DAY for a, b in zip(starts, starts[1:])):
        raise ValueError("duplicate, unsorted or missing calendar day")
    required = contract["train_days"] + contract["test_days"]
    ready = len(starts) == required
    if ready and any(e.get("outcomes_already_examined", False) for e in entries[contract["train_days"]:]):
        raise ValueError("known engineering outcomes cannot be relabeled as a new test")
    return {"status": "READY_TO_FREEZE" if ready else "DATA_INSUFFICIENT",
            "days_available": len(starts), "days_required": required,
            "missing_days": required-len(starts), "per_day": reports,
            "raw_features_recomputed_during_qualification": False,
            "exchange_sequence_completeness_attested": False}


def direct_checks(contract):
    """Small deterministic checks; no pytest, files, collector or wall-clock waits."""
    check_contract(contract)
    checks = []
    t = np.arange(1, 9) * MINUTE
    y, valid = direction_labels(np.array([1, 2, 2, 1, 3, 2, 2, 1.]), t, np.ones(8, bool))
    assert y[:3].tolist() == [1, 0, -1] and valid.sum() == 7
    checks.append("up_flat_down_and_last_endpoint")
    gap_t = np.array([MINUTE, 2*MINUTE, 4*MINUTE, DAY, DAY+MINUTE])
    _, mask = direction_labels(np.arange(1, 6.), gap_t, np.ones(5, bool))
    assert mask.tolist() == [True, False, False, False, False]
    checks.append("no_gap_or_cross_day_label")
    times = np.arange(1, 60*1440+1, dtype=np.int64)*MINUTE
    # Full synthetic calendar, with explicit source-day endpoint exclusions.
    labels, valid = direction_labels(100 + np.sin(np.arange(len(times))/5), times, np.ones(len(times), bool))
    origin = 30*DAY
    a, b, c = split_origin(times, valid, origin, contract)
    assert np.all(times[a]+MINUTE < origin-7*DAY-7*MINUTE)
    assert np.all(times[b]+MINUTE < origin-7*MINUTE)
    assert not set(a).intersection(b) and not set(b).intersection(c)
    assert len(a) == contract["train_time_cap"]
    checks.append("mature_labels_and_purge_on_both_boundaries")
    a2, b2, c2 = split_origin(times, valid, origin, contract)
    assert np.array_equal(a, a2) and np.array_equal(b, b2) and np.array_equal(c, c2)
    checks.append("deterministic_common_time_sample")
    suffix = valid.copy(); suffix[times >= origin] = False
    # Test availability changes must not change training/calibration selection.
    # Keep one future row, solely so split_origin can validate a nonempty test.
    suffix[c[0]] = True
    a3, b3, _ = split_origin(times, suffix, origin, contract)
    assert np.array_equal(a, a3) and np.array_equal(b, b3)
    checks.append("future_suffix_does_not_change_training")
    example = {"times": np.arange(1, 1441, dtype=np.int64)*MINUTE,
               "price": np.ones((1440, 6)), "book": np.ones((1440, 13)),
               "last_trade": np.full(1440, 100.), "midprice": np.full(1440, 100.),
               "bid": np.full(1440, 99.99), "ask": np.full(1440, 100.01)}
    for key in ("last_trade_receipt_us", "book_receipt_us", "feature_max_receipt_us"):
        example[key] = example["times"]*1000-1000
    example["book"][:, 6] = 2.
    report = validate_day(example, contract)
    assert report["eligible_minutes"] == 1410
    checks.append("bounded_bank_schema_and_receipt_checks")
    example["book_receipt_us"][80] = example["times"][80]*1000
    try:
        validate_day(example, contract)
        raise AssertionError("exact-boundary event admitted")
    except ValueError:
        pass
    checks.append("future_or_exact_boundary_receipt_rejected")
    bad_contract = dict(contract, orders_enabled=True)
    try:
        check_contract(bad_contract)
        raise AssertionError("unsafe contract admitted")
    except ValueError:
        pass
    checks.append("unsafe_permissions_rejected")
    return checks
