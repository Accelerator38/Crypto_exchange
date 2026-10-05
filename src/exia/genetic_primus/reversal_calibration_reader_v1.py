"""Bounded local snapshot reader for ONE revealed pilot; no runtime imports."""

from hashlib import sha256, file_digest
import json
from pathlib import Path

BAR_MS = 14_400_000
DAY_MS = 6 * BAR_MS
START_MS = 1780272000000
END_MS = START_MS + 60 * DAY_MS
TRAIN_ANCHOR_MS = START_MS - 729 * DAY_MS
LOWER_MS = TRAIN_ANCHOR_MS - 13 * BAR_MS
MAX_ROWS = 40_000
UNIVERSE = ("ADA/USDT", "BNB/USDT", "BTC/USDT", "DOGE/USDT",
            "ETH/USDT", "LINK/USDT", "SOL/USDT", "XRP/USDT")
MANIFEST = "Reports/Exia/Genetic_Primus/prospective_3d_v1_202609/data_manifest_origin_09.json"
MANIFEST_SHA = "72631df4a0d5abca0d34319170ed0e5b1c64e5562659c89cc130a2b51982fdd9"
SNAPSHOT_SHA = "f548b6af6034b3d59e0c3c64aad287ca2a9b762bca5051331db2e72406861bda"


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()).hexdigest()


def bounds(start_ms, end_ms):
    if (type(start_ms) is not int or type(end_ms) is not int
            or not LOWER_MS <= start_ms < end_ms <= END_MS
            or start_ms % BAR_MS or end_ms % BAR_MS):
        raise ValueError("outside registered pilot input bounds")
    expected = (end_ms-start_ms)//BAR_MS * len(UNIVERSE)
    if expected > MAX_ROWS:
        raise ValueError("bounded row budget exceeded")
    return expected


def local_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("snapshot path outside workspace or missing")
    return path


def verify_snapshot(root):
    manifest_path = local_path(root, MANIFEST)
    raw = manifest_path.read_bytes()
    if sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError("manifest bytes changed")
    manifest = json.loads(raw)
    entries = manifest["snapshot_files"]
    if len(entries) != 11 or digest(entries) != SNAPSHOT_SHA:
        raise ValueError("snapshot commitment changed")
    for entry in entries:
        path = local_path(root, entry["path"])
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError("snapshot size changed")
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                raise ValueError("snapshot bytes changed")
    return entries


def validate_rows(rows, *, start_ms, end_ms, candle_type):
    expected = bounds(start_ms, end_ms)
    if len(rows) != expected:
        raise ValueError("missing/extra full8 candles")
    seen, candles = set(), []
    for row in rows:
        t, symbol = row["timestamp"], row["symbol"]
        if (type(t) is not int or not start_ms <= t < end_ms
                or t % BAR_MS or symbol not in UNIVERSE or (t, symbol) in seen):
            raise ValueError("invalid timestamp/symbol/duplicate")
        seen.add((t, symbol))
        candles.append(candle_type(symbol=symbol, open_ms=t,
                                  **{k: row[k] for k in ("open", "high", "low", "close", "volume")}))
    candles.sort(key=lambda c: (c.open_ms, c.symbol))
    return candles


def read_bounded(root, *, start_ms, end_ms, candle_type):
    expected = bounds(start_ms, end_ms)  # MUST precede parquet decoding.
    entries = verify_snapshot(root)  # Actual file bytes, not just manifest claims.
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = []
    columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
    for entry in entries:
        path = local_path(root, entry["path"])
        schema = pq.ParquetFile(path).schema_arrow
        if schema.field("timestamp").type != pa.int64():
            raise ValueError("unexpected timestamp schema")
        table = pq.read_table(path, columns=columns,
                              filters=[("timestamp", ">=", start_ms), ("timestamp", "<", end_ms)])
        if len(rows)+table.num_rows > expected:
            raise ValueError("duplicate/oversized filtered panel")
        rows.extend(table.to_pylist())
    candles = validate_rows(rows, start_ms=start_ms, end_ms=end_ms, candle_type=candle_type)
    # Detect replacement during read as well. This is local integrity, not signed provenance.
    verify_snapshot(root)
    return candles
