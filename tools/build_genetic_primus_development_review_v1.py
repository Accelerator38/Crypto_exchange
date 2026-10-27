"""Static report evidence inventory. Does not import/run any trading/research model."""

from hashlib import sha256
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Reports/Exia/Genetic_Primus"
OUTPUT = SOURCE / "development_review_v1_20261005"
RELEVANT = re.compile(r"return|profit|bps|lcb|drawdown|sharpe|cagr|trade|turnover|exposure|parameter|architecture|fitness|score|status|evidence|interval|start|end|gate|pass|fail|rejection|baseline|incumbent|paired|cost|positive|winner|best|validation|period|candidate|origin|cutoff|purge|feature|bootstrap|dsr|pbo|ess|support|signal|risk|commission|delay|strict|constraint|duration|fold|scaler|hidden|drift|concentration|genome|model|promotion", re.I)
EXCLUDED_ARRAYS = {"daily_returns", "returns", "weights", "resample_bank", "bootstrap_samples", "indices", "candles", "observations", "bars"}


def scalar_evidence(value, path="", depth=0):
    if depth > 10:
        return []
    if isinstance(value, dict):
        output = []
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else key
            if key in EXCLUDED_ARRAYS and isinstance(child, (list, dict)):
                continue
            output.extend(scalar_evidence(child, child_path, depth+1))
        return output
    if isinstance(value, list):
        output = []
        if len(value) > 64:
            return [{"path": path+".__omitted_array_length", "value": len(value)}]
        for index, child in enumerate(value):
            output.extend(scalar_evidence(child, f"{path}[{index}]", depth+1))
        return output
    if RELEVANT.search(path):
        return [{"path": path, "value": value}]
    return []


def selected(path):
    # Include origin receipts/results and verification summaries, not only common basenames.
    return path.suffix == ".json"


def main():
    paths = sorted(p for p in SOURCE.rglob("*.json") if OUTPUT not in p.parents)
    records, skipped = [], []
    for path in paths:
        if not selected(path):
            continue
        if path.stat().st_size > 2_000_000:
            skipped.append({"path": str(path.relative_to(ROOT)), "reason": "size cap 2MB"})
            continue
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError):
            skipped.append({"path": str(path.relative_to(ROOT)), "reason": "not parseable JSON"})
            continue
        scalars = scalar_evidence(value)
        records.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(raw).hexdigest(),
                        "size_bytes": len(raw), "top_level_keys": list(value) if isinstance(value, dict) else [],
                        "scalar_evidence": scalars})
    index = {"schema": "exia.genetic_primus.static_development_review/1",
             "review_date_utc": "2026-10-05", "all_genetic_json_files_count": len(paths),
             "selected_json_files_reviewed": len(records), "skipped": skipped,
             "scope": "Static JSON artifact inventory; no new training, prices decoded, runtime, network or tests",
             "limitations": ["Not every binary/parquet/plot independently recalculated",
                             "Inventory hashes show current local bytes, not signed provenance",
                             "Large arrays intentionally omitted; full originals retained",
                             "Historical adaptive trials may be missing from the repository"],
             "records": records}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "evidence_index_v2.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(index, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"genetic_json_files": len(paths), "reviewed": len(records), "skipped": len(skipped),
                      "index_bytes": target.stat().st_size, "training_runs": 0, "network": False}))


if __name__ == "__main__":
    main()
