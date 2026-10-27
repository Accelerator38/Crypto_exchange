from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exia.genetic_primus.contracts import sha256_file  # noqa: E402
from exia.genetic_primus.contracts_v5 import load_v5_contract  # noqa: E402
from exia.genetic_primus.data_source_v5 import (  # noqa: E402
    aggregate_symbol_metrics,
    pack_raw_directory,
    parse_checksum,
    read_metric_zip,
    sha256_bytes,
)


DEFAULT_CONTRACT = ROOT / "configs/genetic_primus_v5_contract.json"
DEFAULT_OUTPUT = ROOT / "Retrodate/genetic_primus_v5_binance_metrics_v1_20220101_20241231"


def _fetch(url: str, *, attempts: int = 5) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"User-Agent": "Exia-V5-Research/1.0"})
            with urlopen(request, timeout=45) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            error = exc
            if isinstance(exc, HTTPError) and exc.code == 404:
                break
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"failed to fetch {url}: {error}")


def _download_one(item: dict[str, str], raw_dir: Path) -> dict[str, Any]:
    target = raw_dir / item["exchange_symbol"] / item["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    checksum_payload = _fetch(item["url"] + ".CHECKSUM")
    official_sha = parse_checksum(checksum_payload, item["filename"])
    if target.is_file() and sha256_file(target) == official_sha:
        payload_sha = official_sha
    else:
        payload = _fetch(item["url"])
        payload_sha = sha256_bytes(payload)
        if payload_sha != official_sha:
            raise ValueError(f"official checksum mismatch: {item['filename']}")
        temporary = target.with_suffix(".zip.tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
    return {
        **item,
        "local_path": target.relative_to(raw_dir).as_posix(),
        "bytes": target.stat().st_size,
        "sha256": payload_sha,
        "official_sha256": official_sha,
        "downloaded": True,
        "error": None,
    }


def _items(contract) -> list[dict[str, str]]:
    source = contract.source
    dates = pd.date_range(source["start_date"], source["end_date"], freq="1D")
    rows = []
    for target_symbol, exchange_symbol in source["symbols"].items():
        for timestamp in dates:
            date_text = timestamp.strftime("%Y-%m-%d")
            filename = f"{exchange_symbol}-metrics-{date_text}.zip"
            rows.append(
                {
                    "symbol": target_symbol,
                    "exchange_symbol": exchange_symbol,
                    "date": date_text,
                    "filename": filename,
                    "url": source["url_template"].format(exchange_symbol=exchange_symbol, date=date_text),
                }
            )
    return rows


def _catalog_workbook(path: Path, catalog: pd.DataFrame, coverage: pd.DataFrame) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        catalog.to_excel(writer, sheet_name="Raw Files", index=False)
        coverage.to_excel(writer, sheet_name="Coverage", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions


def build(contract_path: Path, output_dir: Path, *, workers: int, keep_raw: bool) -> dict[str, Any]:
    contract = load_v5_contract(contract_path, root=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("contract_sha256") == contract.sha256:
            return existing
        raise ValueError("stale V5 dataset manifest uses a different contract")
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    items = _items(contract)
    catalog_path = output_dir / "raw_catalog.parquet"
    reusable: dict[tuple[str, str], dict[str, Any]] = {}
    if catalog_path.is_file():
        previous = pd.read_parquet(catalog_path)
        for row in previous.loc[previous["downloaded"] == True].to_dict(orient="records"):  # noqa: E712
            path = raw_dir / str(row["local_path"])
            if path.is_file() and sha256_file(path) == row.get("official_sha256"):
                reusable[(str(row["exchange_symbol"]), str(row["date"]))] = row
    rows: list[dict[str, Any]] = list(reusable.values())
    failures: list[dict[str, Any]] = []
    pending = [item for item in items if (item["exchange_symbol"], item["date"]) not in reusable]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(_download_one, item, raw_dir): item for item in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({**item, "downloaded": False, "error": str(exc)})
            if index % 250 == 0 or index == len(pending):
                print(json.dumps({"downloaded": len(rows), "failed": len(failures), "total": len(items)}), flush=True)
    catalog = pd.DataFrame([*rows, *failures]).sort_values(["exchange_symbol", "date"], kind="stable")
    catalog.to_parquet(catalog_path, index=False)
    frames = []
    raw_rows = 0
    repaired_order_paths: set[str] = set()
    for target_symbol, exchange_symbol in contract.source["symbols"].items():
        symbol_frames = []
        subset = catalog.loc[(catalog["exchange_symbol"] == exchange_symbol) & (catalog["downloaded"] == True)]  # noqa: E712
        for relative in subset["local_path"].dropna().astype(str):
            frame = read_metric_zip(raw_dir / relative, expected_symbol=exchange_symbol)
            if frame.attrs.get("source_was_unsorted") is True:
                repaired_order_paths.add(relative)
            raw_rows += len(frame)
            symbol_frames.append(frame)
        if symbol_frames:
            frames.append(
                aggregate_symbol_metrics(
                    pd.concat(symbol_frames, ignore_index=True),
                    source_symbol=exchange_symbol,
                    target_symbol=target_symbol,
                    timeframe_minutes=int(contract.source["decision_timeframe_minutes"]),
                )
            )
    catalog["timestamp_order_repaired"] = catalog["local_path"].astype(str).isin(repaired_order_paths)
    catalog.to_parquet(catalog_path, index=False)
    dataset = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"], kind="stable")
    dataset_path = output_dir / "derivatives_4h.parquet"
    dataset.to_parquet(dataset_path, index=False)
    expected_files = len(items)
    file_coverage = len(rows) / expected_files
    expected_bars_per_symbol = len(
        pd.date_range(
            contract.source["start_date"],
            pd.Timestamp(contract.source["end_date"]) + pd.Timedelta(days=1),
            freq="4h",
            inclusive="left",
        )
    )
    coverage_rows = []
    for symbol, group in dataset.groupby("symbol", sort=True, observed=True):
        coverage_rows.append(
            {
                "symbol": str(symbol),
                "bars": int(len(group)),
                "expected_bars": int(expected_bars_per_symbol),
                "bar_coverage": float(len(group) / expected_bars_per_symbol),
                "complete_bar_rate": float(group["complete_bar"].mean()),
                **{
                    f"{feature}_coverage": float(group[feature].notna().mean())
                    for feature in contract.source["required_core_features"]
                },
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    workbook_path = output_dir / "dataset_catalog.xlsx"
    _catalog_workbook(workbook_path, catalog, coverage)
    archive_path = output_dir / "raw_metrics.zip"
    pack_raw_directory(raw_dir, archive_path)
    if not keep_raw:
        shutil.rmtree(raw_dir)
    artifact_shas = {
        "dataset": sha256_file(dataset_path),
        "raw_catalog": sha256_file(catalog_path),
        "raw_archive": sha256_file(archive_path),
        "workbook": sha256_file(workbook_path),
    }
    dataset_sha = hashlib.sha256(
        (contract.sha256 + "".join(artifact_shas[key] for key in sorted(artifact_shas))).encode("utf-8")
    ).hexdigest()
    manifest = {
        "schema_version": "exia.genetic_primus_derivatives_dataset.v5",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "V5_DERIVATIVES_DATASET_BUILT" if not failures else "V5_DERIVATIVES_DATASET_INCOMPLETE",
        "contract_sha256": contract.sha256,
        "source_id": contract.source["source_id"],
        "source_family": contract.source["source_family"],
        "provider": contract.source["provider"],
        "requested_start_date": contract.source["start_date"],
        "requested_end_date": contract.source["end_date"],
        "symbols": list(contract.source["symbols"]),
        "expected_raw_files": expected_files,
        "downloaded_raw_files": len(rows),
        "failed_raw_files": len(failures),
        "timestamp_order_repaired_files": len(repaired_order_paths),
        "raw_file_coverage": file_coverage,
        "raw_rows": int(raw_rows),
        "derived_rows": int(len(dataset)),
        "dataset_sha256": dataset_sha,
        "coverage": coverage.to_dict(orient="records"),
        "failures": failures,
        "artifacts": {
            "dataset": {"path": str(dataset_path.relative_to(ROOT)), "sha256": artifact_shas["dataset"]},
            "raw_catalog": {"path": str(catalog_path.relative_to(ROOT)), "sha256": artifact_shas["raw_catalog"]},
            "raw_archive": {"path": str(archive_path.relative_to(ROOT)), "sha256": artifact_shas["raw_archive"]},
            "workbook": {"path": str(workbook_path.relative_to(ROOT)), "sha256": artifact_shas["workbook"]},
        },
        "point_in_time": True,
        "official_checksums_verified": len(failures) == 0,
        "raw_archive_retained": True,
        "validation_opened": False,
        "oos_opened": False,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the preregistered Genetic_Primus V5 derivatives source")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--keep-raw", action="store_true")
    args = parser.parse_args()
    result = build(args.contract.resolve(), args.output_dir.resolve(), workers=args.workers, keep_raw=args.keep_raw)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
