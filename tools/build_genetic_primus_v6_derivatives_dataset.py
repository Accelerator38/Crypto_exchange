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
from exia.genetic_primus.contracts_v6 import load_v6_contract  # noqa: E402
from exia.genetic_primus.data_source_v5 import pack_raw_directory, parse_checksum, sha256_bytes  # noqa: E402
from exia.genetic_primus.data_source_v6 import build_flow_frame, read_kline_zip  # noqa: E402


DEFAULT_CONTRACT = ROOT / "configs/genetic_primus_v6_contract.json"
DEFAULT_OUTPUT = ROOT / "Retrodate/genetic_primus_v6_oi_kline_flow_v1_20220101_20241231"


def _fetch(url: str, attempts: int = 5) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(Request(url, headers={"User-Agent": "Exia-V6-Research/1.0"}), timeout=45) as response:
                return response.read()
        except (HTTPError, URLError, TimeoutError) as exc:
            error = exc
            if isinstance(exc, HTTPError) and exc.code == 404:
                break
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"failed to fetch {url}: {error}")


def _items(contract) -> list[dict[str, str]]:
    months = pd.period_range(contract.flow_source["start_month"], contract.flow_source["end_month"], freq="M")
    items = []
    for target_symbol, exchange_symbol in contract.flow_source["symbols"].items():
        for month in months:
            year_month = str(month)
            filename = f"{exchange_symbol}-4h-{year_month}.zip"
            items.append(
                {
                    "symbol": target_symbol,
                    "exchange_symbol": exchange_symbol,
                    "year_month": year_month,
                    "filename": filename,
                    "url": contract.flow_source["url_template"].format(
                        exchange_symbol=exchange_symbol, year_month=year_month
                    ),
                }
            )
    return items


def _download(item: dict[str, str], raw_dir: Path) -> dict[str, Any]:
    target = raw_dir / item["exchange_symbol"] / item["filename"]
    target.parent.mkdir(parents=True, exist_ok=True)
    official = parse_checksum(_fetch(item["url"] + ".CHECKSUM"), item["filename"])
    if not target.is_file() or sha256_file(target) != official:
        payload = _fetch(item["url"])
        if sha256_bytes(payload) != official:
            raise ValueError(f"V6 official checksum mismatch: {item['filename']}")
        temporary = target.with_suffix(".zip.tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
    return {
        **item,
        "local_path": target.relative_to(raw_dir).as_posix(),
        "bytes": target.stat().st_size,
        "sha256": official,
        "official_sha256": official,
        "downloaded": True,
        "error": None,
    }


def _workbook(path: Path, frames: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, frame in frames.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.book[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions


def build(contract_path: Path, output_dir: Path, workers: int) -> dict[str, Any]:
    contract = load_v6_contract(contract_path, root=ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("contract_sha256") == contract.sha256:
            return existing
        raise ValueError("stale V6 dataset manifest")
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
                reusable[(str(row["exchange_symbol"]), str(row["year_month"]))] = row
    rows: list[dict[str, Any]] = list(reusable.values())
    failures: list[dict[str, Any]] = []
    pending = [item for item in items if (item["exchange_symbol"], item["year_month"]) not in reusable]
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(_download, item, raw_dir): item for item in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            item = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({**item, "downloaded": False, "error": str(exc)})
            if index % 32 == 0 or index == len(pending):
                print(json.dumps({"downloaded": len(rows), "failed": len(failures), "total": len(items)}), flush=True)
    catalog = pd.DataFrame([*rows, *failures]).sort_values(["exchange_symbol", "year_month"], kind="stable")
    catalog.to_parquet(catalog_path, index=False)
    flow_frames = []
    raw_rows = 0
    repaired_header_paths: set[str] = set()
    for target_symbol, exchange_symbol in contract.flow_source["symbols"].items():
        symbol_frames = []
        subset = catalog.loc[(catalog["exchange_symbol"] == exchange_symbol) & (catalog["downloaded"] == True)]  # noqa: E712
        for relative in subset["local_path"].dropna().astype(str):
            frame = read_kline_zip(raw_dir / relative)
            if frame.attrs.get("source_header_missing") is True:
                repaired_header_paths.add(relative)
            raw_rows += len(frame)
            symbol_frames.append(frame)
        if symbol_frames:
            raw = pd.concat(symbol_frames, ignore_index=True).sort_values("open_time", kind="stable")
            if raw["open_time"].duplicated().any():
                raise ValueError(f"duplicate V6 flow bars: {exchange_symbol}")
            flow_frames.append(build_flow_frame(raw, source_symbol=exchange_symbol, target_symbol=target_symbol))
    catalog["missing_header_repaired"] = catalog["local_path"].astype(str).isin(repaired_header_paths)
    catalog.to_parquet(catalog_path, index=False)
    flow = pd.concat(flow_frames, ignore_index=True).sort_values(["timestamp", "symbol"], kind="stable")
    flow_path = output_dir / "kline_flow_4h.parquet"
    flow.to_parquet(flow_path, index=False)
    base_manifest = json.loads((ROOT / contract.payload["base_derivatives_source"]["manifest"]).read_text(encoding="utf-8"))
    base_path = ROOT / base_manifest["artifacts"]["dataset"]["path"]
    if sha256_file(base_path) != contract.payload["base_derivatives_source"]["dataset_artifact_sha256"]:
        raise ValueError("V6 base derivatives artifact SHA mismatch")
    base = pd.read_parquet(base_path)
    combined = base.drop(columns=["taker_imbalance_mean", "taker_imbalance_std"]).merge(
        flow.drop(columns=["available_at_ms", "source_symbol"]),
        on=["timestamp", "symbol"],
        how="left",
        validate="one_to_one",
    )
    combined = combined.sort_values(["timestamp", "symbol"], kind="stable").reset_index(drop=True)
    combined_path = output_dir / "derivatives_4h.parquet"
    combined.to_parquet(combined_path, index=False)
    coverage_rows = []
    for symbol, group in combined.groupby("symbol", sort=True, observed=True):
        coverage_rows.append(
            {
                "symbol": str(symbol),
                "rows": int(len(group)),
                "joined_flow_coverage": float(group["kline_taker_imbalance_4h"].notna().mean()),
                **{
                    f"{feature}_coverage": float(group[feature].notna().mean())
                    for feature in contract.payload["required_core_features"]
                },
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    archive_path = output_dir / "raw_kline_flow.zip"
    pack_raw_directory(raw_dir, archive_path)
    shutil.rmtree(raw_dir)
    workbook_path = output_dir / "dataset_catalog.xlsx"
    _workbook(workbook_path, {"Raw Files": catalog, "Coverage": coverage})
    artifacts = {
        "combined_dataset": combined_path,
        "flow_dataset": flow_path,
        "raw_catalog": catalog_path,
        "raw_archive": archive_path,
        "workbook": workbook_path,
    }
    artifact_rows = {
        key: {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)} for key, path in artifacts.items()
    }
    dataset_sha = hashlib.sha256(
        (contract.sha256 + "".join(artifact_rows[key]["sha256"] for key in sorted(artifact_rows))).encode()
    ).hexdigest()
    manifest = {
        "schema_version": "exia.genetic_primus_derivatives_dataset.v6",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "V6_DERIVATIVES_DATASET_BUILT" if not failures else "V6_DERIVATIVES_DATASET_INCOMPLETE",
        "contract_sha256": contract.sha256,
        "source_id": contract.flow_source["source_id"],
        "source_family": contract.flow_source["source_family"],
        "base_dataset_sha256": contract.payload["base_derivatives_source"]["dataset_sha256"],
        "expected_raw_files": len(items),
        "downloaded_raw_files": len(rows),
        "failed_raw_files": len(failures),
        "missing_header_repaired_files": len(repaired_header_paths),
        "raw_file_coverage": len(rows) / len(items),
        "raw_rows": int(raw_rows),
        "flow_rows": int(len(flow)),
        "combined_rows": int(len(combined)),
        "coverage": coverage.to_dict(orient="records"),
        "failures": failures,
        "dataset_sha256": dataset_sha,
        "artifacts": artifact_rows,
        "point_in_time": True,
        "official_checksums_verified": not failures,
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
    parser = argparse.ArgumentParser(description="Build Genetic_Primus V6 OI plus four-hour kline flow data")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=24)
    args = parser.parse_args()
    print(json.dumps(build(args.contract.resolve(), args.output_dir.resolve(), args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
