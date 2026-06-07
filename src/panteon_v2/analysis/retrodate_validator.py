"""Validation helpers for Retrodate OHLCV CSV files."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


_RETRODATE_NAME_RE = re.compile(
    r"^crypto_(?P<timeframe>[^_]+)_(?P<year>\d{4})_all_symbols\.csv$"
)
_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume", "symbol", "datetime")
_CACHE_FILENAME = ".retrodate_validation_cache.json"
_CACHE_VERSION = 1


class RetrodateValidationError(ValueError):
    """Raised when Retrodate validation must block a run."""


@dataclass(frozen=True)
class RetrodateValidationIssue:
    """One validation issue found in a Retrodate file or directory."""

    code: str
    message: str


@dataclass(frozen=True)
class RetrodateFileReport:
    """Validation result for one Retrodate CSV file."""

    path: Path
    expected_year: Optional[int]
    timeframe: str = ""
    rows: int = 0
    symbols: int = 0
    first_timestamp: Optional[int] = None
    last_timestamp: Optional[int] = None
    first_datetime: str = ""
    last_datetime: str = ""
    found_years: tuple[int, ...] = ()
    issues: list[RetrodateValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True)
class RetrodateDirReport:
    """Validation result for a Retrodate directory."""

    path: Path
    files: list[RetrodateFileReport]
    issues: list[RetrodateValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues and all(item.is_valid for item in self.files)

    @property
    def invalid_files(self) -> list[RetrodateFileReport]:
        return [item for item in self.files if not item.is_valid]


def validate_retrodate_file(path: str | Path) -> RetrodateFileReport:
    """Validate one `crypto_<tf>_<year>_all_symbols.csv` file."""
    csv_path = Path(path)
    expected_year, timeframe, issues = _parse_expected_year(csv_path)
    return _scan_retrodate_file(
        csv_path,
        expected_year=expected_year,
        timeframe=timeframe,
        issues=issues,
    )


def _scan_retrodate_file(
    csv_path: Path,
    *,
    expected_year: Optional[int],
    timeframe: str,
    issues: list[RetrodateValidationIssue],
) -> RetrodateFileReport:
    rows = 0
    symbols: set[str] = set()
    first_ts: Optional[int] = None
    last_ts: Optional[int] = None
    first_dt = ""
    last_dt = ""
    years: set[int] = set()
    bad_timestamps = 0

    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [column for column in _REQUIRED_COLUMNS if column not in (reader.fieldnames or [])]
            if missing:
                issues.append(
                    RetrodateValidationIssue(
                        "missing_columns",
                        f"missing required columns: {', '.join(missing)}",
                    )
                )
                return RetrodateFileReport(
                    path=csv_path,
                    expected_year=expected_year,
                    timeframe=timeframe,
                    issues=issues,
                )

            for row in reader:
                rows += 1
                symbol = str(row.get("symbol") or "").strip()
                if symbol:
                    symbols.add(symbol)
                try:
                    timestamp = int(str(row.get("timestamp") or "").strip())
                except ValueError:
                    bad_timestamps += 1
                    continue
                dt_text = str(row.get("datetime") or "")
                year = datetime.fromtimestamp(timestamp / 1000.0, timezone.utc).year
                years.add(year)
                if first_ts is None or timestamp < first_ts:
                    first_ts = timestamp
                    first_dt = dt_text
                if last_ts is None or timestamp > last_ts:
                    last_ts = timestamp
                    last_dt = dt_text
    except OSError as exc:
        issues.append(RetrodateValidationIssue("read_error", str(exc)))
        return RetrodateFileReport(
            path=csv_path,
            expected_year=expected_year,
            timeframe=timeframe,
            issues=issues,
        )

    if rows == 0:
        issues.append(RetrodateValidationIssue("empty_file", "file has no data rows"))
    if bad_timestamps:
        issues.append(
            RetrodateValidationIssue(
                "bad_timestamp",
                f"{bad_timestamps} rows have invalid timestamp values",
            )
        )
    if expected_year is not None and years and any(year != expected_year for year in years):
        found = ", ".join(str(year) for year in sorted(years))
        issues.append(
            RetrodateValidationIssue(
                "year_mismatch",
                f"{csv_path.name}: expected {expected_year} from filename, found timestamp years {found}",
            )
        )

    return RetrodateFileReport(
        path=csv_path,
        expected_year=expected_year,
        timeframe=timeframe,
        rows=rows,
        symbols=len(symbols),
        first_timestamp=first_ts,
        last_timestamp=last_ts,
        first_datetime=first_dt,
        last_datetime=last_dt,
        found_years=tuple(sorted(years)),
        issues=issues,
    )


def validate_retrodate_dir(path: str | Path) -> RetrodateDirReport:
    """Validate all Retrodate yearly OHLCV CSV files in a directory."""
    root = Path(path)
    issues: list[RetrodateValidationIssue] = []
    cache = _load_cache(root)
    cache_changed = False
    files: list[RetrodateFileReport] = []
    for csv_path in sorted(root.glob("crypto_*_*_all_symbols.csv")):
        cached = _report_from_cache(csv_path, cache)
        if cached is not None:
            files.append(cached)
            continue
        report = validate_retrodate_file(csv_path)
        files.append(report)
        cache_changed = _store_report_in_cache(csv_path, report, cache) or cache_changed
    if cache_changed:
        _write_cache(root, cache)
    if not root.exists():
        issues.append(RetrodateValidationIssue("missing_directory", f"directory not found: {root}"))
    elif not files:
        issues.append(RetrodateValidationIssue("no_retrodate_files", f"no Retrodate CSV files found in {root}"))
    return RetrodateDirReport(path=root, files=files, issues=issues)


def ensure_retrodate_dir_valid(path: str | Path) -> RetrodateDirReport:
    """Validate a Retrodate directory and raise when it is not safe to use."""
    report = validate_retrodate_dir(path)
    if report.is_valid:
        return report
    raise RetrodateValidationError(_format_dir_report_errors(report))


def _parse_expected_year(path: Path) -> tuple[Optional[int], str, list[RetrodateValidationIssue]]:
    match = _RETRODATE_NAME_RE.match(path.name)
    if not match:
        return (
            None,
            "",
            [
                RetrodateValidationIssue(
                    "filename_pattern",
                    f"unexpected Retrodate filename: {path.name}",
                )
            ],
        )
    return int(match.group("year")), match.group("timeframe"), []


def _format_dir_report_errors(report: RetrodateDirReport) -> str:
    lines = [f"Retrodate validation failed for {report.path}"]
    for issue in report.issues:
        lines.append(f"directory:{issue.code}: {issue.message}")
    for file_report in report.invalid_files:
        for issue in file_report.issues:
            lines.append(f"{file_report.path.name}:{issue.code}: {issue.message}")
    return "\n".join(lines)


def _cache_path(root: Path) -> Path:
    return root / _CACHE_FILENAME


def _load_cache(root: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_cache_path(root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": _CACHE_VERSION, "files": {}}
    if not isinstance(payload, dict) or payload.get("version") != _CACHE_VERSION:
        return {"version": _CACHE_VERSION, "files": {}}
    files = payload.get("files")
    if not isinstance(files, dict):
        payload["files"] = {}
    return payload


def _write_cache(root: Path, cache: dict[str, Any]) -> None:
    if not root.exists():
        return
    try:
        _cache_path(root).write_text(
            json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        return


def _report_from_cache(
    csv_path: Path,
    cache: dict[str, Any],
) -> Optional[RetrodateFileReport]:
    try:
        stat = csv_path.stat()
    except OSError:
        return None
    files = cache.get("files")
    if not isinstance(files, dict):
        return None
    entry = files.get(_cache_key(csv_path))
    if not isinstance(entry, dict):
        return None
    if int(entry.get("size", -1) or -1) != int(stat.st_size):
        return None
    if int(entry.get("mtime_ns", -1) or -1) != int(stat.st_mtime_ns):
        return None
    report = entry.get("report")
    if not isinstance(report, dict):
        return None
    return _file_report_from_payload(csv_path, report)


def _store_report_in_cache(
    csv_path: Path,
    report: RetrodateFileReport,
    cache: dict[str, Any],
) -> bool:
    try:
        stat = csv_path.stat()
    except OSError:
        return False
    files = cache.setdefault("files", {})
    if not isinstance(files, dict):
        cache["files"] = files = {}
    files[_cache_key(csv_path)] = {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "report": _file_report_payload(report),
    }
    return True


def _cache_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _file_report_payload(report: RetrodateFileReport) -> dict[str, Any]:
    return {
        "expected_year": report.expected_year,
        "timeframe": report.timeframe,
        "rows": int(report.rows),
        "symbols": int(report.symbols),
        "first_timestamp": report.first_timestamp,
        "last_timestamp": report.last_timestamp,
        "first_datetime": report.first_datetime,
        "last_datetime": report.last_datetime,
        "found_years": list(report.found_years),
        "issues": [
            {"code": issue.code, "message": issue.message}
            for issue in report.issues
        ],
    }


def _file_report_from_payload(
    csv_path: Path,
    payload: dict[str, Any],
) -> RetrodateFileReport:
    return RetrodateFileReport(
        path=csv_path,
        expected_year=(
            int(payload["expected_year"])
            if payload.get("expected_year") is not None
            else None
        ),
        timeframe=str(payload.get("timeframe") or ""),
        rows=int(payload.get("rows") or 0),
        symbols=int(payload.get("symbols") or 0),
        first_timestamp=(
            int(payload["first_timestamp"])
            if payload.get("first_timestamp") is not None
            else None
        ),
        last_timestamp=(
            int(payload["last_timestamp"])
            if payload.get("last_timestamp") is not None
            else None
        ),
        first_datetime=str(payload.get("first_datetime") or ""),
        last_datetime=str(payload.get("last_datetime") or ""),
        found_years=tuple(int(year) for year in payload.get("found_years") or ()),
        issues=[
            RetrodateValidationIssue(
                str(item.get("code") or ""),
                str(item.get("message") or ""),
            )
            for item in payload.get("issues") or ()
            if isinstance(item, dict)
        ],
    )
