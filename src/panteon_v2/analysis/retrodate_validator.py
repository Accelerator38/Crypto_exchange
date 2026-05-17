"""Validation helpers for Retrodate OHLCV CSV files."""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


_RETRODATE_NAME_RE = re.compile(
    r"^crypto_(?P<timeframe>[^_]+)_(?P<year>\d{4})_all_symbols\.csv$"
)
_REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume", "symbol", "datetime")


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
    files = [
        validate_retrodate_file(csv_path)
        for csv_path in sorted(root.glob("crypto_*_*_all_symbols.csv"))
    ]
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
