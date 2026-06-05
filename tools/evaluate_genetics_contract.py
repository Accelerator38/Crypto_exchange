from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
GENETICS_DIR = ROOT / "Genetics_DL_Agents"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(GENETICS_DIR) not in sys.path:
    sys.path.insert(0, str(GENETICS_DIR))

import crypto_genetics as cg  # noqa: E402
from panteon_v2.analysis.genetics_validation import (  # noqa: E402
    build_rolling_year_folds,
    robust_period_score,
    split_precomp_by_periods,
)
from panteon_v2.analysis.retrodate_validator import (  # noqa: E402
    RetrodateDirReport,
    RetrodateFileReport,
    RetrodateValidationIssue,
    RetrodateValidationError,
    validate_retrodate_dir,
)


_RETRODATE_FILE_RE = re.compile(
    r"^crypto_(?P<timeframe>[^_]+)_(?P<year>\d{4})_all_symbols\.csv$"
)
_RETRODATE_REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "symbol",
    "datetime",
)


@dataclass(frozen=True)
class RequestedRetrodateFileSelection:
    report: RetrodateDirReport
    requested_years: tuple[int, ...]
    valid_reports: tuple[RetrodateFileReport, ...]
    excluded_files: tuple[RetrodateFileReport, ...]
    missing_years: tuple[int, ...]

    @property
    def valid_files(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.valid_reports)

    @property
    def executed_years(self) -> tuple[int, ...]:
        years = [item.expected_year for item in self.valid_reports if item.expected_year is not None]
        return tuple(sorted(set(int(year) for year in years)))


def _torch_device():
    try:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        return None


def _parse_config_year(value: Any, fallback: int) -> int:
    match = re.match(r"(\d{4})", str(value or ""))
    if match:
        return int(match.group(1))
    return int(fallback)


def _requested_retrodate_years(start_date: Any, end_date: Any) -> tuple[int, ...]:
    start_year = _parse_config_year(start_date, 2022)
    end_year = _parse_config_year(end_date, 2025)
    if start_year > end_year:
        raise RetrodateValidationError(
            f"Invalid genetics evaluation date range: start year {start_year} > end year {end_year}"
        )
    return tuple(range(start_year, end_year + 1))


def _resolve_eval_data_dir(data_dir_override: str | None = None) -> Path:
    if data_dir_override:
        path = Path(data_dir_override)
        if not path.is_absolute():
            path = ROOT / path
        return path
    return Path(cg._resolve_data_dir())


def _format_retrodate_directory_issues(report: RetrodateDirReport) -> str:
    lines = [f"Retrodate directory is not usable for genetics evaluation: {report.path}"]
    for issue in report.issues:
        lines.append(f"directory:{issue.code}: {issue.message}")
    return "\n".join(lines)


def _format_requested_retrodate_errors(reports: tuple[RetrodateFileReport, ...]) -> str:
    lines = ["Requested Retrodate files failed genetics evaluation validation"]
    for file_report in reports:
        for issue in file_report.issues:
            lines.append(f"{file_report.path.name}:{issue.code}: {issue.message}")
    return "\n".join(lines)


def _expected_retrodate_file_report(
    path: Path,
    *,
    expected_year: int,
    timeframe: str,
    issues: list[RetrodateValidationIssue],
) -> RetrodateFileReport:
    return RetrodateFileReport(
        path=path,
        expected_year=expected_year,
        timeframe=timeframe,
        issues=issues,
    )


def _read_last_nonempty_line(path: Path, *, block_size: int = 65536) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        buffer = b""
        while position > 0:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            buffer = handle.read(read_size) + buffer
            lines = [line for line in buffer.splitlines() if line.strip()]
            if len(lines) >= 2 or position == 0:
                return lines[-1].decode("utf-8", errors="replace") if lines else ""
    return ""


def _validate_requested_retrodate_file_sample(
    path: Path,
    *,
    expected_year: int,
    timeframe: str,
    sample_rows: int = 16,
) -> RetrodateFileReport:
    issues: list[RetrodateValidationIssue] = []
    match = _RETRODATE_FILE_RE.match(path.name)
    if not match:
        issues.append(
            RetrodateValidationIssue("filename_pattern", f"unexpected Retrodate filename: {path.name}")
        )
        return _expected_retrodate_file_report(
            path,
            expected_year=expected_year,
            timeframe=timeframe,
            issues=issues,
        )
    if int(match.group("year")) != expected_year or match.group("timeframe") != timeframe:
        issues.append(
            RetrodateValidationIssue(
                "filename_pattern",
                f"expected crypto_{timeframe}_{expected_year}_all_symbols.csv, got {path.name}",
            )
        )
        return _expected_retrodate_file_report(
            path,
            expected_year=expected_year,
            timeframe=timeframe,
            issues=issues,
        )

    rows = 0
    symbols: set[str] = set()
    years: set[int] = set()
    first_ts: Optional[int] = None
    last_ts: Optional[int] = None
    first_dt = ""
    last_dt = ""
    bad_timestamps = 0

    def observe(row: dict[str, str]) -> None:
        nonlocal rows, first_ts, last_ts, first_dt, last_dt, bad_timestamps
        rows += 1
        symbol = str(row.get("symbol") or "").strip()
        if symbol:
            symbols.add(symbol)
        try:
            timestamp = int(str(row.get("timestamp") or "").strip())
        except ValueError:
            bad_timestamps += 1
            return
        dt_text = str(row.get("datetime") or "")
        years.add(datetime.fromtimestamp(timestamp / 1000.0, timezone.utc).year)
        if first_ts is None or timestamp < first_ts:
            first_ts = timestamp
            first_dt = dt_text
        if last_ts is None or timestamp > last_ts:
            last_ts = timestamp
            last_dt = dt_text

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            header_line = handle.readline()
            if not header_line:
                issues.append(RetrodateValidationIssue("empty_file", "file has no header row"))
                return _expected_retrodate_file_report(
                    path,
                    expected_year=expected_year,
                    timeframe=timeframe,
                    issues=issues,
                )
            reader = csv.DictReader([header_line])
            missing = [
                column
                for column in _RETRODATE_REQUIRED_COLUMNS
                if column not in (reader.fieldnames or [])
            ]
            if missing:
                issues.append(
                    RetrodateValidationIssue(
                        "missing_columns",
                        f"missing required columns: {', '.join(missing)}",
                    )
                )
                return _expected_retrodate_file_report(
                    path,
                    expected_year=expected_year,
                    timeframe=timeframe,
                    issues=issues,
                )

            reader = csv.DictReader(handle, fieldnames=reader.fieldnames)
            for idx, row in enumerate(reader):
                if idx >= sample_rows:
                    break
                observe(row)

        last_line = _read_last_nonempty_line(path)
        if last_line and last_line.strip() != header_line.strip():
            tail_reader = csv.DictReader([header_line, last_line])
            for row in tail_reader:
                observe(row)
                break
    except OSError as exc:
        issues.append(RetrodateValidationIssue("read_error", str(exc)))
        return _expected_retrodate_file_report(
            path,
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
                f"{bad_timestamps} sampled rows have invalid timestamp values",
            )
        )
    if years and any(year != expected_year for year in years):
        found = ", ".join(str(year) for year in sorted(years))
        issues.append(
            RetrodateValidationIssue(
                "year_mismatch",
                f"{path.name}: expected {expected_year} from filename, found timestamp years {found}",
            )
        )

    return RetrodateFileReport(
        path=path,
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


def _validate_requested_retrodate_files(
    data_dir: str | Path,
    *,
    timeframe: str,
    start_date: Any,
    end_date: Any,
) -> RequestedRetrodateFileSelection:
    root = Path(data_dir)
    directory_issues: list[RetrodateValidationIssue] = []
    if not root.exists():
        directory_issues.append(
            RetrodateValidationIssue("missing_directory", f"directory not found: {root}")
        )
    elif not root.is_dir():
        directory_issues.append(
            RetrodateValidationIssue("not_directory", f"path is not a directory: {root}")
        )
    if directory_issues:
        report = RetrodateDirReport(path=root, files=[], issues=directory_issues)
        raise RetrodateValidationError(_format_retrodate_directory_issues(report))

    requested_years = _requested_retrodate_years(start_date, end_date)
    selected_reports: list[RetrodateFileReport] = []
    missing_years: list[int] = []
    for year in requested_years:
        path = root / f"crypto_{timeframe}_{year}_all_symbols.csv"
        if not path.exists():
            missing_years.append(year)
            continue
        selected_reports.append(
            _validate_requested_retrodate_file_sample(
                path,
                expected_year=year,
                timeframe=timeframe,
            )
        )

    report = RetrodateDirReport(path=root, files=selected_reports)
    selected = tuple(selected_reports)
    excluded = tuple(item for item in selected if not item.is_valid)
    if excluded:
        raise RetrodateValidationError(_format_requested_retrodate_errors(excluded))

    valid_reports = tuple(
        sorted(
            (item for item in selected if item.is_valid),
            key=lambda item: (item.expected_year or 0, item.path.name),
        )
    )
    missing_years_tuple = tuple(missing_years)
    if missing_years_tuple:
        raise RetrodateValidationError(
            "Missing requested Retrodate files for genetics evaluation: "
            + ", ".join(f"crypto_{timeframe}_{year}_all_symbols.csv" for year in missing_years_tuple)
        )
    if not valid_reports:
        raise RetrodateValidationError(
            "No valid Retrodate files selected for requested genetics evaluation years: "
            + ", ".join(str(year) for year in requested_years)
        )

    return RequestedRetrodateFileSelection(
        report=report,
        requested_years=requested_years,
        valid_reports=valid_reports,
        excluded_files=excluded,
        missing_years=missing_years_tuple,
    )


def _period_stats(period_rets: List[float]) -> Dict[str, Any]:
    arr = np.asarray(period_rets, dtype=np.float64)
    if arr.size == 0:
        return {}
    return {
        "n_periods": int(arr.size),
        "mean_ret": float(arr.mean()),
        "median_ret": float(np.median(arr)),
        "std_ret": float(arr.std()),
        "min_ret": float(arr.min()),
        "max_ret": float(arr.max()),
        "positive_period_pct": float((arr > 0.0).mean() * 100.0),
        "best_period_idx": int(arr.argmax()),
        "worst_period_idx": int(arr.argmin()),
    }


def _evaluate_population(population: np.ndarray, precomp: list) -> tuple[np.ndarray, List[List[float]]]:
    device = _torch_device()
    if device is not None:
        return cg.GPUEvaluator(device).evaluate(population, precomp)
    ev = cg.CPUEvaluator(precomp, n_workers=1)
    try:
        return ev.evaluate(population)
    finally:
        shutdown = getattr(ev, "shutdown", None) or getattr(ev, "close", None)
        if shutdown is not None:
            shutdown()


def _infer_position_state_features_enabled(genome_path: Path) -> bool:
    """Infer whether a genome was trained with position-state feature injection."""
    path = Path(genome_path)
    if not path.exists():
        return False
    metadata_candidates = [
        path.with_name(f"{path.stem}_meta.json"),
        path.parent / "best_genome_meta.json",
    ]
    for meta_path in metadata_candidates:
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "position_state_features_enabled" in payload:
            return bool(payload["position_state_features_enabled"])
    return False


def _contract_regime_label(entry: tuple) -> str:
    raw = str(entry[6]) if len(entry) > 6 else "neutral"
    if raw in {"strong_crash", "crash"}:
        return "crash"

    avg_ret_pct = None
    if len(entry) > 7:
        try:
            avg_ret_pct = float(entry[7])
        except (TypeError, ValueError):
            avg_ret_pct = None
    if avg_ret_pct is not None and avg_ret_pct <= -20.0:
        return "crash"

    return cg.map_regime_3(raw)


def _open_output_bias_indices() -> np.ndarray:
    b4_start = int(cg.GENOME_SIZE - cg.N_ACTIONS)
    open_action_codes = [
        int(code)
        for code in getattr(cg, "_OPEN_ACTION_CODES", ())
        if 0 <= int(code) < int(cg.N_ACTIONS)
    ]
    return b4_start + np.asarray(open_action_codes, dtype=np.int64)


def _apply_open_output_bias(genome: np.ndarray, open_bias: float) -> np.ndarray:
    adjusted = np.asarray(genome, dtype=np.float32).copy()
    bias_indices = _open_output_bias_indices()
    if bias_indices.size:
        adjusted[bias_indices] -= np.float32(open_bias)
    return adjusted


def _parse_regime_open_bias_specs(raw_specs: Optional[List[str]]) -> Dict[str, float]:
    if not raw_specs:
        return {}
    parsed: Dict[str, float] = {}
    allowed = {"crash", "bearish", "neutral", "bullish", "default"}
    for raw_spec in raw_specs:
        key, separator, value = str(raw_spec).partition("=")
        regime = key.strip().lower()
        if not separator or not regime:
            raise ValueError(f"Invalid --regime-open-bias value: {raw_spec!r}; expected REGIME=BIAS")
        if regime not in allowed:
            raise ValueError(
                f"Unsupported regime for --regime-open-bias: {regime!r}; "
                f"expected one of {', '.join(sorted(allowed))}"
            )
        try:
            parsed[regime] = float(value)
        except ValueError as exc:
            raise ValueError(f"Invalid bias for regime {regime!r}: {value!r}") from exc
    return parsed


def _resolve_regime_open_bias(regime: str, regime_open_bias: Mapping[str, float]) -> float:
    normalized = str(regime).strip().lower()
    if normalized in regime_open_bias:
        return float(regime_open_bias[normalized])
    if normalized == "crash" and "bearish" in regime_open_bias:
        return float(regime_open_bias["bearish"])
    if "default" in regime_open_bias:
        return float(regime_open_bias["default"])
    return 0.0


def _normalize_symbol_base(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if "/" in text:
        text = text.split("/", 1)[0]
    elif text.endswith("USDT"):
        text = text[:-4]
    return text.strip()


def _parse_symbol_filter(raw_specs: Optional[List[str]]) -> tuple[str, ...]:
    if not raw_specs:
        return ()
    symbols: list[str] = []
    seen: set[str] = set()
    for raw_spec in raw_specs:
        for raw_item in str(raw_spec).replace(";", ",").split(","):
            symbol = _normalize_symbol_base(raw_item)
            if not symbol or symbol in seen:
                continue
            seen.add(symbol)
            symbols.append(symbol)
    return tuple(symbols)


def _filter_array_symbol_axis(value: Any, indices: np.ndarray, symbol_count: int) -> Any:
    arr = np.asarray(value)
    if arr.ndim >= 2 and arr.shape[1] == symbol_count:
        return arr[:, indices, ...]
    if arr.ndim >= 1 and arr.shape[0] == symbol_count:
        return arr[indices, ...]
    return value


def _filter_precomp_symbols(precomp: list, allowed_symbols: tuple[str, ...]) -> tuple[list, Dict[str, Any]]:
    allowed = {_normalize_symbol_base(symbol) for symbol in allowed_symbols if _normalize_symbol_base(symbol)}
    summary: Dict[str, Any] = {
        "requested_symbols": sorted(allowed),
        "periods_before": int(len(precomp)),
        "periods_after": 0,
        "periods_dropped": 0,
        "periods": [],
    }
    if not allowed:
        summary["periods_after"] = int(len(precomp))
        return precomp, summary

    filtered_precomp: list = []
    for entry in precomp:
        if len(entry) < 3:
            continue
        feat, prices, syms = entry[:3]
        symbol_names = list(syms) if syms is not None else []
        indices = [
            idx
            for idx, symbol in enumerate(symbol_names)
            if _normalize_symbol_base(symbol) in allowed
        ]
        period = str(entry[4]) if len(entry) > 4 else ""
        if not indices:
            summary["periods"].append({
                "period": period,
                "symbols_before": int(len(symbol_names)),
                "symbols_after": 0,
                "kept_symbols": [],
                "dropped": True,
            })
            continue

        index_arr = np.asarray(indices, dtype=np.int64)
        filtered_symbols = [symbol_names[idx] for idx in indices]
        filtered_entry = (
            _filter_array_symbol_axis(feat, index_arr, len(symbol_names)),
            _filter_array_symbol_axis(prices, index_arr, len(symbol_names)),
            filtered_symbols,
            *entry[3:],
        )
        filtered_precomp.append(filtered_entry)
        summary["periods"].append({
            "period": period,
            "symbols_before": int(len(symbol_names)),
            "symbols_after": int(len(filtered_symbols)),
            "kept_symbols": [str(symbol) for symbol in filtered_symbols],
            "dropped": False,
        })

    summary["periods_after"] = int(len(filtered_precomp))
    summary["periods_dropped"] = int(len(precomp) - len(filtered_precomp))
    return filtered_precomp, summary


def _position_exposure_metrics(
    actions_arr: np.ndarray,
    *,
    max_positions: int,
    symbols: Optional[List[str]] = None,
    top_n_symbols: int = 5,
) -> Dict[str, Any]:
    actions = np.asarray(actions_arr, dtype=np.int32)
    if actions.ndim != 3:
        raise ValueError("actions_arr must have shape (G, T, NC)")

    G, T, NC = actions.shape
    if G == 0 or T == 0 or NC == 0:
        return {
            "mean_long_slot_rate": 0.0,
            "mean_short_slot_rate": 0.0,
            "mean_position_slot_rate": 0.0,
            "mean_net_direction_bias": 0.0,
            "mean_capacity_usage": 0.0,
            "max_open_positions": 0,
            "capacity_bar_rate": 0.0,
            "top_position_symbols": [],
        }

    max_pos = max(0, int(max_positions))
    spot = np.zeros((G, NC), dtype=bool)
    fut_dir = np.zeros((G, NC), dtype=np.int8)
    symbol_long_slots = np.zeros(NC, dtype=np.float64)
    symbol_short_slots = np.zeros(NC, dtype=np.float64)
    symbol_position_slots = np.zeros(NC, dtype=np.float64)
    long_slots = 0.0
    short_slots = 0.0
    position_slots = 0.0
    open_position_sum = 0.0
    max_open_seen = 0
    capacity_bars = 0.0

    for t in range(T):
        for g in range(G):
            occupied_before = spot[g] | (fut_dir[g] != 0)
            n_pos = int(occupied_before.sum())
            for c in range(NC):
                action = int(actions[g, t, c])
                occupied = bool(spot[g, c] or fut_dir[g, c] != 0)
                if action in (1, 2, 9):
                    if not spot[g, c]:
                        if not occupied and max_pos > 0 and n_pos >= max_pos:
                            continue
                        if not occupied:
                            n_pos += 1
                        spot[g, c] = True
                elif action == 3 and spot[g, c]:
                    spot[g, c] = False
                    if fut_dir[g, c] == 0:
                        n_pos -= 1
                elif action in (4, 5):
                    if fut_dir[g, c] < 0:
                        continue
                    if fut_dir[g, c] == 0:
                        if not occupied and max_pos > 0 and n_pos >= max_pos:
                            continue
                        if not occupied:
                            n_pos += 1
                    fut_dir[g, c] = 1
                elif action in (6, 7):
                    if fut_dir[g, c] > 0:
                        continue
                    if fut_dir[g, c] == 0:
                        if not occupied and max_pos > 0 and n_pos >= max_pos:
                            continue
                        if not occupied:
                            n_pos += 1
                    fut_dir[g, c] = -1
                elif action == 8 and fut_dir[g, c] != 0:
                    fut_dir[g, c] = 0
                    if not spot[g, c]:
                        n_pos -= 1

            long_now = spot[g] | (fut_dir[g] > 0)
            short_now = fut_dir[g] < 0
            occupied_now = spot[g] | (fut_dir[g] != 0)
            open_now = int(occupied_now.sum())
            symbol_long_slots += long_now.astype(np.float64)
            symbol_short_slots += short_now.astype(np.float64)
            symbol_position_slots += occupied_now.astype(np.float64)
            long_slots += float(long_now.sum())
            short_slots += float(short_now.sum())
            position_slots += float(open_now)
            open_position_sum += float(open_now)
            max_open_seen = max(max_open_seen, open_now)
            if max_pos > 0 and open_now >= max_pos:
                capacity_bars += 1.0

    slot_denom = float(G * T * NC)
    bar_denom = float(G * T)
    directional_total = long_slots + short_slots
    symbol_names = list(symbols or [str(i) for i in range(NC)])
    if len(symbol_names) < NC:
        symbol_names.extend(str(i) for i in range(len(symbol_names), NC))
    per_symbol_denom = float(G * T)
    top_n = max(0, min(int(top_n_symbols), NC))
    order = np.argsort(-symbol_position_slots)[:top_n]
    top_position_symbols = [
        {
            "symbol": str(symbol_names[int(idx)]),
            "position_slot_rate": float(symbol_position_slots[int(idx)] / per_symbol_denom),
            "long_slot_rate": float(symbol_long_slots[int(idx)] / per_symbol_denom),
            "short_slot_rate": float(symbol_short_slots[int(idx)] / per_symbol_denom),
        }
        for idx in order
        if symbol_position_slots[int(idx)] > 0.0
    ]
    return {
        "mean_long_slot_rate": float(long_slots / slot_denom),
        "mean_short_slot_rate": float(short_slots / slot_denom),
        "mean_position_slot_rate": float(position_slots / slot_denom),
        "mean_net_direction_bias": float(
            (long_slots - short_slots) / directional_total
            if directional_total > 0.0
            else 0.0
        ),
        "mean_capacity_usage": float(
            open_position_sum / (bar_denom * max_pos)
            if max_pos > 0
            else 0.0
        ),
        "max_open_positions": int(max_open_seen),
        "capacity_bar_rate": float(capacity_bars / bar_denom),
        "top_position_symbols": top_position_symbols,
    }


def _contract_metrics_for_genome(
    *,
    genome: np.ndarray,
    precomp: list,
    execution_lag_bars: int,
) -> Dict[str, Any]:
    population = genome[None].astype(np.float32)
    period_rows: list[dict[str, Any]] = []
    turnover_rates: list[float] = []
    saturation_rates: list[float] = []
    invalid_open_logit_pressures: list[float] = []
    capacity_bar_rates: list[float] = []
    same_side_open_rates: list[float] = []

    for entry in precomp:
        feat, prices, syms, _month, period = entry[:5]
        regime = _contract_regime_label(entry)
        suppression_metrics = None
        if (
            cg.POSITION_STATE_FEATURES_ENABLED
            and (not cg.CURRENCY_SELECTION_ENABLED or cg.CURRENCY_LEARNING_ENABLED)
        ):
            actions, suppression_metrics = cg._batch_forward_position_aware_numpy(
                population,
                feat,
                prices,
                return_suppression_metrics=True,
            )
        else:
            actions = cg.GeneticTrainer._batch_forward_numpy(population, feat, prices=prices)
        if cg.CURRENCY_SELECTION_ENABLED:
            actions = cg._apply_currency_selection(
                actions.astype(np.int32),
                feat,
                regime,
                cg.TOP_CURRENCIES_N,
            ).astype(np.int8)
        metrics = cg._action_contract_metrics(actions.astype(np.int32))
        exec_actions = cg._apply_position_aware_action_policy(actions.astype(np.int32))
        exec_metrics = cg._action_contract_metrics(exec_actions.astype(np.int32))
        turnover_rate = float(metrics["turnover_rates"][0])
        saturation_rate = float(
            suppression_metrics["saturation_rates"][0]
            if suppression_metrics is not None
            else metrics["saturation_rates"][0]
        )
        invalid_open_logit_pressure = float(
            suppression_metrics["invalid_open_logit_pressures"][0]
            if suppression_metrics is not None
            else 0.0
        )
        raw_capacity_bar_rate = float(
            suppression_metrics["capacity_bar_rates"][0]
            if suppression_metrics is not None
            else metrics["capacity_bar_rates"][0]
        )
        same_side_open_rate = float(
            suppression_metrics["same_side_open_rates"][0]
            if suppression_metrics is not None
            else metrics["same_side_open_rates"][0]
        )
        effective_turnover_rate = float(exec_metrics["turnover_rates"][0])
        exposure_metrics = _position_exposure_metrics(
            exec_actions.astype(np.int32),
            max_positions=int(cg.TRAIN_MAX_POS),
            symbols=[str(sym) for sym in syms],
            top_n_symbols=len(syms),
        )
        turnover_rates.append(turnover_rate)
        saturation_rates.append(saturation_rate)
        invalid_open_logit_pressures.append(invalid_open_logit_pressure)
        capacity_bar_rates.append(raw_capacity_bar_rate)
        same_side_open_rates.append(same_side_open_rate)
        period_rows.append({
            "period": str(period),
            "regime": str(regime),
            "n_bars": int(feat.shape[0]),
            "n_symbols": int(len(syms)),
            "turnover_rate": turnover_rate,
            "effective_turnover_rate": effective_turnover_rate,
            "saturation_rate": saturation_rate,
            "invalid_open_logit_pressure": invalid_open_logit_pressure,
            "raw_capacity_bar_rate": raw_capacity_bar_rate,
            "same_side_open_rate": same_side_open_rate,
            **exposure_metrics,
        })

    turnover_arr = np.asarray(turnover_rates, dtype=np.float64)
    saturation_arr = np.asarray(saturation_rates, dtype=np.float64)
    if turnover_arr.size == 0:
        return {
            "execution_lag_bars": int(execution_lag_bars),
            "turnover_target_rate": float(cg.TURNOVER_TARGET_RATE),
            "mean_turnover_rate": 0.0,
            "max_turnover_rate": 0.0,
            "mean_effective_turnover_rate": 0.0,
            "mean_saturation_rate": 0.0,
            "max_saturation_rate": 0.0,
            "mean_invalid_open_logit_pressure": 0.0,
            "max_invalid_open_logit_pressure": 0.0,
            "mean_raw_capacity_bar_rate": 0.0,
            "max_raw_capacity_bar_rate": 0.0,
            "mean_same_side_open_rate": 0.0,
            "max_same_side_open_rate": 0.0,
            "mean_long_slot_rate": 0.0,
            "mean_short_slot_rate": 0.0,
            "mean_position_slot_rate": 0.0,
            "mean_net_direction_bias": 0.0,
            "mean_capacity_usage": 0.0,
            "max_open_positions": 0,
            "capacity_bar_rate": 0.0,
            "periods": [],
        }
    effective_turnover_arr = np.asarray(
        [float(row["effective_turnover_rate"]) for row in period_rows],
        dtype=np.float64,
    )
    invalid_open_pressure_arr = np.asarray(invalid_open_logit_pressures, dtype=np.float64)
    raw_capacity_bar_arr = np.asarray(capacity_bar_rates, dtype=np.float64)
    same_side_open_arr = np.asarray(same_side_open_rates, dtype=np.float64)
    long_slot_arr = np.asarray(
        [float(row["mean_long_slot_rate"]) for row in period_rows],
        dtype=np.float64,
    )
    short_slot_arr = np.asarray(
        [float(row["mean_short_slot_rate"]) for row in period_rows],
        dtype=np.float64,
    )
    position_slot_arr = np.asarray(
        [float(row["mean_position_slot_rate"]) for row in period_rows],
        dtype=np.float64,
    )
    direction_bias_arr = np.asarray(
        [float(row["mean_net_direction_bias"]) for row in period_rows],
        dtype=np.float64,
    )
    capacity_usage_arr = np.asarray(
        [float(row["mean_capacity_usage"]) for row in period_rows],
        dtype=np.float64,
    )
    capacity_bar_arr = np.asarray(
        [float(row["capacity_bar_rate"]) for row in period_rows],
        dtype=np.float64,
    )
    return {
        "execution_lag_bars": int(execution_lag_bars),
        "turnover_target_rate": float(cg.TURNOVER_TARGET_RATE),
        "mean_turnover_rate": float(turnover_arr.mean()),
        "max_turnover_rate": float(turnover_arr.max()),
        "mean_effective_turnover_rate": float(effective_turnover_arr.mean()),
        "mean_saturation_rate": float(saturation_arr.mean()),
        "max_saturation_rate": float(saturation_arr.max()),
        "mean_invalid_open_logit_pressure": float(invalid_open_pressure_arr.mean()),
        "max_invalid_open_logit_pressure": float(invalid_open_pressure_arr.max()),
        "mean_raw_capacity_bar_rate": float(raw_capacity_bar_arr.mean()),
        "max_raw_capacity_bar_rate": float(raw_capacity_bar_arr.max()),
        "mean_same_side_open_rate": float(same_side_open_arr.mean()),
        "max_same_side_open_rate": float(same_side_open_arr.max()),
        "mean_long_slot_rate": float(long_slot_arr.mean()),
        "mean_short_slot_rate": float(short_slot_arr.mean()),
        "mean_position_slot_rate": float(position_slot_arr.mean()),
        "mean_net_direction_bias": float(direction_bias_arr.mean()),
        "mean_capacity_usage": float(capacity_usage_arr.mean()),
        "max_open_positions": int(max(int(row["max_open_positions"]) for row in period_rows)),
        "capacity_bar_rate": float(capacity_bar_arr.mean()),
        "periods": period_rows,
    }


def _aggregate_contract_metrics_from_period_rows(
    period_rows: list[dict[str, Any]],
    *,
    execution_lag_bars: int,
) -> Dict[str, Any]:
    if not period_rows:
        return {
            "execution_lag_bars": int(execution_lag_bars),
            "turnover_target_rate": float(cg.TURNOVER_TARGET_RATE),
            "mean_turnover_rate": 0.0,
            "max_turnover_rate": 0.0,
            "mean_effective_turnover_rate": 0.0,
            "mean_saturation_rate": 0.0,
            "max_saturation_rate": 0.0,
            "mean_invalid_open_logit_pressure": 0.0,
            "max_invalid_open_logit_pressure": 0.0,
            "mean_raw_capacity_bar_rate": 0.0,
            "max_raw_capacity_bar_rate": 0.0,
            "mean_same_side_open_rate": 0.0,
            "max_same_side_open_rate": 0.0,
            "mean_long_slot_rate": 0.0,
            "mean_short_slot_rate": 0.0,
            "mean_position_slot_rate": 0.0,
            "mean_net_direction_bias": 0.0,
            "mean_capacity_usage": 0.0,
            "max_open_positions": 0,
            "capacity_bar_rate": 0.0,
            "periods": [],
        }

    def mean_of(key: str) -> float:
        return float(np.asarray([float(row.get(key, 0.0)) for row in period_rows], dtype=np.float64).mean())

    def max_of(key: str) -> float:
        return float(np.asarray([float(row.get(key, 0.0)) for row in period_rows], dtype=np.float64).max())

    return {
        "execution_lag_bars": int(execution_lag_bars),
        "turnover_target_rate": float(cg.TURNOVER_TARGET_RATE),
        "mean_turnover_rate": mean_of("turnover_rate"),
        "max_turnover_rate": max_of("turnover_rate"),
        "mean_effective_turnover_rate": mean_of("effective_turnover_rate"),
        "mean_saturation_rate": mean_of("saturation_rate"),
        "max_saturation_rate": max_of("saturation_rate"),
        "mean_invalid_open_logit_pressure": mean_of("invalid_open_logit_pressure"),
        "max_invalid_open_logit_pressure": max_of("invalid_open_logit_pressure"),
        "mean_raw_capacity_bar_rate": mean_of("raw_capacity_bar_rate"),
        "max_raw_capacity_bar_rate": max_of("raw_capacity_bar_rate"),
        "mean_same_side_open_rate": mean_of("same_side_open_rate"),
        "max_same_side_open_rate": max_of("same_side_open_rate"),
        "mean_long_slot_rate": mean_of("mean_long_slot_rate"),
        "mean_short_slot_rate": mean_of("mean_short_slot_rate"),
        "mean_position_slot_rate": mean_of("mean_position_slot_rate"),
        "mean_net_direction_bias": mean_of("mean_net_direction_bias"),
        "mean_capacity_usage": mean_of("mean_capacity_usage"),
        "max_open_positions": int(max(int(row.get("max_open_positions", 0)) for row in period_rows)),
        "capacity_bar_rate": mean_of("capacity_bar_rate"),
        "periods": period_rows,
    }


def _evaluate_mode(
    *,
    genome: np.ndarray,
    precomp: list,
    mode_name: str,
    execution_lag_bars: int,
    futures_fee: float,
    position_state_features_enabled: bool,
) -> Dict[str, Any]:
    old_lag = cg.TRAIN_EXECUTION_LAG_BARS
    old_futures_fee = cg.TRAIN_FUTURES_FEE
    old_position_state = cg.POSITION_STATE_FEATURES_ENABLED
    try:
        cg.TRAIN_EXECUTION_LAG_BARS = int(execution_lag_bars)
        cg.TRAIN_FUTURES_FEE = float(futures_fee)
        cg.POSITION_STATE_FEATURES_ENABLED = bool(position_state_features_enabled)
        t0 = time.time()
        fits, period_ret_lists = _evaluate_population(genome[None].astype(np.float32), precomp)
        elapsed = time.time() - t0
        contract_metrics = _contract_metrics_for_genome(
            genome=genome,
            precomp=precomp,
            execution_lag_bars=execution_lag_bars,
        )
    finally:
        cg.TRAIN_EXECUTION_LAG_BARS = old_lag
        cg.TRAIN_FUTURES_FEE = old_futures_fee
        cg.POSITION_STATE_FEATURES_ENABLED = old_position_state

    period_rets = period_ret_lists[0] if period_ret_lists else []
    return {
        "mode": mode_name,
        "position_state_features_enabled": bool(position_state_features_enabled),
        "fitness": float(fits[0]),
        "execution_lag_bars": int(execution_lag_bars),
        "spot_fee": float(cg.TRAIN_FEE),
        "futures_fee": float(futures_fee),
        "slippage": float(cg.TRAIN_SLIPPAGE),
        "elapsed_sec": float(elapsed),
        "period_stats": _period_stats(period_rets),
        "robust_score": robust_period_score(period_rets),
        "contract_metrics": contract_metrics,
        "period_rets": [float(x) for x in period_rets],
    }


def _evaluate_regime_adaptive_output_bias_mode(
    *,
    source_genome: np.ndarray,
    precomp: list,
    regime_open_bias: Mapping[str, float],
    mode_name: str,
    execution_lag_bars: int,
    futures_fee: float,
    position_state_features_enabled: bool,
) -> Dict[str, Any]:
    old_lag = cg.TRAIN_EXECUTION_LAG_BARS
    old_futures_fee = cg.TRAIN_FUTURES_FEE
    old_position_state = cg.POSITION_STATE_FEATURES_ENABLED
    period_rets: list[float] = []
    period_rows: list[dict[str, Any]] = []
    bias_periods: list[dict[str, Any]] = []
    genome_cache: dict[float, np.ndarray] = {}

    try:
        cg.TRAIN_EXECUTION_LAG_BARS = int(execution_lag_bars)
        cg.TRAIN_FUTURES_FEE = float(futures_fee)
        cg.POSITION_STATE_FEATURES_ENABLED = bool(position_state_features_enabled)
        t0 = time.time()
        for entry in precomp:
            period = str(entry[4]) if len(entry) > 4 else ""
            regime = _contract_regime_label(entry)
            open_bias = _resolve_regime_open_bias(regime, regime_open_bias)
            genome = genome_cache.get(open_bias)
            if genome is None:
                genome = _apply_open_output_bias(source_genome, open_bias)
                genome_cache[open_bias] = genome

            fits, period_ret_lists = _evaluate_population(genome[None].astype(np.float32), [entry])
            if period_ret_lists and period_ret_lists[0]:
                period_ret = float(period_ret_lists[0][0])
            else:
                period_ret = float(fits[0]) if len(fits) else 0.0
            period_rets.append(period_ret)

            metrics = _contract_metrics_for_genome(
                genome=genome,
                precomp=[entry],
                execution_lag_bars=execution_lag_bars,
            )
            if metrics.get("periods"):
                row = dict(metrics["periods"][0])
            else:
                row = {
                    "period": period,
                    "regime": regime,
                    "turnover_rate": 0.0,
                    "effective_turnover_rate": 0.0,
                    "saturation_rate": 0.0,
                    "invalid_open_logit_pressure": 0.0,
                }
            row["open_output_bias"] = float(open_bias)
            period_rows.append(row)
            bias_periods.append({
                "period": period,
                "regime": str(regime),
                "open_output_bias": float(open_bias),
            })
        elapsed = time.time() - t0
    finally:
        cg.TRAIN_EXECUTION_LAG_BARS = old_lag
        cg.TRAIN_FUTURES_FEE = old_futures_fee
        cg.POSITION_STATE_FEATURES_ENABLED = old_position_state

    robust = robust_period_score(period_rets)
    return {
        "mode": mode_name,
        "position_state_features_enabled": bool(position_state_features_enabled),
        "fitness": float(robust.get("robust_utility", np.mean(period_rets) if period_rets else 0.0)),
        "execution_lag_bars": int(execution_lag_bars),
        "spot_fee": float(cg.TRAIN_FEE),
        "futures_fee": float(futures_fee),
        "slippage": float(cg.TRAIN_SLIPPAGE),
        "elapsed_sec": float(elapsed),
        "period_stats": _period_stats(period_rets),
        "robust_score": robust,
        "contract_metrics": _aggregate_contract_metrics_from_period_rows(
            period_rows,
            execution_lag_bars=execution_lag_bars,
        ),
        "period_rets": [float(x) for x in period_rets],
        "regime_adaptive_output_bias": {
            "enabled": True,
            "regime_open_bias": {str(key): float(value) for key, value in regime_open_bias.items()},
            "periods": bias_periods,
        },
    }


def _evaluate_walk_forward(
    *,
    genome: np.ndarray,
    precomp: list,
    folds: list[dict[str, Any]],
    execution_lag_bars: int,
    futures_fee: float,
    position_state_features_enabled: bool,
) -> Dict[str, Any]:
    fold_reports: list[dict[str, Any]] = []
    aggregate_period_rets: list[float] = []

    for fold in folds:
        validation_precomp = split_precomp_by_periods(precomp, fold["validation_periods"])
        if not validation_precomp:
            continue

        validation_report = _evaluate_mode(
            genome=genome,
            precomp=validation_precomp,
            mode_name="walk_forward_validation_fee_fixed_nextbar",
            execution_lag_bars=execution_lag_bars,
            futures_fee=futures_fee,
            position_state_features_enabled=position_state_features_enabled,
        )
        aggregate_period_rets.extend(validation_report["period_rets"])
        fold_reports.append({
            "fold_id": fold["fold_id"],
            "train_years": fold["train_years"],
            "validation_years": fold["validation_years"],
            "n_train_periods": len(fold["train_periods"]),
            "n_embargo_periods": len(fold["embargo_periods"]),
            "n_validation_periods": len(fold["validation_periods"]),
            "validation": validation_report,
        })

    return {
        "n_folds": len(fold_reports),
        "validation_aggregate_stats": _period_stats(aggregate_period_rets),
        "validation_aggregate_robust_score": robust_period_score(aggregate_period_rets),
        "folds": fold_reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved genetic genomes under cost/timing contracts.")
    parser.add_argument(
        "--genome",
        action="append",
        default=None,
        help="Path to .npy genome. Can be repeated. Defaults to best_genome.npy.",
    )
    parser.add_argument(
        "--regime-adaptive-output-bias-genome",
        action="append",
        default=None,
        help=(
            "Path to a source .npy genome evaluated with per-period open-output bias "
            "selected by regime. Can be repeated."
        ),
    )
    parser.add_argument(
        "--regime-open-bias",
        action="append",
        default=None,
        help=(
            "Open-output bias for adaptive evaluation as REGIME=BIAS. "
            "Can be repeated for crash, bearish, neutral, bullish, or default."
        ),
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "Results" / "neiro_genetics" / "genetics_contract_eval.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="Optional evaluation start date passed to the genetics data loader.",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="Optional evaluation end date passed to the genetics data loader.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional Retrodate CSV directory override for this evaluation only.",
    )
    parser.add_argument(
        "--exchange",
        default=os.getenv("CRYPTO_EXCHANGE", ""),
        help="Exchange/domain for this genetics evaluation, for example MEXC or BITGET.",
    )
    parser.add_argument(
        "--symbol",
        "--symbols",
        dest="symbol_filter",
        action="append",
        default=None,
        help=(
            "Restrict evaluation to these base symbols. Can be repeated or comma-separated, "
            "for example --symbols BTC,ETH,SOL."
        ),
    )
    parser.add_argument(
        "--walk-forward",
        action="store_true",
        help="Also evaluate validation slices from rolling train/validation year folds.",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        default=4,
        help="Number of training years per walk-forward fold.",
    )
    parser.add_argument(
        "--validation-years",
        type=int,
        default=1,
        help="Number of validation years per walk-forward fold.",
    )
    parser.add_argument(
        "--final-test-year",
        type=int,
        default=2025,
        help="Year reserved as final holdout and excluded from walk-forward validation.",
    )
    parser.add_argument(
        "--embargo-months",
        type=int,
        default=0,
        help="Number of final training months removed before each validation window.",
    )
    args = parser.parse_args()

    regime_open_bias = _parse_regime_open_bias_specs(args.regime_open_bias)
    symbol_filter = _parse_symbol_filter(args.symbol_filter)
    adaptive_genome_paths = args.regime_adaptive_output_bias_genome or []
    if adaptive_genome_paths and not regime_open_bias:
        raise ValueError("--regime-adaptive-output-bias-genome requires at least one --regime-open-bias")

    genome_paths = args.genome or [str(GENETICS_DIR / "Agents" / "genetics" / "best_genome.npy")]
    genomes = []
    for raw_path in genome_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        genome = np.load(path).astype(np.float32)
        if genome.shape[0] != cg.GENOME_SIZE:
            raise ValueError(f"{path} genome size {genome.shape[0]} != expected {cg.GENOME_SIZE}")
        genomes.append((path, genome, _infer_position_state_features_enabled(path)))
    adaptive_genomes = []
    for raw_path in adaptive_genome_paths:
        path = Path(raw_path)
        if not path.is_absolute():
            path = ROOT / path
        genome = np.load(path).astype(np.float32)
        if genome.shape[0] != cg.GENOME_SIZE:
            raise ValueError(f"{path} genome size {genome.shape[0]} != expected {cg.GENOME_SIZE}")
        adaptive_genomes.append((path, genome, _infer_position_state_features_enabled(path)))

    old_cfg_start = cg._cx._CFG.get("start_date")
    old_cfg_end = cg._cx._CFG.get("end_date")
    old_data_dir = cg._cx.DATA_DIR
    try:
        if args.start_date is not None:
            cg._cx._CFG["start_date"] = args.start_date
        if args.end_date is not None:
            cg._cx._CFG["end_date"] = args.end_date

        data_dir = _resolve_eval_data_dir(args.data_dir)
        if args.data_dir is not None:
            cg._cx.DATA_DIR = str(data_dir)
        timeframe = {"1m": "1m", "1h": "1h", "1d": "1d"}.get(cg._cx.TIMEFRAME, "1m")
        retrodate_selection = _validate_requested_retrodate_files(
            data_dir,
            timeframe=timeframe,
            start_date=cg._cx._CFG.get("start_date"),
            end_date=cg._cx._CFG.get("end_date"),
        )
        print(
            "[eval] retrodate files="
            + ", ".join(item.path.name for item in retrodate_selection.valid_reports)
        )
        print("[eval] loading precomputed features...")
        precomp = cg._load_precomp()
    finally:
        if old_cfg_start is not None:
            cg._cx._CFG["start_date"] = old_cfg_start
        if old_cfg_end is not None:
            cg._cx._CFG["end_date"] = old_cfg_end
        cg._cx.DATA_DIR = old_data_dir

    if symbol_filter:
        periods_before = len(precomp)
        precomp, symbol_filter_summary = _filter_precomp_symbols(precomp, symbol_filter)
        print(
            "[eval] symbol_filter="
            + ",".join(symbol_filter)
            + f" periods={len(precomp)}/{periods_before}"
        )
    else:
        symbol_filter_summary = {
            "requested_symbols": [],
            "periods_before": int(len(precomp)),
            "periods_after": int(len(precomp)),
            "periods_dropped": 0,
            "periods": [],
        }

    if not precomp:
        raise RuntimeError("No precomputed periods loaded")
    print(f"[eval] periods={len(precomp)}")

    modes = [
        {
            "mode_name": "legacy_samebar_spot_fee_for_futures",
            "execution_lag_bars": 0,
            "futures_fee": float(cg.TRAIN_FEE),
        },
        {
            "mode_name": "fee_fixed_samebar",
            "execution_lag_bars": 0,
            "futures_fee": float(cg.TRAIN_FUTURES_FEE),
        },
        {
            "mode_name": "fee_fixed_nextbar",
            "execution_lag_bars": 1,
            "futures_fee": float(cg.TRAIN_FUTURES_FEE),
        },
    ]

    report: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exchange": str(args.exchange or "").strip().upper(),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "periods": len(precomp),
        "retrodate_data_dir": str(retrodate_selection.report.path),
        "retrodate_timeframe": timeframe,
        "retrodate_requested_years": list(retrodate_selection.requested_years),
        "retrodate_valid_files": [item.path.name for item in retrodate_selection.valid_reports],
        "retrodate_missing_years": list(retrodate_selection.missing_years),
        "default_position_state_features_enabled": bool(cg.POSITION_STATE_FEATURES_ENABLED),
        "genomes": [],
    }
    if symbol_filter:
        report["symbol_filter"] = symbol_filter_summary
    if regime_open_bias:
        report["regime_open_bias"] = {str(key): float(value) for key, value in regime_open_bias.items()}
    folds: list[dict[str, Any]] = []
    if args.walk_forward:
        period_labels = [str(entry[4]) for entry in precomp if len(entry) > 4]
        folds = build_rolling_year_folds(
            period_labels,
            train_years=args.train_years,
            validation_years=args.validation_years,
            final_test_year=args.final_test_year,
            embargo_months=args.embargo_months,
        )
        report["walk_forward_config"] = {
            "train_years": args.train_years,
            "validation_years": args.validation_years,
            "final_test_year": args.final_test_year,
            "embargo_months": args.embargo_months,
            "n_folds": len(folds),
        }

    for path, genome, position_state_features_enabled in genomes:
        print(f"[eval] genome={path}")
        genome_report = {
            "path": str(path),
            "position_state_features_enabled": bool(position_state_features_enabled),
            "modes": [],
        }
        for mode in modes:
            print(f"[eval]   mode={mode['mode_name']}")
            genome_report["modes"].append(
                _evaluate_mode(
                    genome=genome,
                    precomp=precomp,
                    position_state_features_enabled=position_state_features_enabled,
                    **mode,
                )
            )
        if args.walk_forward:
            print(f"[eval]   walk_forward_folds={len(folds)}")
            genome_report["walk_forward"] = _evaluate_walk_forward(
                genome=genome,
                precomp=precomp,
                folds=folds,
                execution_lag_bars=1,
                futures_fee=float(cg.TRAIN_FUTURES_FEE),
                position_state_features_enabled=position_state_features_enabled,
            )
        report["genomes"].append(genome_report)

    for path, genome, position_state_features_enabled in adaptive_genomes:
        print(f"[eval] regime_adaptive_output_bias_genome={path}")
        genome_report = {
            "path": f"{path}::regime_adaptive_output_bias",
            "source_path": str(path),
            "position_state_features_enabled": bool(position_state_features_enabled),
            "regime_adaptive_output_bias": {
                "enabled": True,
                "regime_open_bias": {str(key): float(value) for key, value in regime_open_bias.items()},
            },
            "modes": [],
        }
        for mode in modes:
            print(f"[eval]   mode={mode['mode_name']}")
            genome_report["modes"].append(
                _evaluate_regime_adaptive_output_bias_mode(
                    source_genome=genome,
                    precomp=precomp,
                    regime_open_bias=regime_open_bias,
                    position_state_features_enabled=position_state_features_enabled,
                    **mode,
                )
            )
        if args.walk_forward:
            print("[eval]   walk_forward skipped for regime_adaptive_output_bias")
            genome_report["walk_forward"] = {
                "skipped": True,
                "reason": "per-period adaptive output bias is evaluated in report modes only",
            }
        report["genomes"].append(genome_report)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
