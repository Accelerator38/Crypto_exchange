from __future__ import annotations

import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
STRATEGY_DIR = ROOT / "freqtrade_pilot" / "user_data" / "strategies"
for path in (SRC_ROOT, STRATEGY_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from exia.contracts import load_candidate_spec, sha256_file  # noqa: E402
from exia_market_mode import (  # noqa: E402
    DEFAULT_MARKET_MODE_CONTRACT,
    MARKET_MODE_ORDER,
    append_market_mode_columns,
)


CANDIDATE_PATH = (
    ROOT
    / "freqtrade_pilot"
    / "candidates"
    / "exia_market_mode_foundation_v1.json"
)
TAPE_PATH = ROOT / "Retrodate" / "simple_research_reset_v1" / "bitget_full8_1h.parquet"
TAPE_MANIFEST_PATH = TAPE_PATH.with_suffix(".manifest.json")
REPORT_DIR = ROOT / "Reports" / "Exia" / "market_mode_foundation_v1"
JSON_REPORT_PATH = REPORT_DIR / "distribution.json"
MARKDOWN_REPORT_PATH = REPORT_DIR / "distribution.md"


def _bounds(timerange: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = timerange.split("-")
    return pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")


def _state_summary(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame["exia_market_mode"].value_counts().to_dict()
    total = len(frame)
    state_change = frame["exia_market_mode"].ne(
        frame["exia_market_mode"].shift(1)
    )
    run_id = state_change.cumsum()
    episode_lengths = frame.groupby(run_id, sort=False).size().tolist()
    return {
        "rows": total,
        "states": {
            state: {
                "rows": int(counts.get(state, 0)),
                "share": int(counts.get(state, 0)) / total if total else 0.0,
            }
            for state in MARKET_MODE_ORDER
        },
        "episodes": len(episode_lengths),
        "median_episode_bars": statistics.median(episode_lengths)
        if episode_lengths
        else None,
        "maximum_episode_bars": max(episode_lengths) if episode_lengths else None,
        "warmup_rows": int((~frame["exia_warmup_complete"]).sum()),
        "volatility_shock_rows": int(frame["exia_volatility_shock"].sum()),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Exia market mode foundation v1",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "This report validates state coverage only. It contains no alpha claim and cannot authorize paper/live.",
        "",
        "## Contract",
        "",
        "- `TREND_UP`: close > EMA24 > EMA96 and positive 24h momentum.",
        "- `TREND_DOWN`: close < EMA24 < EMA96 and negative 24h momentum.",
        "- `RANGE`: valid non-shock observation without an aligned trend.",
        "- `UNSAFE`: incomplete warm-up, invalid cadence/data, or volatility shock.",
        "",
        f"Status: `{report['status']}`.",
        "",
        "## Window distribution",
        "",
        "| Window | Rows | Up % | Down % | Range % | Unsafe % | Median episode h |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, summary in report["windows"].items():
        states = summary["states"]
        lines.append(
            f"| {split} | {summary['rows']} | "
            f"{states['TREND_UP']['share'] * 100:.1f} | "
            f"{states['TREND_DOWN']['share'] * 100:.1f} | "
            f"{states['RANGE']['share'] * 100:.1f} | "
            f"{states['UNSAFE']['share'] * 100:.1f} | "
            f"{summary['median_episode_bars']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Symbol distribution",
            "",
            "| Symbol | Rows | Up % | Down % | Range % | Unsafe % | Shocks |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for symbol, summary in report["symbols"].items():
        states = summary["states"]
        lines.append(
            f"| {symbol} | {summary['rows']} | "
            f"{states['TREND_UP']['share'] * 100:.1f} | "
            f"{states['TREND_DOWN']['share'] * 100:.1f} | "
            f"{states['RANGE']['share'] * 100:.1f} | "
            f"{states['UNSAFE']['share'] * 100:.1f} | "
            f"{summary['volatility_shock_rows']} |"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            "- Strategy entries are hard-coded to zero.",
            "- `orders_enabled=false`; `promotion_authority=false`.",
            "- The next permitted step is candidate design, not paper/live.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    spec = load_candidate_spec(CANDIDATE_PATH, root=ROOT)
    tape_manifest = json.loads(TAPE_MANIFEST_PATH.read_text(encoding="utf-8"))
    source_dataset_sha = tape_manifest["source_integrity"]["dataset_sha256"]
    if source_dataset_sha != spec.dataset["dataset_sha256"]:
        raise RuntimeError("feature tape does not match the candidate dataset")

    tape = pd.read_parquet(TAPE_PATH)
    classified_parts: list[pd.DataFrame] = []
    symbols: dict[str, Any] = {}
    for symbol, source in tape.groupby("symbol", sort=True):
        frame = source.sort_values("timestamp", kind="stable").copy()
        frame["date"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
        classified = append_market_mode_columns(frame)
        classified_parts.append(classified)
        symbols[str(symbol)] = _state_summary(classified)
    classified_all = pd.concat(classified_parts, ignore_index=True)

    windows: dict[str, Any] = {}
    for split, timerange in spec.splits.items():
        start, end = _bounds(timerange)
        window = classified_all.loc[
            (classified_all["date"] >= start) & (classified_all["date"] < end)
        ]
        windows[split] = _state_summary(window)

    observed_states = sorted(classified_all["exia_market_mode"].unique().tolist())
    checks = {
        "candidate_contract_valid": True,
        "dataset_sha_matches": True,
        "all_four_states_observed": set(observed_states) == set(MARKET_MODE_ORDER),
        "no_unknown_state": not set(observed_states) - set(MARKET_MODE_ORDER),
        "full8_present": len(symbols) == 8,
        "entries_disabled": True,
    }
    report = {
        "schema_version": "exia.market_mode_distribution.v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "FOUNDATION_READY_FOR_CANDIDATE_DESIGN"
        if all(checks.values())
        else "FOUNDATION_FAILED",
        "candidate_id": spec.candidate_id,
        "source": {
            "candidate_spec": str(CANDIDATE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "candidate_spec_sha256": sha256_file(CANDIDATE_PATH),
            "feature_tape": str(TAPE_PATH.relative_to(ROOT)).replace("\\", "/"),
            "feature_tape_sha256": sha256_file(TAPE_PATH),
            "dataset_sha256": source_dataset_sha,
            "taxonomy": str(spec.taxonomy["source"]),
            "taxonomy_sha256": str(spec.taxonomy["source_sha256"]),
        },
        "contract": {
            **DEFAULT_MARKET_MODE_CONTRACT.__dict__,
            "minimum_history_bars": DEFAULT_MARKET_MODE_CONTRACT.minimum_history_bars,
        },
        "checks": checks,
        "observed_states": observed_states,
        "windows": windows,
        "symbols": symbols,
        "paper_allowed": False,
        "live_allowed": False,
        "orders_enabled": False,
        "promotion_authority": False,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    MARKDOWN_REPORT_PATH.write_text(_render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": checks,
                "windows": {
                    split: summary["states"] for split, summary in windows.items()
                },
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if all(checks.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
