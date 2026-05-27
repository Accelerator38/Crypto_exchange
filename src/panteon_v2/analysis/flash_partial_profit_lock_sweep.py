"""A/B sweep command planning for Flash partial profit lock."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_PARTIAL_LOCK_TRIGGERS = (2.0, 3.0, 4.0)
DEFAULT_PARTIAL_LOCK_CLOSE_FRACTIONS = (0.25, 0.33, 0.5)


def build_partial_profit_lock_ab_variants(
    *,
    triggers: Sequence[float] = DEFAULT_PARTIAL_LOCK_TRIGGERS,
    close_fractions: Sequence[float] = DEFAULT_PARTIAL_LOCK_CLOSE_FRACTIONS,
) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    for trigger in triggers:
        for fraction in close_fractions:
            variants.append(
                {
                    "name": _variant_name(trigger, fraction),
                    "trigger_pnl_pct": float(trigger),
                    "close_fraction": float(fraction),
                    "skip_protected_signal_keys": True,
                    "production_default_enabled": False,
                }
            )
    return variants


def build_partial_profit_lock_command(
    *,
    python: str,
    candidate_runner: str | Path,
    manifest: str | Path,
    results_root: str | Path,
    years: str,
    variant: Mapping[str, Any],
    max_bars: int | None = None,
    risk_capital_fraction: float = 0.10,
    risk_max_open_positions: int = 8,
    max_new_opens_per_bar: int = 1,
    stale_exit_age_bars: int = 168,
    disable_flash_audit_events: bool = False,
    disable_shadow_audit_events: bool = False,
    disable_step_result_retention: bool = False,
) -> list[str]:
    command = [
        str(python),
        str(candidate_runner),
        "--manifest",
        str(manifest),
        "--results-root",
        str(Path(results_root) / str(variant["name"])),
        "--years",
        str(years),
        "--risk-capital-fraction",
        f"{float(risk_capital_fraction):.4g}",
        "--stride-minutes",
        "60",
        "--max-new-opens-per-bar",
        str(int(max_new_opens_per_bar)),
        "--risk-max-open-positions",
        str(int(risk_max_open_positions)),
        "--stale-exit-age-bars",
        str(int(stale_exit_age_bars)),
        "--enable-partial-profit-lock",
        "--partial-profit-lock-trigger-pnl-pct",
        f"{float(variant['trigger_pnl_pct']):.4g}",
        "--partial-profit-lock-close-fraction",
        f"{float(variant['close_fraction']):.4g}",
        "--partial-profit-lock-min-age-bars",
        "2",
    ]
    if max_bars is not None:
        command += ["--max-bars", str(int(max_bars))]
    if disable_flash_audit_events:
        command.append("--disable-flash-audit-events")
    if disable_shadow_audit_events:
        command.append("--disable-shadow-audit-events")
    if disable_step_result_retention:
        command.append("--disable-step-result-retention")
    return command


def write_partial_profit_lock_ab_plan(
    *,
    output_csv: str | Path,
    python: str,
    candidate_runner: str | Path,
    manifest: str | Path,
    results_root: str | Path,
    years: str,
    max_bars: int | None = None,
    variants: Sequence[Mapping[str, Any]] | None = None,
) -> Path:
    rows: list[dict[str, Any]] = []
    for variant in variants or build_partial_profit_lock_ab_variants():
        command = build_partial_profit_lock_command(
            python=python,
            candidate_runner=candidate_runner,
            manifest=manifest,
            results_root=results_root,
            years=years,
            max_bars=max_bars,
            variant=variant,
            disable_flash_audit_events=True,
            disable_shadow_audit_events=True,
            disable_step_result_retention=True,
        )
        rows.append(
            {
                **dict(variant),
                "years": years,
                "max_bars": "" if max_bars is None else int(max_bars),
                "command": " ".join(command),
            }
        )
    out = Path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    return out


def _variant_name(trigger: float, fraction: float) -> str:
    trigger_text = str(float(trigger)).rstrip("0").rstrip(".").replace(".", "p")
    fraction_text = str(float(fraction)).rstrip("0").rstrip(".").replace(".", "p")
    return f"partial_lock_trigger_{trigger_text}_fraction_{fraction_text}"
