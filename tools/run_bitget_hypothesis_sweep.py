from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]


def _csv(values: object) -> str:
    if isinstance(values, str):
        return values
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        return ",".join(str(item) for item in values)
    return ""


def _append_runner_args(command: list[str], *, flag: str, values: object) -> None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return
    for value in values:
        text = str(value).strip()
        if text:
            command.append(f"{flag}={text}")


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    exchange = str(manifest.get("exchange") or "").upper()
    if exchange != "BITGET":
        raise ValueError(f"manifest exchange must be BITGET, got {exchange!r}")


def build_sweep_commands(
    manifest: Mapping[str, Any],
    *,
    python_executable: str,
    results_root: Path,
    reports_root: Path,
) -> list[dict[str, Any]]:
    _validate_manifest(manifest)
    commands: list[dict[str, Any]] = []
    for item in manifest.get("hypotheses", []):
        if not isinstance(item, Mapping):
            continue
        hypothesis_id = str(item["id"])
        label = str(item.get("single_component_candidate_label") or "").strip()
        candidate_variant = str(item.get("candidate_variant") or "").strip()
        if not candidate_variant:
            if not label:
                raise ValueError(
                    f"hypothesis {hypothesis_id!r} must define candidate_variant "
                    "or single_component_candidate_label"
                )
            candidate_variant = f"single_component__{label}"
        result_dir = results_root / hypothesis_id
        report_dir = reports_root / hypothesis_id
        command = [
            python_executable,
            "tools/run_panteon3_pre_live_matrix.py",
            "--data-dir",
            str(item.get("data_dir", manifest.get("data_dir", "Retrodate/mexc_bitget_futures"))),
            "--results-root",
            str(result_dir),
            "--reports-dir",
            str(report_dir),
            "--years",
            str(item.get("years", manifest.get("default_years", "2026"))),
            "--max-bars",
            str(item.get("max_bars", manifest.get("default_max_bars", 480))),
            "--window-skip-bars",
            _csv(item.get("window_skip_bars", manifest.get("default_window_skip_bars", [0]))),
            "--candidate-variant",
            candidate_variant,
            "--min-filled",
            str(item.get("min_filled", 20)),
            "--min-closed-trades",
            str(item.get("min_closed_trades", 10)),
            "--min-profitable-windows",
            str(item.get("min_profitable_windows", 2)),
            "--min-positive-regimes",
            str(item.get("min_positive_regimes", 1)),
            "--min-positive-symbols",
            str(item.get("min_positive_symbols", 1)),
            "--max-drawdown-usd",
            str(item.get("max_drawdown_usd", 0.2)),
            "--require-cost-attribution",
            "--fail-on-promotion-failure",
        ]
        if label:
            command.extend(["--single-component-candidate-label", label])
        baseline_compare_mode = str(
            item.get(
                "baseline_compare_mode",
                manifest.get("default_baseline_compare_mode", ""),
            )
            or ""
        ).strip()
        if baseline_compare_mode:
            command.extend(["--baseline-compare-mode", baseline_compare_mode])
        if bool(item.get("include_derivatives_context_actors", False)):
            command.append("--include-derivatives-context-actors")
        _append_runner_args(
            command,
            flag="--extra-runner-arg",
            values=item.get("extra_runner_args", []),
        )
        _append_runner_args(
            command,
            flag="--candidate-runner-arg",
            values=item.get("candidate_runner_args", []),
        )
        commands.append({
            "id": hypothesis_id,
            "actor": str(item.get("actor") or ""),
            "single_component_candidate_label": label,
            "candidate_variant": candidate_variant,
            "command": command,
            "results_root": str(result_dir),
            "reports_dir": str(report_dir),
            "symbols": list(item.get("symbols") or []),
            "candidate_runner_args": list(item.get("candidate_runner_args") or []),
        })
    return commands


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Bitget hypothesis matrix sweep.")
    parser.add_argument("--manifest", default="configs/bitget_hypotheses_20260706.json")
    parser.add_argument("--results-root", default="Results/BitgetHypothesisSweep/20260706")
    parser.add_argument("--reports-root", default="Reports/BitgetHypothesisSweep/20260706")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    commands = build_sweep_commands(
        manifest,
        python_executable=args.python,
        results_root=Path(args.results_root),
        reports_root=Path(args.reports_root),
    )
    plan_path = Path(args.reports_root) / "sweep_plan.json"
    _write_json(plan_path, commands)
    if args.dry_run:
        print(json.dumps({"plan": str(plan_path), "commands": len(commands)}, indent=2))
        return 0

    failures: list[dict[str, Any]] = []
    for entry in commands:
        completed = subprocess.run(entry["command"], cwd=ROOT, check=False)
        entry["returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            failures.append(entry)

    result_path = Path(args.reports_root) / "sweep_results.json"
    _write_json(result_path, commands)
    print(json.dumps({"results": str(result_path), "failures": len(failures)}, indent=2))
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
