from __future__ import annotations

import importlib
import json
import types

import numpy as np
import pytest


def _genome(
    path: str,
    rets: list[float],
    turnovers: list[float],
    saturations: list[float],
    raw_capacity_bars: list[float] | None = None,
    same_side_opens: list[float] | None = None,
) -> dict:
    raw_capacity_bars = raw_capacity_bars or [0.0] * len(rets)
    same_side_opens = same_side_opens or [0.0] * len(rets)
    return {
        "path": path,
        "position_state_features_enabled": False,
        "modes": [
            {
                "mode": "fee_fixed_nextbar",
                "period_rets": [float(value) for value in rets],
                "period_stats": {
                    "mean_ret": float(sum(rets) / len(rets)),
                    "min_ret": float(min(rets)),
                    "max_ret": float(max(rets)),
                    "positive_period_pct": float(
                        sum(1 for value in rets if value > 0.0) * 100.0 / len(rets)
                    ),
                },
                "robust_score": {
                    "passes_default_gates": True,
                    "failed_gates": [],
                },
                "contract_metrics": {
                    "mean_turnover_rate": float(sum(turnovers) / len(turnovers)),
                    "max_turnover_rate": float(max(turnovers)),
                    "mean_saturation_rate": float(sum(saturations) / len(saturations)),
                    "max_saturation_rate": float(max(saturations)),
                    "mean_invalid_open_logit_pressure": 0.0,
                    "max_invalid_open_logit_pressure": 0.0,
                    "mean_raw_capacity_bar_rate": float(
                        sum(raw_capacity_bars) / len(raw_capacity_bars)
                    ),
                    "max_raw_capacity_bar_rate": float(max(raw_capacity_bars)),
                    "mean_same_side_open_rate": float(
                        sum(same_side_opens) / len(same_side_opens)
                    ),
                    "max_same_side_open_rate": float(max(same_side_opens)),
                },
            }
        ],
    }


def _report(
    *,
    baseline_rets: list[float],
    candidate_rets: list[float],
    candidate_turnovers: list[float] | None = None,
    candidate_saturations: list[float] | None = None,
    candidate_raw_capacity_bars: list[float] | None = None,
    candidate_same_side_opens: list[float] | None = None,
) -> dict:
    n = len(baseline_rets)
    return {
        "exchange": "BITGET",
        "start_date": "2026-02-01",
        "end_date": "2026-03-31",
        "genomes": [
            _genome("baseline.npy", baseline_rets, [0.02] * n, [0.02] * n),
            _genome(
                "candidate.npy",
                candidate_rets,
                candidate_turnovers or [0.03] * n,
                candidate_saturations or [0.03] * n,
                candidate_raw_capacity_bars,
                candidate_same_side_opens,
            ),
        ],
    }


def test_oos_gate_rejects_candidate_that_loses_to_baseline():
    tool = importlib.import_module("tools.run_genetics_validated_smoke")
    report = _report(
        baseline_rets=[0.20, 0.30],
        candidate_rets=[-0.10, 0.10],
    )

    summary = tool.summarize_single_candidate_oos_gate(report)

    assert summary["promotion_eligible"] is False
    assert summary["selected_is_baseline"] is True
    assert "oos_mean_ret" in summary["promotion_failures"]
    assert "oos_min_ret" in summary["promotion_failures"]
    assert summary["candidate"]["mean_ret_delta"] == pytest.approx(-0.25)


def test_oos_gate_accepts_candidate_only_when_it_beats_baseline_and_risk_limits():
    tool = importlib.import_module("tools.run_genetics_validated_smoke")
    report = _report(
        baseline_rets=[0.20, 0.30],
        candidate_rets=[0.25, 0.36],
        candidate_turnovers=[0.04, 0.05],
        candidate_saturations=[0.03, 0.04],
    )

    summary = tool.summarize_single_candidate_oos_gate(
        report,
        max_turnover_rate=0.10,
        max_saturation_rate=0.10,
    )

    assert summary["promotion_eligible"] is True
    assert summary["selected_is_baseline"] is False
    assert summary["promotion_failures"] == []
    assert summary["candidate"]["mean_ret_delta"] == pytest.approx(0.055)
    assert summary["candidate"]["min_ret_delta"] == pytest.approx(0.05)


def test_oos_gate_rejects_candidate_that_breaks_risk_limits():
    tool = importlib.import_module("tools.run_genetics_validated_smoke")
    report = _report(
        baseline_rets=[0.20, 0.30],
        candidate_rets=[0.25, 0.36],
        candidate_turnovers=[0.04, 0.18],
        candidate_saturations=[0.03, 0.17],
    )

    summary = tool.summarize_single_candidate_oos_gate(
        report,
        max_turnover_rate=0.10,
        max_saturation_rate=0.10,
    )

    assert summary["promotion_eligible"] is False
    assert summary["selected_is_baseline"] is True
    assert "oos_max_turnover" in summary["promotion_failures"]
    assert "oos_max_saturation" in summary["promotion_failures"]


def test_oos_gate_rejects_candidate_that_breaks_capacity_and_same_side_limits():
    tool = importlib.import_module("tools.run_genetics_validated_smoke")
    report = _report(
        baseline_rets=[0.20, 0.30],
        candidate_rets=[0.25, 0.36],
        candidate_raw_capacity_bars=[0.10, 0.40],
        candidate_same_side_opens=[0.01, 0.09],
    )

    summary = tool.summarize_single_candidate_oos_gate(
        report,
        max_raw_capacity_bar_rate=0.25,
        max_same_side_open_rate=0.05,
    )

    assert summary["promotion_eligible"] is False
    assert "oos_max_raw_capacity_bar" in summary["promotion_failures"]
    assert "oos_max_same_side_open" in summary["promotion_failures"]
    assert summary["candidate"]["max_raw_capacity_bar_rate"] == pytest.approx(0.40)
    assert summary["candidate"]["max_same_side_open_rate"] == pytest.approx(0.09)


def test_training_window_uses_existing_smoke_summary_when_available(tmp_path):
    tool = importlib.import_module("tools.run_genetics_validated_smoke")
    run_dir = tmp_path / "existing_run"
    run_dir.mkdir()
    (run_dir / "smoke_summary.json").write_text(
        json.dumps(
            {
                "start_date": "2025-10-01",
                "end_date": "2026-01-31",
                "population": 18,
                "generations": 2,
            }
        ),
        encoding="utf-8",
    )
    args = types.SimpleNamespace(
        train_start_date="2025-01-01",
        train_end_date="2025-12-31",
        population=24,
        generations=3,
    )

    window = tool._training_window_payload(run_dir, args)

    assert window == {
        "start_date": "2025-10-01",
        "end_date": "2026-01-31",
        "population": 18,
        "generations": 2,
    }


def test_symbol_filter_prefers_explicit_symbols_over_settings(tmp_path):
    tool = importlib.import_module("tools.run_genetics_validated_smoke")
    settings_path = tmp_path / "settings.txt"
    settings_path.write_text(
        "symbols = all\nbitget_symbols = BTC,ETH,SOL\n",
        encoding="utf-8",
    )

    symbols = tool._resolve_symbol_filter(
        "BITGET",
        ["DOGE/USDT, xrp_usdt"],
        settings_path=settings_path,
    )

    assert symbols == ("DOGE", "XRP")


def test_symbol_filter_uses_exchange_scoped_settings(tmp_path):
    tool = importlib.import_module("tools.run_genetics_validated_smoke")
    settings_path = tmp_path / "settings.txt"
    settings_path.write_text(
        "symbols = all\nbitget_symbols = BTC,ETH,SOL\n",
        encoding="utf-8",
    )

    symbols = tool._resolve_symbol_filter(
        "BITGET",
        None,
        settings_path=settings_path,
    )

    assert symbols == ("BTC", "ETH", "SOL")


def test_symbol_filter_all_means_unrestricted_universe(tmp_path):
    tool = importlib.import_module("tools.run_genetics_validated_smoke")
    settings_path = tmp_path / "settings.txt"
    settings_path.write_text(
        "symbols = all\nmexc_symbols = all\n",
        encoding="utf-8",
    )

    symbols = tool._resolve_symbol_filter(
        "MEXC",
        None,
        settings_path=settings_path,
    )

    assert symbols == ()


def test_symbol_args_are_appended_to_child_commands():
    tool = importlib.import_module("tools.run_genetics_validated_smoke")

    args = ["python", "tools\\evaluate_genetics_contract.py"]
    tool._append_symbol_filter_args(args, ("BTC", "ETH"))

    assert args[-2:] == ["--symbols", "BTC,ETH"]


def test_training_tuning_args_are_appended_to_child_command():
    tool = importlib.import_module("tools.run_genetics_validated_smoke")
    namespace = types.SimpleNamespace(
        feasible_best_seed_frac=0.40,
        feasible_best_seed_sigma=0.03,
        feasible_open_logit_bias=0.50,
        feasible_mutation_scope="output",
    )

    args = ["python", "tools\\run_genetics_smoke_training.py"]
    tool._append_training_tuning_args(args, namespace)

    assert "--feasible-best-seed-frac" in args
    assert "0.4" in args
    assert "--feasible-best-seed-sigma" in args
    assert "0.03" in args
    assert "--feasible-open-logit-bias" in args
    assert "0.5" in args
    assert args[-2:] == ["--feasible-mutation-scope", "output"]


def test_smoke_training_filters_precomp_to_requested_symbols():
    tool = importlib.import_module("tools.run_genetics_smoke_training")
    feat = np.arange(2 * 3 * 2, dtype=np.float32).reshape(2, 3, 2)
    prices = np.arange(2 * 3, dtype=np.float64).reshape(2, 3)
    precomp = [(
        feat,
        prices,
        ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
        4,
        "2026-04",
        1.5,
        "neutral",
    )]

    filtered, summary = tool._filter_precomp_symbols(precomp, ("SOL", "BTC"))

    assert summary["requested_symbols"] == ["BTC", "SOL"]
    assert summary["periods_after"] == 1
    assert summary["periods"][0]["kept_symbols"] == ["BTC/USDT", "SOL/USDT"]
    assert filtered[0][2] == ["BTC/USDT", "SOL/USDT"]
    assert np.array_equal(filtered[0][0], feat[:, [0, 2], :])
    assert np.array_equal(filtered[0][1], prices[:, [0, 2]])
