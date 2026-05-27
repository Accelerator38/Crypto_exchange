from pathlib import Path

from panteon_v2.analysis.flash_contra_validation import (
    build_flash_contra_risk_sizing_manifest,
    build_flash_contra_validation_manifest,
    write_flash_contra_risk_sizing_manifest,
    write_flash_contra_validation_manifest,
)


def _metrics(pnl_pct: float, *, max_dd_pct: float = 0.0, closed: int = 1):
    return {
        "panteon_pnl_pct": pnl_pct,
        "panteon_realized_pnl_usd": pnl_pct * 10.0,
        "panteon_max_drawdown_pct": max_dd_pct,
        "closed_trades": closed,
    }


def _split(name: str, role: str, baseline: dict, candidate: dict):
    return {
        "name": name,
        "role": role,
        "baseline": baseline,
        "candidate": candidate,
    }


def test_contra_validation_rejects_train_gain_when_validation_loses():
    key = "agent:GeneticsNeutral|ADA/USDT|FUT_LONG_FULL"

    manifest = build_flash_contra_validation_manifest(
        contra_signal_keys=[key],
        split_results=[
            _split(
                "2026_train",
                "train",
                _metrics(6.65, max_dd_pct=1.69),
                _metrics(6.92, max_dd_pct=0.0),
            ),
            _split(
                "2025_validation",
                "validation",
                _metrics(1.74, max_dd_pct=0.0),
                _metrics(1.29, max_dd_pct=0.0),
            ),
        ],
    )

    assert manifest["summary"]["eligible"] is False
    assert manifest["allowed_contra_signal_keys"] == []
    assert manifest["rejected_contra_signal_keys"] == [
        {
            "signal_key": key,
            "reason": "validation_pnl_not_strictly_better:2025_validation",
        }
    ]
    assert "validation_pnl_not_strictly_better:2025_validation" in manifest[
        "promotion_failures"
    ]


def test_contra_validation_rejects_validation_tie():
    key = "agent:GeneticsNeutral|ADA/USDT|FUT_LONG_FULL"

    manifest = build_flash_contra_validation_manifest(
        contra_signal_keys=[key],
        split_results=[
            _split(
                "2025_validation",
                "validation",
                _metrics(1.0),
                _metrics(1.0),
            ),
        ],
    )

    assert manifest["summary"]["eligible"] is False
    assert manifest["allowed_contra_signal_keys"] == []
    assert "validation_tie:2025_validation" in manifest["promotion_failures"]


def test_contra_validation_rejects_validation_drawdown_worse():
    key = "agent:GeneticsNeutral|ADA/USDT|FUT_LONG_FULL"

    manifest = build_flash_contra_validation_manifest(
        contra_signal_keys=[key],
        split_results=[
            _split(
                "2025_validation",
                "validation",
                _metrics(1.0, max_dd_pct=0.5),
                _metrics(1.2, max_dd_pct=0.8),
            ),
        ],
    )

    assert manifest["summary"]["eligible"] is False
    assert manifest["allowed_contra_signal_keys"] == []
    assert "validation_max_dd_worse:2025_validation" in manifest[
        "promotion_failures"
    ]


def test_contra_validation_requires_oos_split_by_default():
    key = "agent:GeneticsNeutral|ADA/USDT|FUT_LONG_FULL"

    manifest = build_flash_contra_validation_manifest(
        contra_signal_keys=[key],
        split_results=[
            _split("2025_validation", "validation", _metrics(1.0), _metrics(1.2)),
        ],
    )

    assert manifest["summary"]["eligible"] is False
    assert manifest["summary"]["require_oos_split"] is True
    assert manifest["allowed_contra_signal_keys"] == []
    assert "oos_split_missing" in manifest["promotion_failures"]


def test_contra_validation_allows_validation_only_only_when_explicit():
    key = "agent:GeneticsNeutral|ADA/USDT|FUT_LONG_FULL"

    manifest = build_flash_contra_validation_manifest(
        contra_signal_keys=[key],
        split_results=[
            _split("2025_validation", "validation", _metrics(1.0), _metrics(1.2)),
        ],
        require_oos_split=False,
    )

    assert manifest["summary"]["eligible"] is True
    assert manifest["summary"]["require_oos_split"] is False
    assert manifest["allowed_contra_signal_keys"] == [key]


def test_contra_validation_allows_keys_only_after_strict_validation_and_oos_win():
    keys = [
        "agent:GeneticsNeutral|ADA/USDT|FUT_LONG_FULL",
        "agent:GeneticsBearish|ADA/USDT|FUT_LONG_FULL",
    ]

    manifest = build_flash_contra_validation_manifest(
        contra_signal_keys=keys,
        split_results=[
            _split("2026_train", "train", _metrics(6.65), _metrics(6.92)),
            _split("2025_validation", "validation", _metrics(1.74), _metrics(1.90)),
            _split("2024_oos", "oos", _metrics(0.25), _metrics(0.35)),
        ],
    )

    assert manifest["summary"]["eligible"] is True
    assert manifest["allowed_contra_signal_keys"] == keys
    assert manifest["rejected_contra_signal_keys"] == []


def test_contra_risk_sizing_allows_bounded_pnl_decay_when_drawdown_improves():
    key = "agent:GeneticsNeutral|ADA/USDT|FUT_LONG_FULL"

    manifest = build_flash_contra_risk_sizing_manifest(
        contra_signal_keys=[key],
        split_results=[
            _split(
                "2025_validation",
                "validation",
                _metrics(10.00, max_dd_pct=4.00),
                _metrics(9.90, max_dd_pct=2.80),
            ),
            _split(
                "2026_h1_oos",
                "oos",
                _metrics(5.00, max_dd_pct=3.50),
                _metrics(4.80, max_dd_pct=2.30),
            ),
        ],
        risk_mult=0.40,
    )

    assert manifest["summary"]["eligible"] is True
    assert manifest["summary"]["mode"] == "risk_sizing"
    assert manifest["summary"]["min_drawdown_improvement_pct"] == 1.0
    assert manifest["summary"]["max_pnl_degradation_pct"] == 0.25
    assert manifest["risk_sizing_contra_signal_keys"] == [
        {"signal_key": key, "risk_mult": 0.40}
    ]


def test_contra_risk_sizing_rejects_when_oos_pnl_decay_is_too_large():
    key = "agent:GeneticsNeutral|ADA/USDT|FUT_LONG_FULL"

    manifest = build_flash_contra_risk_sizing_manifest(
        contra_signal_keys=[key],
        split_results=[
            _split(
                "2026_h1_oos",
                "oos",
                _metrics(5.00, max_dd_pct=3.50),
                _metrics(4.60, max_dd_pct=2.20),
            ),
        ],
        require_validation_split=False,
    )

    assert manifest["summary"]["eligible"] is False
    assert manifest["risk_sizing_contra_signal_keys"] == []
    assert "oos_pnl_degradation_too_large:2026_h1_oos" in manifest[
        "promotion_failures"
    ]


def test_write_contra_validation_manifest_outputs_json_and_markdown(tmp_path: Path):
    key = "agent:GeneticsNeutral|ADA/USDT|FUT_LONG_FULL"

    json_path, md_path = write_flash_contra_validation_manifest(
        tmp_path,
        contra_signal_keys=[key],
        split_results=[
            _split("2025_validation", "validation", _metrics(1.0), _metrics(1.1)),
        ],
        require_oos_split=False,
    )

    assert json_path.name == "flash_genetics_contra_validation_manifest.json"
    assert md_path.name == "flash_genetics_contra_validation_manifest.md"
    assert json_path.exists()
    assert md_path.exists()


def test_write_contra_risk_sizing_manifest_outputs_json_and_markdown(tmp_path: Path):
    key = "agent:GeneticsNeutral|ADA/USDT|FUT_LONG_FULL"

    json_path, md_path = write_flash_contra_risk_sizing_manifest(
        tmp_path,
        contra_signal_keys=[key],
        split_results=[
            _split(
                "2025_validation",
                "validation",
                _metrics(10.00, max_dd_pct=4.00),
                _metrics(9.90, max_dd_pct=2.80),
            ),
            _split(
                "2026_h1_oos",
                "oos",
                _metrics(5.00, max_dd_pct=3.50),
                _metrics(4.80, max_dd_pct=2.30),
            ),
        ],
        risk_mult=0.40,
    )

    assert json_path.name == "flash_genetics_contra_risk_sizing_manifest.json"
    assert md_path.name == "flash_genetics_contra_risk_sizing_manifest.md"
    assert json_path.exists()
    assert md_path.exists()
