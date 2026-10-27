from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exia.genetic_primus.benchmarks_v4 import mean_reversion_policy, momentum_policy
from exia.genetic_primus.contracts import FEATURE_NAMES, STATIC_FEATURE_NAMES
from exia.genetic_primus.labels import _strict_row_median_three
from exia.genetic_primus.rolling import (
    fixed_baseline_scaler,
    fixed_baseline_sha256,
    validate_rolling_output_directory,
)
from exia.genetic_primus.rolling_v4 import should_resume_island


def test_fixed_baselines_use_raw_returns_and_window_independent_sha() -> None:
    frame = pd.DataFrame(
        {
            name: [-0.02, 0.03] if name == "return_1" else [0.0, 0.0]
            for name in STATIC_FEATURE_NAMES
        }
    )
    static = fixed_baseline_scaler().transform(frame)
    dynamic = np.zeros(len(FEATURE_NAMES) - len(STATIC_FEATURE_NAMES), dtype=np.float32)
    momentum = momentum_policy()
    targets = [momentum.target(np.concatenate([row, dynamic])) for row in static]

    assert targets == [-1, 1]
    assert fixed_baseline_sha256(momentum.policy_id) == fixed_baseline_sha256(
        momentum_policy().policy_id
    )
    assert fixed_baseline_sha256(momentum.policy_id) != fixed_baseline_sha256(
        mean_reversion_policy().policy_id
    )


def test_strict_three_horizon_median_avoids_dataframe_nanmedian() -> None:
    values = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [3.0, 1.0, 2.0],
            [np.nan, 1.0, 2.0],
            [-np.inf, 0.0, np.inf],
        ],
        dtype=np.float64,
    )
    actual = _strict_row_median_three(values)
    assert actual[:2].tolist() == [2.0, 2.0]
    assert np.isnan(actual[2])
    assert actual[3] == 0.0


def test_resume_guard_accepts_only_incomplete_output(tmp_path: Path) -> None:
    new_output = tmp_path / "new"
    validate_rolling_output_directory(new_output, resume_partial=False)
    with pytest.raises(FileNotFoundError):
        validate_rolling_output_directory(new_output, resume_partial=True)

    partial = tmp_path / "partial"
    (partial / "origin_0000" / "island_00").mkdir(parents=True)
    validate_rolling_output_directory(partial, resume_partial=True)
    with pytest.raises(FileExistsError):
        validate_rolling_output_directory(partial, resume_partial=False)

    (partial / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        validate_rolling_output_directory(partial, resume_partial=True)

    (partial / "manifest.json").unlink()
    (partial / "origins.parquet").write_bytes(b"partial")
    with pytest.raises(FileExistsError):
        validate_rolling_output_directory(partial, resume_partial=True)


def test_island_resume_is_enabled_only_when_checkpoint_exists(tmp_path: Path) -> None:
    checkpoint = tmp_path / "island_00" / "checkpoint.npz"
    assert not should_resume_island(checkpoint)
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-placeholder")
    assert should_resume_island(checkpoint)
