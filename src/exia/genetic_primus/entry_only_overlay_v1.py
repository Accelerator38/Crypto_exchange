"""Causal, offline entry permission; EMA exclusively controls subsequent exits."""
from __future__ import annotations

import numpy as np

POLICY = {
    "version": "entry_only_overlay/1",
    "initial_position": "flat_per_window",
    "entry": "registered_positive_logit_gate; delayed_entry_allowed",
    "hold": "ignore_gate_while_valid_EMA_direction_unchanged",
    "exit": "EMA_flat_or_direction_change_or_invalid_features",
    "reversal": "close_then_check_gate_for_new_direction",
    "reset": "forget_held_position_before_processing_reset_bar",
    "terminal_exit": "unchanged_accounting_interval_end",
    "runtime_authority": False,
    "promotion_authority": False,
}


def entry_only_targets(ema_targets, entry_allowed, *, reset_before=None):
    """Map time x instrument inputs to targets with no lookahead or wall clock."""
    ema = np.asarray(ema_targets)
    allowed = np.asarray(entry_allowed)
    if ema.ndim != 2 or not all(ema.shape) or allowed.shape != ema.shape:
        raise ValueError("matching nonempty time-by-instrument matrices required")
    if not np.isfinite(ema).all() or not np.isin(ema, [-1, 0, 1]).all():
        raise ValueError("EMA directions must be finite -1/0/1")
    if allowed.dtype != np.bool_:
        raise ValueError("explicit boolean gate required")
    resets = np.zeros(len(ema), dtype=bool) if reset_before is None else np.asarray(reset_before)
    if resets.shape != (len(ema),) or resets.dtype != np.bool_:
        raise ValueError("boolean reset mask required")
    result = np.zeros(ema.shape, dtype=np.int8)
    held = np.zeros(ema.shape[1], dtype=np.int8)
    for row in range(len(ema)):
        if resets[row]:
            held.fill(0)
        held = np.where(held == ema[row], held, 0).astype(np.int8)
        enter = (held == 0) & (ema[row] != 0) & allowed[row]
        held = np.where(enter, ema[row], held).astype(np.int8)
        result[row] = held
    return result
