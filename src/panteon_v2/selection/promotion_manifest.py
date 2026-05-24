"""Pure signal-key promotion manifest builder."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence, Tuple


# Z-score for the lower bound of a one-sided 95% confidence interval.
Z_95_ONE_SIDED: float = 1.6448536


@dataclass(frozen=True)
class PromotionManifestConfig:
    min_full_closed_trades: int = 50
    min_latest_closed_trades: int = 10
    min_full_pnl_pct: float = 0.0
    min_latest_pnl_pct: float = 0.0
    min_full_pnl_per_trade_lcb_pct: float = 0.0
    min_latest_pnl_per_trade_lcb_pct: float = 0.0
    pnl_per_trade_lcb_z: float = Z_95_ONE_SIDED
    max_drawdown_pct: float = 25.0
    min_win_rate_pct: float = 52.0
    # Wilson lower bound of the one-sided 95% CI on win_rate; default 50%.
    min_win_rate_lcb_pct: float = 50.0
    win_rate_lcb_z: float = Z_95_ONE_SIDED
    max_recent_downside_usd: float = 5.0


@dataclass(frozen=True)
class PromotionRejection:
    signal_key: str
    reason: str


@dataclass(frozen=True)
class PromotionManifest:
    allowed_signal_keys: Tuple[str, ...]
    rejected: Tuple[PromotionRejection, ...]

    def as_dict(self) -> dict[str, list[object]]:
        return {
            "allowed_signal_keys": list(self.allowed_signal_keys),
            "rejected": [asdict(row) for row in self.rejected],
        }


def build_promotion_manifest(
    rows: Sequence[Mapping[str, object]],
    config: PromotionManifestConfig = PromotionManifestConfig(),
) -> PromotionManifest:
    allowed_signal_keys: set[str] = set()
    rejected: list[PromotionRejection] = []

    for row in rows:
        signal_key = str(row.get("signal_key") or "").strip()
        if not signal_key:
            continue

        reason = _promotion_rejection_reason(row, config)
        if reason is None:
            allowed_signal_keys.add(signal_key)
        else:
            rejected.append(PromotionRejection(signal_key=signal_key, reason=reason))

    return PromotionManifest(
        allowed_signal_keys=tuple(sorted(allowed_signal_keys)),
        rejected=tuple(sorted(rejected, key=lambda row: (row.signal_key, row.reason))),
    )


def _promotion_rejection_reason(
    row: Mapping[str, object],
    config: PromotionManifestConfig,
) -> str | None:
    """Return the first gate that rejects a signal-key row.

    PnL-per-trade LCB gates are opt-in: thresholds <= 0 disable those gates and
    do not require legacy callers to provide the LCB fields. Once a threshold is
    positive, missing or malformed LCB fields reject the row conservatively.
    """
    if _as_int(row.get("full_closed_trades")) < config.min_full_closed_trades:
        return "full_closed_trades_below_gate"
    if _as_int(row.get("latest_closed_trades")) < config.min_latest_closed_trades:
        return "latest_closed_trades_below_gate"
    full_pnl_pct = _as_float(row.get("full_pnl_pct"))
    if full_pnl_pct is None or full_pnl_pct < config.min_full_pnl_pct:
        return "full_pnl_below_gate"
    latest_pnl_pct = _as_float(row.get("latest_pnl_pct"))
    if latest_pnl_pct is None or latest_pnl_pct < config.min_latest_pnl_pct:
        return "latest_pnl_below_gate"
    max_drawdown_pct = _as_float(row.get("max_drawdown_pct"))
    if max_drawdown_pct is None or max_drawdown_pct > config.max_drawdown_pct:
        return "drawdown_above_gate"
    win_rate_pct = _as_float(row.get("win_rate_pct"))
    if win_rate_pct is None or win_rate_pct < config.min_win_rate_pct:
        return "win_rate_below_gate"
    if config.min_win_rate_lcb_pct > 0.0:
        closed = _as_int(row.get("full_closed_trades"))
        lcb_pct = _beta_win_rate_lcb_pct(
            win_rate_pct=win_rate_pct,
            closed_trades=closed,
            z=float(config.win_rate_lcb_z),
        )
        if lcb_pct < config.min_win_rate_lcb_pct:
            return "win_rate_lcb_below_gate"
    if config.min_full_pnl_per_trade_lcb_pct > 0.0:
        full_pnl_per_trade_lcb_pct = _as_float(row.get("full_pnl_per_trade_lcb_pct"))
        if (
            full_pnl_per_trade_lcb_pct is None
            or full_pnl_per_trade_lcb_pct < config.min_full_pnl_per_trade_lcb_pct
        ):
            return "full_pnl_per_trade_lcb_below_gate"
    if config.min_latest_pnl_per_trade_lcb_pct > 0.0:
        latest_pnl_per_trade_lcb_pct = _as_float(row.get("latest_pnl_per_trade_lcb_pct"))
        if (
            latest_pnl_per_trade_lcb_pct is None
            or latest_pnl_per_trade_lcb_pct < config.min_latest_pnl_per_trade_lcb_pct
        ):
            return "latest_pnl_per_trade_lcb_below_gate"
    recent_downside_usd = _as_float(row.get("recent_downside_usd"))
    if recent_downside_usd is None or recent_downside_usd > config.max_recent_downside_usd:
        return "recent_downside_above_gate"
    return None


def _beta_win_rate_lcb_pct(
    *,
    win_rate_pct: float,
    closed_trades: int,
    z: float,
) -> float:
    """Lower bound of the one-sided 95% CI on win_rate via Wilson score.

    Wilson lower bound is well-behaved for small samples and is the standard
    test-statistic for binomial proportions. It is preferred over the normal
    approximation of the Beta posterior (which is biased downward at small n).

    Returns the bound in percent (0..100).
    """
    if closed_trades <= 0:
        return 0.0
    n = float(closed_trades)
    wins = max(0.0, min(n, n * float(win_rate_pct) / 100.0))
    p = wins / n
    z = max(0.0, float(z))
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = p + z2 / (2.0 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, (centre - margin) / denom) * 100.0


def _as_int(value: object) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed
