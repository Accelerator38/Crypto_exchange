from __future__ import annotations

from typing import Dict, Iterable, Optional


class PromotionGateAgent:
    """
    Session-level promotion gate for enabling next-generation players in live mode.

    The gate compares a candidate shadow player against one or more baseline
    shadow players on the same warmup/session data.
    """

    def __init__(
        self,
        *,
        candidate_shadow: str = "V_PanteonNextResearch",
        baseline_shadows: Iterable[str] = (
            "V_Panteon_shadow",
            "V_PanteonResearch",
        ),
        min_signals: int = 3,
        min_entries: int = 1,
        min_closed_trades: int = 1,
        max_pnl_gap_pct: float = 1.25,
        max_sharpe_gap: float = 0.20,
        max_dd_multiplier: float = 1.20,
        max_dd_buffer_pct: float = 1.50,
    ):
        self.candidate_shadow = str(candidate_shadow)
        self.baseline_shadows = tuple(str(name) for name in baseline_shadows)
        self.min_signals = int(min_signals)
        self.min_entries = int(min_entries)
        self.min_closed_trades = int(min_closed_trades)
        self.max_pnl_gap_pct = float(max_pnl_gap_pct)
        self.max_sharpe_gap = float(max_sharpe_gap)
        self.max_dd_multiplier = float(max_dd_multiplier)
        self.max_dd_buffer_pct = float(max_dd_buffer_pct)

    def _normalize_metrics(self, metrics: Optional[dict]) -> Optional[dict]:
        if not isinstance(metrics, dict):
            return None
        pnl_pct = float(metrics.get("pnl_pct", 0.0) or 0.0)
        sharpe = float(metrics.get("sharpe", 0.0) or 0.0)
        max_dd = abs(float(metrics.get("max_dd", 0.0) or 0.0))
        signals = int(metrics.get("signals", 0) or 0)
        entries = int(metrics.get("entries", 0) or 0)
        closed = int(metrics.get("closed_trades", metrics.get("total_trades", 0)) or 0)
        win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
        activity = min(signals, 40) * 0.03 + min(entries, 20) * 0.06 + min(closed, 12) * 0.07
        score = pnl_pct + sharpe * 1.25 + activity - max_dd * 0.20
        return {
            "pnl_pct": pnl_pct,
            "sharpe": sharpe,
            "max_dd": max_dd,
            "signals": signals,
            "entries": entries,
            "closed_trades": closed,
            "win_rate": win_rate,
            "score": score,
        }

    def evaluate(self, shadow_perf: Dict[str, dict]) -> dict:
        result = {
            "candidate_shadow": self.candidate_shadow,
            "baseline_shadow": "",
            "ready": False,
            "reasons": [],
            "candidate_metrics": None,
            "baseline_metrics": None,
        }
        candidate = self._normalize_metrics((shadow_perf or {}).get(self.candidate_shadow))
        if candidate is None:
            result["reasons"].append(f"candidate {self.candidate_shadow} is missing from shadow perf")
            return result
        result["candidate_metrics"] = dict(candidate)

        baselines = []
        for name in self.baseline_shadows:
            metrics = self._normalize_metrics((shadow_perf or {}).get(name))
            if metrics is not None:
                baselines.append((name, metrics))
        if baselines:
            baselines.sort(key=lambda item: item[1]["score"], reverse=True)
            baseline_name, baseline = baselines[0]
            result["baseline_shadow"] = baseline_name
            result["baseline_metrics"] = dict(baseline)
        else:
            baseline_name, baseline = "", None
            result["reasons"].append("no baseline shadow player available for comparison")

        if candidate["signals"] < self.min_signals:
            result["reasons"].append(
                f"candidate generated too few signals ({candidate['signals']} < {self.min_signals})"
            )
        if candidate["entries"] < self.min_entries:
            result["reasons"].append(
                f"candidate opened too few trades ({candidate['entries']} < {self.min_entries})"
            )
        if candidate["closed_trades"] < self.min_closed_trades:
            result["reasons"].append(
                "candidate has not produced enough closed trades for a meaningful comparison"
            )

        if baseline is not None:
            if candidate["pnl_pct"] + self.max_pnl_gap_pct < baseline["pnl_pct"]:
                result["reasons"].append(
                    f"candidate PnL trails baseline by more than {self.max_pnl_gap_pct:.2f} pct "
                    f"({candidate['pnl_pct']:.2f}% vs {baseline['pnl_pct']:.2f}% from {baseline_name})"
                )
            if candidate["sharpe"] + self.max_sharpe_gap < baseline["sharpe"]:
                result["reasons"].append(
                    f"candidate Sharpe is materially below baseline "
                    f"({candidate['sharpe']:.2f} vs {baseline['sharpe']:.2f} from {baseline_name})"
                )
            allowed_dd = max(
                baseline["max_dd"] * self.max_dd_multiplier,
                baseline["max_dd"] + self.max_dd_buffer_pct,
            )
            if candidate["max_dd"] > allowed_dd:
                result["reasons"].append(
                    f"candidate drawdown exceeds gate ({candidate['max_dd']:.2f}% > {allowed_dd:.2f}%)"
                )

        result["ready"] = len(result["reasons"]) == 0
        return result
