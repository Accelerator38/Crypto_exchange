from __future__ import annotations

import logging
from collections import OrderedDict, deque
from typing import Dict, Optional

from agent_core import MarketSnapshot, intent_from_action
from agent_execution import IntentExecutionPlanner
from panteon import (
    CarryFlowAgentV2,
    ExternalSignalAgent,
    FundingArb,
    LiveAfterShock,
    LiveCrashHunter,
    LiveMeanRev,
    LiveRegimePullback,
    LiveTrendFollow,
    LiveVolCompress,
    Panteon,
    ResearchValidatorAgent,
    RichardDennisTurtle,
)


log = logging.getLogger("player_next")


class PanteonNextResearch(Panteon):
    """
    Shadow-first next-generation player with a modular pipeline.

    Pipeline:
      MarketSnapshot -> AgentIntent -> IntentExecutionPlanner -> PositionExitGovernor

    The class stays API-compatible with Panteon, but is intended for shadow use
    until the pipeline proves itself against the current live player.
    """

    ROTATION_INT = 60
    ENABLE_META_PLAYERS = False
    MIN_AGENTS = 3
    MAX_AGENTS = 6
    MIN_WEIGHT = 0.10
    MAX_POS = 5
    OPEN_THRESHOLD = 0.22
    CLOSE_THRESHOLD = 0.18
    MAX_NEW_PER_BAR = 2

    SHADOW_MAP = {
        **Panteon.SHADOW_MAP,
        "V_CarryFlowAgentV2": "CarryFlowAgentV2",
        "V_ExternalSignalAgent": "ExternalSignalAgent",
        "V_ResearchValidatorAgent": "ResearchValidatorAgent",
    }

    def __init__(self):
        super().__init__()
        self._fa = FundingArb()
        self._las = LiveAfterShock()
        self._lch = LiveCrashHunter()
        self._lrp = LiveRegimePullback()
        self._lmr = LiveMeanRev()
        self._ltf = LiveTrendFollow()
        self._lvc = LiveVolCompress()
        self._rd = RichardDennisTurtle()
        self._cfv2 = CarryFlowAgentV2()
        self._extsig = ExternalSignalAgent()
        self._rv = ResearchValidatorAgent()
        self._planner = IntentExecutionPlanner(
            open_threshold=self.OPEN_THRESHOLD,
            close_threshold=self.CLOSE_THRESHOLD,
            max_new_per_bar=self.MAX_NEW_PER_BAR,
            max_positions=self.MAX_POS,
            long_action=5,
            short_action=7,
            close_action=8,
        )
        self._pipeline_logger = log
        self._agent_pool = OrderedDict(
            (label, agent) for _, label, agent in self.iter_subagents()
        )

        if not self._memory_bootstrap_loaded:
            desired = OrderedDict([
                ("CarryFlowAgentV2", 0.24),
                ("ResearchValidatorAgent", 0.19),
                ("LiveRegimePullback", 0.16),
                ("FundingArb", 0.13),
                ("LiveMeanRev", 0.10),
                ("RichardDennis", 0.08),
                ("LiveAfterShock", 0.05),
                ("LiveCrashHunter", 0.05),
                ("ExternalSignalAgent", 0.04),
            ])
            filtered = OrderedDict(
                (label, weight) for label, weight in desired.items()
                if label in self._agent_pool
            )
            total = sum(filtered.values())
            if total > 0:
                self._active_weights = {
                    label: weight / total for label, weight in filtered.items()
                }

    def iter_subagents(self):
        for attr_name, label in (
            ("_cfv2", "CarryFlowAgentV2"),
            ("_extsig", "ExternalSignalAgent"),
            ("_rv", "ResearchValidatorAgent"),
            ("_fa", "FundingArb"),
            ("_lrp", "LiveRegimePullback"),
            ("_lmr", "LiveMeanRev"),
            ("_las", "LiveAfterShock"),
            ("_lch", "LiveCrashHunter"),
            ("_ltf", "LiveTrendFollow"),
            ("_lvc", "LiveVolCompress"),
            ("_rd", "RichardDennis"),
            ("_gb", "GeneticsBullish"),
            ("_gbr", "GeneticsBearish"),
        ):
            agent = getattr(self, attr_name, None)
            if agent is not None:
                yield attr_name, label, agent

    def _score_shadow_candidate(self, label: str, perf: dict) -> float:
        pnl = float(perf.get("pnl_pct", 0.0) or 0.0)
        sharpe = float(perf.get("sharpe", 0.0) or 0.0)
        max_dd = abs(float(perf.get("max_dd", 0.0) or 0.0))
        signals = int(perf.get("signals", 0) or 0)
        entries = int(perf.get("entries", 0) or 0)
        closed = int(perf.get("closed_trades", perf.get("total_trades", 0)) or 0)
        activity = min(signals, 40) * 0.03 + min(entries, 24) * 0.07 + min(closed, 16) * 0.08
        inactivity_penalty = 1.4 if signals == 0 and entries == 0 else 0.0
        research_bonus = 0.0
        if label in {"CarryFlowAgentV2", "ResearchValidatorAgent", "ExternalSignalAgent"}:
            research_bonus = 0.20 if signals > 0 else 0.0
        return pnl * 0.90 + sharpe * 1.35 + activity + research_bonus - max_dd * 0.40 - inactivity_penalty

    def _refresh_snapshot(
        self,
        prices: dict,
        volumes: dict,
        month: Optional[int],
        portfolio_value: Optional[float],
        bar_index: Optional[int],
    ) -> MarketSnapshot:
        self._t = bar_index if bar_index is not None else getattr(self, "_t", 0) + 1
        t = self._t
        prev_regime = self._canonical_market_regime(self._r) if self._r is not None else None
        for sym, price in prices.items():
            self._ph.setdefault(sym, deque(maxlen=5760)).append(float(price))
            self._vh.setdefault(sym, deque(maxlen=5760)).append(float(volumes.get(sym, 0.0) or 0.0))

        if t - self._lr >= self.STATE_INT or self._r is None:
            self._r = self._detect_regime_fast()
            self._lr = t
        if t - self._last_context_bar >= self.CONTEXT_REFRESH_INT or not self._current_context:
            self._current_context = self._build_market_context(month=month)
            self._last_context_bar = t

        regime = self._canonical_market_regime(self._r or "neutral")
        regime_changed = prev_regime is not None and regime != prev_regime
        if self._shadow_perf is not None and (regime_changed or t - self._last_rotation >= self.ROTATION_INT):
            if regime_changed and not self._quiet_info_logs():
                log.info("  [PanteonNextResearch] regime changed %s -> %s; forcing agent rotation", prev_regime, regime)
            self._rotate_agents()
            self._last_rotation = t

        return MarketSnapshot(
            prices={str(sym): float(px) for sym, px in prices.items()},
            volumes={str(sym): float(volumes.get(sym, 0.0) or 0.0) for sym in prices},
            month=month,
            portfolio_value=portfolio_value,
            bar_index=t,
            regime=regime,
            context=dict(self._current_context or {}),
        )

    def _collect_intents(self, snapshot: MarketSnapshot):
        intents = []
        for label, weight in sorted(self._active_weights.items(), key=lambda item: item[1], reverse=True):
            agent = self._agent_pool.get(label)
            if agent is None:
                continue
            try:
                actions = agent.act(
                    snapshot.prices,
                    volumes=snapshot.volumes,
                    month=snapshot.month,
                    portfolio_value=snapshot.portfolio_value,
                    bar_index=snapshot.bar_index,
                ) or {}
            except Exception:
                actions = {}
            for sym, action in actions.items():
                if not action or sym not in snapshot.prices:
                    continue
                score = float(weight)
                if action in (1, 2, 4, 5, 6, 7):
                    score = self._symbol_vote_weight(label, sym, float(weight))
                intent = intent_from_action(
                    label,
                    sym,
                    action,
                    score,
                    metadata={
                        "bar": snapshot.bar_index,
                        "regime": snapshot.regime,
                    },
                )
                if intent is not None:
                    intents.append(intent)
        return intents

    def _finalize_metadata(self, out: dict, contributors: Dict[str, Dict[str, float]], regime: str) -> None:
        self._last_contributors = {}
        self._last_agreement_scores = {}
        self._last_risk_multipliers = {}
        if not hasattr(self, "_sub_signal_counts"):
            self._sub_signal_counts = {}
        total_active_weight = max(
            sum(float(weight) for weight in self._active_weights.values()),
            1e-9,
        )
        for sym, action in out.items():
            if action == 0:
                continue
            voters = sorted(contributors.get(sym, {}).items(), key=lambda item: item[1], reverse=True)
            voter_names = [agent for agent, _ in voters]
            if voter_names:
                self._last_contributors[sym] = "+".join(voter_names)
            agreement = min(
                1.0,
                sum(float(weight) for _, weight in voters) / total_active_weight,
            )
            self._last_agreement_scores[sym] = float(agreement)
            self._last_risk_multipliers[sym] = self._signal_risk_multiplier(
                sym, action, agreement, voter_names,
            )
            for agent, _ in voters:
                self._sub_signal_counts[agent] = self._sub_signal_counts.get(agent, 0) + 1
        self._last_regime = regime

    def act(self, prices, volumes, month=None, portfolio_value=None, bar_index=None):
        snapshot = self._refresh_snapshot(prices, volumes, month, portfolio_value, bar_index)
        intents = self._collect_intents(snapshot)
        plan = self._planner.plan(
            snapshot.prices.keys(),
            intents,
            open_symbols=self._open_pos.keys(),
            blacklist=self._BLACKLIST,
        )

        out = dict(plan.actions)
        self._position_safety.apply_exit_actions(
            self._open_pos,
            snapshot.prices,
            int(snapshot.bar_index or 0),
            out,
            stop_loss_pct=self.SL_PCT,
            take_profit_pct=self.TP_PCT,
            trail_pct=self.TRAIL_PCT,
            stale_bars=self.STALE_BARS,
            logger=log,
        )

        for sym in list(out.keys()):
            if sym in self._BLACKLIST and out[sym] != 0:
                out[sym] = 0

        self._position_safety.sync_positions(
            self._open_pos,
            out,
            snapshot.prices,
            int(snapshot.bar_index or 0),
        )
        self._finalize_metadata(out, plan.contributors, snapshot.regime)
        if any(action != 0 for action in out.values()):
            self.save_memory_snapshot(reason="next-pipeline-signal")
        return out

PanteonNext = PanteonNextResearch
