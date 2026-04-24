from __future__ import annotations

from typing import Iterable, Sequence

from agent_core import AgentIntent, ExecutionPlan


class IntentExecutionPlanner:
    def __init__(
        self,
        *,
        open_threshold: float = 0.22,
        close_threshold: float = 0.18,
        max_new_per_bar: int = 1,
        max_positions: int = 4,
        long_action: int = 5,
        short_action: int = 7,
        close_action: int = 8,
    ):
        self.open_threshold = float(open_threshold)
        self.close_threshold = float(close_threshold)
        self.max_new_per_bar = int(max_new_per_bar)
        self.max_positions = int(max_positions)
        self.long_action = int(long_action)
        self.short_action = int(short_action)
        self.close_action = int(close_action)

    def _record(self, plan: ExecutionPlan, sym: str, agent: str, score: float) -> None:
        bucket = plan.contributors.setdefault(sym, {})
        bucket[agent] = bucket.get(agent, 0.0) + float(score)

    def plan(
        self,
        symbols: Iterable[str],
        intents: Sequence[AgentIntent],
        *,
        open_symbols: Iterable[str],
        blacklist: Iterable[str] = (),
    ) -> ExecutionPlan:
        plan = ExecutionPlan(actions={str(sym): 0 for sym in symbols})
        open_set = {str(sym) for sym in open_symbols}
        blacklist_set = {str(sym) for sym in blacklist}

        for intent in intents:
            sym = str(intent.symbol)
            if sym not in plan.actions:
                continue
            self._record(plan, sym, intent.agent, intent.score)
            if intent.direction == "close":
                plan.close_scores[sym] = plan.close_scores.get(sym, 0.0) + float(intent.score)
            elif intent.direction == "long":
                plan.long_scores[sym] = plan.long_scores.get(sym, 0.0) + float(intent.score)
            elif intent.direction == "short":
                plan.short_scores[sym] = plan.short_scores.get(sym, 0.0) + float(intent.score)

        for sym, score in plan.close_scores.items():
            if score >= self.close_threshold:
                plan.actions[sym] = self.close_action

        candidates = []
        for sym in plan.actions.keys():
            if plan.actions[sym] == self.close_action:
                continue
            if sym in open_set or sym in blacklist_set:
                continue
            long_score = float(plan.long_scores.get(sym, 0.0))
            short_score = float(plan.short_scores.get(sym, 0.0))
            if long_score < self.open_threshold and short_score < self.open_threshold:
                continue
            if long_score >= self.open_threshold and long_score > short_score * 1.05:
                candidates.append((sym, self.long_action, long_score))
            elif short_score >= self.open_threshold and short_score > long_score * 1.05:
                candidates.append((sym, self.short_action, short_score))

        candidates.sort(key=lambda item: item[2], reverse=True)
        n_open = len(open_set)
        n_new = 0
        for sym, action, weight in candidates:
            if n_new >= self.max_new_per_bar or n_open >= self.max_positions:
                break
            plan.actions[sym] = action
            plan.notes[sym] = f"entry_score={weight:.4f}"
            n_new += 1
            n_open += 1

        return plan
