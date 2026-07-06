"""Регистрация v1-агентов в v2 AgentRegistry.

Это место в panteon_v2/, где допустим импорт из panteon_runtime/.
Здесь мы оборачиваем v1-агенты под v2 Agent Protocol через V1AgentAdapter.

Использование:
    from panteon_v2.app.agent_bootstrap import register_all_v1_agents
    registry = AgentRegistry()
    register_all_v1_agents(registry)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, List, Mapping, Optional, Sequence, Tuple

from ..domain.types import Action, MarketSnapshot, Regime
from ..selection import AgentRegistry
from ..shadow.adapters import (
    GeneticsRegimeRouterV2AgentAdapter,
    GeneticsV2AgentAdapter,
    V1AgentAdapter,
)


log = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────
# Список известных v1-агентов с реальными именами классов
# ────────────────────────────────────────────────────────────────────

# Формат: (label_in_v2, "module_path:ClassName")
# label_in_v2 — короткое имя без V_ и суффиксов, используется в QM seed
# и в Player.agent_labels.
KNOWN_V1_AGENTS: List[Tuple[str, str]] = [
    # Базовые из panteon_agents
    ("FundingArb",          "panteon_agents:FundingArb"),
    ("MomentumScalper",     "panteon_agents:MomentumScalper"),
    ("LiveAfterShock",      "panteon_agents:LiveAfterShock"),
    ("LiveCrashHunter",     "panteon_agents:LiveCrashHunter"),
    ("LiveRegimePullback",  "panteon_agents:LiveRegimePullback"),
    ("LiveMeanRev",         "panteon_agents:LiveMeanRev"),
    ("LiveTrendFollow",     "panteon_agents:LiveTrendFollow"),
    ("LiveVolCompress",     "panteon_agents:LiveVolCompress"),
    # RichardDennisTurtle — в v1 класс называется именно так
    ("RichardDennis",       "panteon_agents:RichardDennisTurtle"),
    ("LiveOIBreakout",      "panteon_agents:LiveOIBreakout"),
    ("ResearchValidatorAgent", "panteon_agents:ResearchValidatorAgent"),
    ("VolBreakoutHunter",   "panteon_agents:VolBreakoutHunter"),
    ("CarryFlowAgentV2",    "panteon_agents:CarryFlowAgentV2"),
    ("CandlePatternAgent",  "panteon_agents:CandlePatternAgent"),
    ("BullRotationAgent",   "panteon_agents:BullRotationAgent"),
    ("BearReliefFadeAgent", "panteon_agents:BearReliefFadeAgent"),
    ("NeutralRangeScalper", "panteon_agents:NeutralRangeScalper"),
    ("NeutralLiquiditySweep", "panteon_agents:NeutralLiquiditySweepAgent"),
    ("AnchorFlowMomentum",  "panteon_agents:AnchorFlowMomentumAgent"),
    ("CrashPanicShortAgent", "panteon_agents:CrashPanicShortAgent"),
    ("PlayerFunding",       "panteon_agents:PlayerFunding"),
]

# Опциональные агенты с тяжёлым импортом (numba/datasets/settings).
# Подключаются ТОЛЬКО если PANTEON_V2_LOAD_GENETICS=1 в env, или явно
# через register_optional_agents(). Иначе зависают на startup-е.
SHADOW_ONLY_V1_AGENT_LABELS: Tuple[str, ...] = (
    "CarryFlowAgentV2",
    "CandlePatternAgent",
)

FLASH_LEGACY_REAL_AGENT_LABELS: Tuple[str, ...] = (
    "CarryFlowAgentV2",
    "CandlePatternAgent",
)

FUTURES_REPLAY_SIGNAL_FIX_OVERRIDES: dict[str, dict[str, Any]] = {
    "CarryFlowAgentV2": {
        "CHECK_INT": 1,
        "HOLD": 48,
        "EMA_FAST": 12,
        "EMA_SLOW": 48,
        "RSI_OB": 58,
        "RSI_OS": 42,
        "EXTREME_EXT": 0.004,
        "ENTRY_COOLDOWN": 24,
    },
    "MomentumScalper": {
        "FUTURES_REPLAY_MODE": True,
        "CHECK_INT": 1,
        "EMA_F": 4,
        "EMA_M": 12,
        "EMA_S": 48,
        "VOL_WIN": 12,
        "VOL_MULT": 0.90,
        "MOM_MIN": 0.0015,
    },
    "LiveVolCompress": {
        "FUTURES_REPLAY_MODE": True,
        "CHECK_INT": 1,
        "HOLD": 48,
        "BBW_THRESH": 0.035,
        "MIN_BBW": 0.0008,
        "MOM_MIN": 0.0012,
        "ENTRY_COOLDOWN": 12,
    },
}

OPTIONAL_V1_AGENTS: List[Tuple[str, str]] = [
    ("GeneticsGenomeEnsemble", "agents_v2:GenomeEnsembleAgent"),
    ("GeneticsCore",        "crypto_genetics:GeneticsAgent"),
]

MANIFEST_GENETICS_AGENT_LABELS: Tuple[str, ...] = (
    "GeneticsBest",
    "GeneticsRiskTight",
    "GeneticsCrash",
)
MANIFEST_GENETICS_ENV = "PANTEON_V2_GENETICS_SPECIALISTS_MANIFEST"
REGIME_ADAPTIVE_BIAS_ENV = "PANTEON_V2_GENETICS_REGIME_ADAPTIVE_BIAS_MANIFEST"
GENETICS_CORE_GENOME_ENV = "PANTEON_V2_GENETICS_CORE_GENOME"
GENETICS_CORE_REQUIRE_REAL_ENV = "PANTEON_V2_REQUIRE_REAL_GENETICS_CORE"

OPTIONAL_SPECIAL_AGENTS: Tuple[str, ...] = (
    "GeneticsRegimeRouter",
    "GeneticsRegimeAdaptiveBias",
)

EXPERIMENTAL_FLASH_SPOT_QUALITY_SYMBOLS: Tuple[str, ...] = (
    "FIL/USDT",
    "AVAX/USDT",
    "SOL/USDT",
    "APT/USDT",
    "NEAR/USDT",
    "BNB/USDT",
    "ETC/USDT",
)

EXPERIMENTAL_FLASH_ULTIMA_QUALITY_LONG_SYMBOLS: Tuple[str, ...] = (
    "FIL/USDT",
    "APT/USDT",
    "ICP/USDT",
    "LTC/USDT",
)

EXPERIMENTAL_FLASH_ULTIMA_MAJOR_SHORT_SYMBOLS: Tuple[str, ...] = (
    "BTC/USDT",
    "DOGE/USDT",
    "ETH/USDT",
)

EXPERIMENTAL_FLASH_ULTIMA_OI_SPOT_SYMBOLS: Tuple[str, ...] = (
    "ADA/USDT",
    "FIL/USDT",
    "XLM/USDT",
)

EXPERIMENTAL_FLASH_AGENT_SPECS: Tuple[
    Tuple[
        str,
        str,
        Tuple[Action, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        Tuple[str, ...],
        dict[str, Any],
    ],
    ...
] = (
    (
        "MomentumScalperShortOnly",
        "MomentumScalper",
        (Action.FUT_SHORT_HALF, Action.FUT_SHORT_FULL, Action.FUT_CLOSE_ALL),
        (),
        (),
        (),
        {},
    ),
    (
        "MomentumScalperSpotQuality",
        "MomentumScalper",
        (Action.SPOT_BUY_HALF, Action.SPOT_BUY_FULL, Action.SPOT_SELL_ALL),
        EXPERIMENTAL_FLASH_SPOT_QUALITY_SYMBOLS,
        (),
        (),
        {},
    ),
    (
        "MomentumScalperUltimaQualityLongs",
        "MomentumScalper",
        (Action.FUT_LONG_HALF, Action.FUT_LONG_FULL),
        EXPERIMENTAL_FLASH_ULTIMA_QUALITY_LONG_SYMBOLS,
        (),
        (),
        {},
    ),
    (
        "MomentumScalperUltimaMajorShorts",
        "MomentumScalper",
        (Action.FUT_SHORT_HALF, Action.FUT_SHORT_FULL),
        EXPERIMENTAL_FLASH_ULTIMA_MAJOR_SHORT_SYMBOLS,
        (),
        ("bearish", "crash"),
        {},
    ),
    (
        "LiveOIBreakoutUltimaSpot",
        "LiveOIBreakout",
        (Action.SPOT_BUY_HALF, Action.SPOT_BUY_FULL, Action.SPOT_SELL_ALL),
        EXPERIMENTAL_FLASH_ULTIMA_OI_SPOT_SYMBOLS,
        (),
        (),
        {},
    ),
    (
        "VolBreakoutSpotOnly",
        "VolBreakoutHunter",
        (Action.SPOT_BUY_HALF, Action.SPOT_BUY_FULL, Action.SPOT_SELL_ALL),
        (),
        (),
        (),
        {},
    ),
    (
        "MomentumScalperShortCrashOnly",
        "MomentumScalper",
        (Action.FUT_SHORT_HALF, Action.FUT_SHORT_FULL),
        (),
        (),
        ("crash",),
        {},
    ),
    (
        "MomentumScalperShortBearOnly",
        "MomentumScalper",
        (Action.FUT_SHORT_HALF, Action.FUT_SHORT_FULL),
        (),
        (),
        ("bearish",),
        {},
    ),
    (
        "MomentumScalperSpotPullbackOnly",
        "MomentumScalper",
        (Action.SPOT_BUY_HALF, Action.SPOT_BUY_FULL),
        (),
        (),
        ("neutral", "bullish"),
        {},
    ),
    (
        "ResearchValidatorNeutralOnly",
        "ResearchValidatorAgent",
        (),
        (),
        (),
        ("neutral",),
        {},
    ),
    (
        "FundingArbBearOnly",
        "FundingArb",
        (),
        (),
        (),
        ("bearish",),
        {},
    ),
    (
        "CrashPanicCrashOnly",
        "CrashPanicShortAgent",
        (),
        (),
        (),
        ("crash",),
        {},
    ),
    (
        "CrashHunterStrict",
        "LiveCrashHunter",
        (Action.FUT_SHORT_HALF, Action.FUT_SHORT_FULL, Action.FUT_CLOSE_ALL),
        (),
        (),
        ("crash",),
        {
            "min_regime_confidence": 0.70,
            "max_lookback_return_pct_by_bars": {12: -2.5},
        },
    ),
    (
        "VolBreakoutFundingAware",
        "VolBreakoutHunter",
        (Action.FUT_SHORT_HALF, Action.FUT_SHORT_FULL, Action.FUT_CLOSE_ALL),
        (),
        (),
        ("bearish", "crash"),
        {"funding_cost_aligned_opens": True},
    ),
    (
        "AfterShockRegimeOnly",
        "LiveAfterShock",
        (Action.SPOT_BUY_HALF, Action.SPOT_BUY_FULL, Action.SPOT_SELL_ALL),
        (),
        (),
        ("crash",),
        {
            "min_regime_confidence": 0.70,
            "max_lookback_return_pct_by_bars": {12: -2.5, 24: -4.0},
        },
    ),
    (
        "LiveTrendFollowBullOnly",
        "LiveTrendFollow",
        (Action.SPOT_BUY_HALF, Action.SPOT_BUY_FULL, Action.SPOT_SELL_ALL),
        (),
        (),
        ("bullish",),
        {"min_lookback_return_pct_by_bars": {24: 0.5}},
    ),
    (
        "LiveMeanRevNeutralOnly",
        "LiveMeanRev",
        (),
        (),
        (),
        ("neutral",),
        {},
    ),
)


@dataclass
class ActionFilterAgent:
    """Wrap an agent under a new label and expose only a narrow action slice."""

    prefers_full_market_snapshot: ClassVar[bool] = True
    label: str
    base_agent: Any
    allowed_actions: Tuple[Action, ...] = ()
    allowed_symbols: Tuple[str, ...] = ()
    denied_symbols: Tuple[str, ...] = ()
    allowed_regimes: Tuple[str, ...] = ()
    min_regime_confidence: Optional[float] = None
    funding_cost_aligned_opens: bool = False
    min_lookback_return_pct_by_bars: Optional[dict[int, float]] = None
    max_lookback_return_pct_by_bars: Optional[dict[int, float]] = None
    last_signal_diagnostics: dict[str, dict[str, Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.allowed_actions = tuple(
            action if isinstance(action, Action) else Action(int(action))
            for action in self.allowed_actions
        )
        self.allowed_symbols = tuple(
            _normalize_symbol(symbol) for symbol in self.allowed_symbols
        )
        self.denied_symbols = tuple(
            _normalize_symbol(symbol) for symbol in self.denied_symbols
        )
        self.allowed_regimes = tuple(
            _normalize_regime_label(regime)
            for regime in self.allowed_regimes
        )
        if self.min_regime_confidence is not None:
            self.min_regime_confidence = float(self.min_regime_confidence)
        self.min_lookback_return_pct_by_bars = _normalize_lookback_gate(
            self.min_lookback_return_pct_by_bars
        )
        self.max_lookback_return_pct_by_bars = _normalize_lookback_gate(
            self.max_lookback_return_pct_by_bars
        )

    def clone_for_shadow(self) -> Optional["ActionFilterAgent"]:
        base_clone = _clone_agent_for_wrapper(self.base_agent)
        if base_clone is None:
            return None
        return ActionFilterAgent(
            label=self.label,
            base_agent=base_clone,
            allowed_actions=self.allowed_actions,
            allowed_symbols=self.allowed_symbols,
            denied_symbols=self.denied_symbols,
            allowed_regimes=self.allowed_regimes,
            min_regime_confidence=self.min_regime_confidence,
            funding_cost_aligned_opens=self.funding_cost_aligned_opens,
            min_lookback_return_pct_by_bars=self.min_lookback_return_pct_by_bars,
            max_lookback_return_pct_by_bars=self.max_lookback_return_pct_by_bars,
        )

    def act(self, market: MarketSnapshot) -> dict[str, Action]:
        self.last_signal_diagnostics = {}
        try:
            raw = self.base_agent.act(market)
        except Exception:
            return {}
        if not isinstance(raw, dict):
            return {}

        diagnostics = _normalize_agent_diagnostics(
            getattr(self.base_agent, "last_signal_diagnostics", {}),
            market,
        )
        allowed_actions = frozenset(self.allowed_actions)
        allowed_symbols = frozenset(self.allowed_symbols)
        denied_symbols = frozenset(self.denied_symbols)
        allowed_regimes = frozenset(self.allowed_regimes)
        out: dict[str, Action] = {}
        for raw_symbol, raw_action in raw.items():
            symbol = _normalize_symbol(raw_symbol)
            if symbol not in market.prices:
                continue
            regime_label = _normalize_regime_label(market.regime_for_symbol(symbol))
            action = _coerce_action(raw_action)
            diag = dict(diagnostics.get(symbol, {}))
            diag["wrapper_label"] = self.label
            diag["wrapper_base_label"] = str(getattr(self.base_agent, "label", "") or "")
            diag["wrapper_action"] = None if action is None else action.name
            if action is None or action.is_hold:
                diag["wrapper_reason"] = "invalid_action" if action is None else "base_hold"
                diagnostics[symbol] = diag
                continue
            if allowed_actions and action not in allowed_actions:
                diag["wrapper_reason"] = "action_not_allowed"
                diagnostics[symbol] = diag
                continue
            if allowed_symbols and symbol not in allowed_symbols:
                diag["wrapper_reason"] = "symbol_not_allowed"
                diagnostics[symbol] = diag
                continue
            if symbol in denied_symbols:
                diag["wrapper_reason"] = "symbol_denied"
                diagnostics[symbol] = diag
                continue
            if allowed_regimes and regime_label not in allowed_regimes:
                out[symbol] = Action.HOLD
                diag["wrapper_reason"] = "regime_not_allowed"
                diagnostics[symbol] = diag
                continue
            if action.is_open and not self._open_context_allows(market, symbol, action):
                out[symbol] = Action.HOLD
                diag["wrapper_reason"] = "open_context_blocked"
                diagnostics[symbol] = diag
                continue
            diag["wrapper_reason"] = "passed"
            diagnostics[symbol] = diag
            out[symbol] = action
        self.last_signal_diagnostics = diagnostics
        return out

    def _open_context_allows(
        self,
        market: MarketSnapshot,
        symbol: str,
        action: Action,
    ) -> bool:
        if self.min_regime_confidence is not None:
            try:
                confidence = float(market.regime_confidence)
            except (TypeError, ValueError):
                return False
            if confidence < self.min_regime_confidence:
                return False

        if self.funding_cost_aligned_opens and action in (
            Action.FUT_LONG_HALF,
            Action.FUT_LONG_FULL,
            Action.FUT_SHORT_HALF,
            Action.FUT_SHORT_FULL,
        ):
            funding = _lookup_numeric_symbol_value(market.funding, symbol)
            if funding is None:
                return False
            if action.is_long_open and funding > 0.0:
                return False
            if action.is_short_open and funding < 0.0:
                return False

        if self.min_lookback_return_pct_by_bars:
            for bars, threshold in self.min_lookback_return_pct_by_bars.items():
                value = _lookup_lookback_return_pct(market, symbol, bars)
                if value is None or value < threshold:
                    return False
        if self.max_lookback_return_pct_by_bars:
            for bars, threshold in self.max_lookback_return_pct_by_bars.items():
                value = _lookup_lookback_return_pct(market, symbol, bars)
                if value is None or value > threshold:
                    return False
        return True

    def sync_from_execution_results(self, results) -> None:
        sync = getattr(self.base_agent, "sync_from_execution_results", None)
        if callable(sync):
            sync(results)


def _normalize_symbol(symbol: Any) -> str:
    return str(symbol or "").strip().upper()


def _normalize_regime_label(regime: Any) -> str:
    label = getattr(regime, "label", None)
    if label is None:
        label = getattr(regime, "name", None)
    if label is None:
        label = regime
    return Regime.from_string(str(label or "")).label


def _normalize_lookback_gate(
    gate: Optional[dict[int, float]],
) -> dict[int, float]:
    if not gate:
        return {}
    normalized: dict[int, float] = {}
    for raw_bars, raw_threshold in gate.items():
        try:
            bars = int(raw_bars)
            threshold = float(raw_threshold)
        except (TypeError, ValueError):
            continue
        if bars > 0:
            normalized[bars] = threshold
    return normalized


def _symbol_candidates(symbol: str) -> tuple[str, ...]:
    normalized = _normalize_symbol(symbol)
    base = normalized.split("/", 1)[0]
    compact = normalized.replace("/", "")
    return tuple(dict.fromkeys((normalized, base, compact)))


def _lookup_symbol_value(mapping: Any, symbol: str) -> Any:
    if not isinstance(mapping, Mapping):
        return None
    for key in _symbol_candidates(symbol):
        if key in mapping:
            return mapping[key]
    return None


def _normalize_agent_diagnostics(
    raw: Any,
    market: MarketSnapshot,
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for raw_symbol in market.prices:
        symbol = _normalize_symbol(raw_symbol)
        payload = _lookup_symbol_value(raw, symbol)
        if payload is None:
            continue
        if isinstance(payload, Mapping):
            out[symbol] = dict(payload)
        else:
            out[symbol] = {"value": payload}
    return out


def _lookup_numeric_symbol_value(mapping: Any, symbol: str) -> Optional[float]:
    value = _lookup_symbol_value(mapping, symbol)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lookup_lookback_return_pct(
    market: MarketSnapshot,
    symbol: str,
    bars: int,
) -> Optional[float]:
    returns_by_bar = _lookup_symbol_value(market.lookback_returns_pct, symbol)
    if not isinstance(returns_by_bar, dict):
        return None
    for key in (bars, str(bars)):
        if key in returns_by_bar:
            try:
                return float(returns_by_bar[key])
            except (TypeError, ValueError):
                return None
    return None


def _coerce_action(value: Any) -> Optional[Action]:
    if isinstance(value, Action):
        return value
    try:
        return Action(int(value))
    except (TypeError, ValueError):
        return None


def _live_oi_breakout_futures_action_mapper(value: Any) -> Optional[Action]:
    action = _coerce_action(value)
    if action == Action.SPOT_BUY_HALF:
        return Action.FUT_LONG_HALF
    if action == Action.SPOT_BUY_FULL:
        return Action.FUT_LONG_FULL
    if action == Action.SPOT_SELL_ALL:
        return Action.FUT_CLOSE_ALL
    if action == Action.FUT_SHORT_HALF:
        return Action.FUT_SHORT_FULL
    return action


def _clone_agent_for_wrapper(agent: Any) -> Optional[Any]:
    clone = getattr(agent, "clone_for_shadow", None)
    if callable(clone):
        try:
            cloned = clone()
        except Exception as exc:
            log.warning(
                "Failed to clone %s via clone_for_shadow: %s",
                getattr(agent, "label", type(agent).__name__),
                type(exc).__name__,
            )
            return None
        if cloned is None:
            log.warning(
                "Failed to clone %s via clone_for_shadow: returned None",
                getattr(agent, "label", type(agent).__name__),
            )
        return cloned
    try:
        return copy.deepcopy(agent)
    except Exception as exc:
        log.warning(
            "Failed to deepcopy %s for wrapper: %s",
            getattr(agent, "label", type(agent).__name__),
            type(exc).__name__,
        )
        return None


def experimental_flash_agent_labels() -> List[str]:
    return [
        label
        for label, _base, _actions, _allowed, _denied, _regimes, _kwargs
        in EXPERIMENTAL_FLASH_AGENT_SPECS
    ]


def legacy_flash_real_agent_labels() -> List[str]:
    return list(FLASH_LEGACY_REAL_AGENT_LABELS)


def register_experimental_flash_agents(
    registry: AgentRegistry,
    *,
    skip_missing: bool = True,
) -> List[str]:
    registered: List[str] = []
    for (
        label,
        base_label,
        actions,
        allowed_symbols,
        denied_symbols,
        allowed_regimes,
        filter_kwargs,
    ) in EXPERIMENTAL_FLASH_AGENT_SPECS:
        base_agent = registry.get(base_label)
        if base_agent is None:
            if skip_missing:
                log.warning(
                    "Skipping experimental Flash agent %s: base %s missing",
                    label,
                    base_label,
                )
                continue
            raise ValueError(f"base agent {base_label!r} is not registered")
        base_clone = _clone_agent_for_wrapper(base_agent)
        if base_clone is None:
            if skip_missing:
                log.warning(
                    "Skipping experimental Flash agent %s: base %s clone failed",
                    label,
                    base_label,
                )
                continue
            raise ValueError(
                f"base agent {base_label!r} could not be cloned for wrapper {label!r}"
            )
        registry.register(
            ActionFilterAgent(
                label=label,
                base_agent=base_clone,
                allowed_actions=actions,
                allowed_symbols=allowed_symbols,
                denied_symbols=denied_symbols,
                allowed_regimes=allowed_regimes,
                **dict(filter_kwargs),
            ),
            replace=True,
        )
        registered.append(label)
    return registered


def promote_legacy_flash_real_agents(
    registry: AgentRegistry,
    *,
    labels: Optional[Sequence[str]] = None,
    skip_missing: bool = True,
) -> List[str]:
    if labels is None:
        selected = FLASH_LEGACY_REAL_AGENT_LABELS
    else:
        raw_labels: Sequence[str]
        raw_labels = (labels,) if isinstance(labels, str) else labels
        selected = tuple(dict.fromkeys(
            str(label).strip() for label in raw_labels if str(label).strip()
        ))
    registered: List[str] = []
    for label in selected:
        agent = registry.get(label)
        if agent is None:
            if skip_missing:
                log.warning(
                    "Skipping legacy Flash real agent %s: base label missing",
                    label,
                )
                continue
            raise ValueError(f"legacy Flash real agent {label!r} is not registered")
        setattr(agent, "shadow_only", False)
        setattr(agent, "paper_trading_eligible", True)
        setattr(agent, "live_trading_eligible", True)
        setattr(agent, "flash_legacy_real_enabled", True)
        registered.append(label)
    return registered


def _env_flag(name: str, default: bool = False) -> bool:
    import os as _os

    raw = _os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_paths():
    """Добавить нужные пути в sys.path (если ещё не там)."""
    project_root = Path(__file__).resolve().parents[3]
    candidates = [
        project_root / "src" / "panteon_runtime",
        project_root / "Genetics_DL_Agents",
        project_root / "src",
    ]
    for p in candidates:
        sp = str(p)
        if p.is_dir() and sp not in sys.path:
            sys.path.insert(0, sp)


def _default_genetics_results_root() -> Path:
    return Path(__file__).resolve().parents[3] / "Results" / "neiro_genetics"


def _infer_genetics_results_root(path: str | Path) -> Path:
    raw = Path(path)
    resolved = raw if raw.is_absolute() else Path.cwd() / raw
    resolved = resolved.resolve()
    for parent in (resolved.parent, *resolved.parents):
        if (
            parent.name.lower() == "neiro_genetics"
            and parent.parent.name.lower() == "results"
        ):
            return parent
    return _default_genetics_results_root()


def _resolve_router_manifest_path(path: str | Path, *, results_root: Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        resolved = raw
    else:
        cwd_candidate = (Path.cwd() / raw).resolve()
        if cwd_candidate.exists():
            resolved = cwd_candidate
        else:
            resolved = results_root / raw
    resolved = resolved.resolve()
    try:
        resolved.relative_to(results_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"genetics router path must be under {results_root}: {resolved}"
        ) from exc
    return resolved


def _manifest_genetics_configured() -> bool:
    return bool(os.environ.get(MANIFEST_GENETICS_ENV))


def _resolve_specialists_manifest_path() -> tuple[Path, Path] | None:
    raw = os.environ.get(MANIFEST_GENETICS_ENV)
    if not raw:
        return None
    root = _infer_genetics_results_root(raw).resolve()
    return _resolve_router_manifest_path(raw, results_root=root), root


def _router_promotion_eligible(selection: dict[str, Any]) -> bool:
    if bool(selection.get("selected_is_baseline", False)):
        return False
    validation = selection.get("validation") or {}
    baseline = selection.get("baseline_validation") or {}
    try:
        validation_mean = float(validation.get("mean_ret", 0.0))
        baseline_mean = float(baseline.get("mean_ret", 0.0))
        validation_min = float(validation.get("min_ret", 0.0))
        baseline_min = float(baseline.get("min_ret", 0.0))
        validation_positive = float(validation.get("positive_period_pct", 0.0))
        baseline_positive = float(baseline.get("positive_period_pct", 0.0))
    except (TypeError, ValueError):
        return False
    return (
        validation_mean > baseline_mean
        and validation_min >= baseline_min
        and validation_positive >= baseline_positive
    )


def _router_live_trading_eligible(selection: dict[str, Any]) -> bool:
    return (
        _router_promotion_eligible(selection)
        and bool(selection.get("live_trading_eligible", False))
        and _router_paper_gate_confirmed(selection)
    )


def _router_paper_gate_confirmed(selection: dict[str, Any]) -> bool:
    if not bool(selection.get("paper_trading_eligible", False)):
        return False
    paper_gate = selection.get("paper_gate")
    if not isinstance(paper_gate, dict):
        return False
    if not bool(paper_gate.get("paper_trading_eligible", False)):
        return False
    if paper_gate.get("paper_failures"):
        return False
    paper_reports = paper_gate.get("paper_reports")
    if not isinstance(paper_reports, list) or len(paper_reports) < 2:
        return False
    return all(
        isinstance(report, dict) and bool(report.get("accepted", False))
        for report in paper_reports
    )


def build_genetics_regime_router_adapter(
    manifest_path: str | Path,
    genetics_cls: Any,
    *,
    results_root: Optional[str | Path] = None,
    label: str = "GeneticsRegimeRouter",
    portfolio_value_fn: Optional[callable] = None,
    min_regime_confidence: float = 0.70,
    expected_genome_size: Optional[int] = None,
) -> GeneticsRegimeRouterV2AgentAdapter:
    import numpy as np

    root = Path(results_root).resolve() if results_root is not None else _default_genetics_results_root().resolve()
    manifest = _resolve_router_manifest_path(manifest_path, results_root=root)
    selection = json.loads(manifest.read_text(encoding="utf-8"))
    selected_raw = selection.get("selected_regime_map")
    baseline_raw = selection.get("baseline_regime_map")
    if not isinstance(selected_raw, dict) or not isinstance(baseline_raw, dict):
        raise ValueError("genetics router manifest must contain selected_regime_map and baseline_regime_map")

    selected_map = {
        str(regime): _resolve_router_manifest_path(path, results_root=root)
        for regime, path in selected_raw.items()
    }
    baseline_map = {
        str(regime): _resolve_router_manifest_path(path, results_root=root)
        for regime, path in baseline_raw.items()
    }
    baseline_paths = set(baseline_map.values())
    if len(baseline_paths) != 1:
        raise ValueError("genetics router baseline_regime_map must point to one baseline genome")
    baseline_path = next(iter(baseline_paths))

    agent_cache: dict[Path, Any] = {}

    def load_agent(path: Path) -> Any:
        if path not in agent_cache:
            genome = np.load(path).astype(np.float32)
            if expected_genome_size is not None and genome.size != int(expected_genome_size):
                raise ValueError(
                    f"genetics router genome size mismatch for {path}: "
                    f"{genome.size} != {int(expected_genome_size)}"
                )
            agent_cache[path] = genetics_cls(genome=genome)
        return agent_cache[path]

    adapter = GeneticsRegimeRouterV2AgentAdapter(
        label=label,
        baseline_agent=load_agent(baseline_path),
        regime_agents={
            regime: load_agent(path)
            for regime, path in selected_map.items()
        },
        portfolio_value_fn=portfolio_value_fn,
        min_regime_confidence=min_regime_confidence,
    )
    adapter.selection_manifest_path = str(manifest)
    adapter.selected_regime_map = {
        regime: str(path)
        for regime, path in selected_map.items()
    }
    adapter.baseline_regime_map = {
        regime: str(path)
        for regime, path in baseline_map.items()
    }
    adapter.promotion_eligible = _router_promotion_eligible(selection)
    adapter.paper_trading_eligible = bool(
        selection.get("paper_trading_eligible", False)
    ) and adapter.promotion_eligible
    adapter.live_trading_eligible = _router_live_trading_eligible(selection)
    adapter.shadow_only = not adapter.live_trading_eligible
    return adapter


def _regime_adaptive_bias_from_manifest(payload: dict[str, Any]) -> dict[str, float]:
    raw = payload.get("regime_open_bias")
    if raw is None and isinstance(payload.get("regime_adaptive_output_bias"), dict):
        raw = payload["regime_adaptive_output_bias"].get("regime_open_bias")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("regime adaptive bias manifest must contain regime_open_bias")
    return {
        str(key).strip().lower(): float(value)
        for key, value in raw.items()
        if str(key).strip()
    }


def _regime_adaptive_source_genome_from_manifest(payload: dict[str, Any]) -> str:
    for key in ("source_genome", "source_path", "source_genome_path"):
        value = payload.get(key)
        if value:
            return str(value)
    selected = str(payload.get("selected_genome") or "")
    if selected:
        return selected.split("::", 1)[0]
    raise ValueError("regime adaptive bias manifest must contain source_genome")


def _genome_position_state_features_enabled(genome_path: Path) -> Optional[bool]:
    metadata_candidates = [
        genome_path.with_name(f"{genome_path.stem}_meta.json"),
        genome_path.parent / "best_genome_meta.json",
    ]
    for meta_path in metadata_candidates:
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception:
            continue
        if "position_state_features_enabled" in payload:
            return bool(payload["position_state_features_enabled"])
    return None


def build_genetics_regime_adaptive_bias_adapter(
    manifest_path: str | Path,
    genetics_cls: Any,
    *,
    results_root: Optional[str | Path] = None,
    label: str = "GeneticsRegimeAdaptiveBias",
    portfolio_value_fn: Optional[callable] = None,
    expected_genome_size: Optional[int] = None,
) -> GeneticsV2AgentAdapter:
    import numpy as np

    root = Path(results_root).resolve() if results_root is not None else _default_genetics_results_root().resolve()
    manifest = _resolve_router_manifest_path(manifest_path, results_root=root)
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    regime_open_bias = _regime_adaptive_bias_from_manifest(payload)
    genome_path = _resolve_router_manifest_path(
        _regime_adaptive_source_genome_from_manifest(payload),
        results_root=root,
    )
    genome = np.load(genome_path).astype(np.float32).ravel()
    if expected_genome_size is not None and genome.size != int(expected_genome_size):
        raise ValueError(
            f"regime adaptive genetics genome size mismatch for {genome_path}: "
            f"{genome.size} != {int(expected_genome_size)}"
        )
    agent = genetics_cls(genome=genome)
    configure = getattr(agent, "configure_regime_adaptive_output_bias", None)
    if not callable(configure):
        raise ValueError("GeneticsAgent does not support regime_adaptive_output_bias")
    configure(regime_open_bias, enabled=True)
    position_state_features_enabled = _genome_position_state_features_enabled(genome_path)
    if position_state_features_enabled is not None:
        setattr(
            agent,
            "_position_state_features_enabled",
            bool(position_state_features_enabled),
        )

    adapter = GeneticsV2AgentAdapter(
        label=label,
        v1_agent=agent,
        portfolio_value_fn=portfolio_value_fn,
    )
    adapter.selection_manifest_path = str(manifest)
    adapter.source_genome_path = str(genome_path)
    adapter.source_position_state_features_enabled = position_state_features_enabled
    adapter.regime_open_bias = dict(regime_open_bias)
    adapter.promotion_eligible = bool(payload.get("promotion_eligible", False))
    adapter.paper_trading_eligible = bool(payload.get("paper_trading_eligible", False)) and adapter.promotion_eligible
    adapter.live_trading_eligible = False
    adapter.shadow_only = True
    return adapter


def _filter_manifest_genetics_labels(
    labels: Optional[Sequence[str]],
) -> List[str]:
    allowed_manifest = set(MANIFEST_GENETICS_AGENT_LABELS)
    if labels is None:
        return list(MANIFEST_GENETICS_AGENT_LABELS)
    return [str(label) for label in labels if str(label) in allowed_manifest]


def _register_manifest_genetics_agents(
    registry: AgentRegistry,
    labels: Optional[Sequence[str]],
    *,
    portfolio_value_fn: Optional[callable] = None,
    skip_on_error: bool = True,
) -> List[str]:
    resolved_manifest = _resolve_specialists_manifest_path()
    selected_labels = _filter_manifest_genetics_labels(labels)
    if resolved_manifest is None or not selected_labels:
        return []

    import numpy as np

    manifest, root = resolved_manifest
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    raw_map = payload.get("specialist_genome_map")
    if not isinstance(raw_map, dict):
        raise ValueError("genetics specialists manifest must contain specialist_genome_map")

    module = __import__("crypto_genetics", fromlist=["GeneticsAgent"])
    genetics_cls = getattr(module, "GeneticsAgent")
    expected_genome_size = getattr(module, "GENOME_SIZE", None)
    paper_eligible = bool(payload.get("paper_trading_eligible", False))
    promotion_eligible = bool(payload.get("promotion_eligible", False))

    registered: List[str] = []
    for label in selected_labels:
        try:
            raw_path = raw_map.get(label)
            if raw_path is None:
                raise ValueError(f"{label} missing from specialist_genome_map")
            genome_path = _resolve_router_manifest_path(raw_path, results_root=root)
            genome = np.load(genome_path).astype(np.float32).ravel()
            if expected_genome_size is not None and genome.size != int(expected_genome_size):
                raise ValueError(
                    f"genetics specialist genome size mismatch for {genome_path}: "
                    f"{genome.size} != {int(expected_genome_size)}"
                )
            adapter = GeneticsV2AgentAdapter(
                label=label,
                v1_agent=genetics_cls(genome=genome),
                portfolio_value_fn=portfolio_value_fn,
            )
            adapter.selection_manifest_path = str(manifest)
            adapter.source_genome_path = str(genome_path)
            adapter.promotion_eligible = promotion_eligible
            adapter.paper_trading_eligible = paper_eligible
            adapter.live_trading_eligible = False
            adapter.shadow_only = True
            registry.register(adapter, replace=True)
            registered.append(label)
        except Exception as exc:
            if skip_on_error:
                log.warning("Skipping %s from specialists manifest: %s", label, type(exc).__name__)
                continue
            raise
    return registered


def _resolve_genetics_core_genome_path() -> Path | None:
    raw = os.environ.get(GENETICS_CORE_GENOME_ENV, "").strip()
    if raw:
        path = Path(raw)
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()
    default_path = (
        Path(__file__).resolve().parents[3]
        / "Genetics_DL_Agents"
        / "Agents"
        / "genetics"
        / "best_genome.npy"
    )
    return default_path.resolve() if default_path.exists() else None


def _register_genetics_core_agent(
    registry: AgentRegistry,
    *,
    module_path: str,
    class_name: str,
    portfolio_value_fn: Optional[callable] = None,
) -> str:
    import numpy as np

    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    genome_path = _resolve_genetics_core_genome_path()
    if genome_path is None:
        raise FileNotFoundError(
            "GeneticsCore requires a real genome source; set "
            f"{GENETICS_CORE_GENOME_ENV} or provide "
            "Genetics_DL_Agents/Agents/genetics/best_genome.npy"
        )
    source = "explicit_core_genome"
    source_meta_path: Path | None = None
    if not genome_path.exists():
        raise FileNotFoundError(
            f"{GENETICS_CORE_GENOME_ENV} points to missing genome: {genome_path}"
        )
    genome = np.load(genome_path).astype(np.float32).ravel()
    expected_genome_size = getattr(module, "GENOME_SIZE", None)
    if expected_genome_size is not None and genome.size != int(expected_genome_size):
        raise ValueError(
            f"GeneticsCore genome size mismatch for {genome_path}: "
            f"{genome.size} != {int(expected_genome_size)}"
        )
    instance = cls(genome=genome)
    for candidate in (
        genome_path.with_name(f"{genome_path.stem}_meta.json"),
        genome_path.parent / "best_genome_meta.json",
    ):
        if candidate.exists():
            source_meta_path = candidate.resolve()
            break

    adapter = GeneticsV2AgentAdapter(
        label="GeneticsCore",
        v1_agent=instance,
        portfolio_value_fn=portfolio_value_fn,
    )
    adapter.genetics_signal_source = source
    adapter.source_genome_path = str(genome_path)
    setattr(instance, "source_genome_path", str(genome_path))
    if source_meta_path is not None:
        adapter.source_meta_path = str(source_meta_path)
    registry.register(adapter, replace=True)
    return "GeneticsCore"


def _register_from_list(
    registry: AgentRegistry,
    items: List[Tuple[str, str]],
    *,
    portfolio_value_fn: Optional[callable] = None,
    skip_on_error: bool = True,
    futures_replay_signal_fixes_enabled: bool = False,
) -> List[str]:
    registered: List[str] = []
    for label, dotted in items:
        try:
            module_path, class_name = dotted.split(":")
            if label == "GeneticsCore" and module_path == "crypto_genetics":
                registered.append(_register_genetics_core_agent(
                    registry,
                    module_path=module_path,
                    class_name=class_name,
                    portfolio_value_fn=portfolio_value_fn,
                ))
                continue
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            instance = cls()
            if futures_replay_signal_fixes_enabled:
                _apply_futures_replay_signal_fixes(label, instance)
            adapter_cls = (
                GeneticsV2AgentAdapter
                if module_path == "crypto_genetics" or label.startswith("Genetics")
                else V1AgentAdapter
            )
            adapter_kwargs = {
                "label": label,
                "v1_agent": instance,
                "portfolio_value_fn": portfolio_value_fn,
            }
            if (
                adapter_cls is V1AgentAdapter
                and futures_replay_signal_fixes_enabled
                and label == "LiveOIBreakout"
            ):
                adapter_kwargs["action_mapper"] = _live_oi_breakout_futures_action_mapper
            if adapter_cls is GeneticsV2AgentAdapter and label == "GeneticsGenomeEnsemble":
                adapter_kwargs["allowed_open_regimes"] = ("bearish", "crash")
            adapter = adapter_cls(**adapter_kwargs)
            if label in SHADOW_ONLY_V1_AGENT_LABELS:
                setattr(adapter, "shadow_only", True)
                setattr(adapter, "paper_trading_eligible", True)
                setattr(adapter, "live_trading_eligible", False)
            registry.register(adapter, replace=True)
            registered.append(label)
        except Exception as exc:
            if skip_on_error:
                log.warning("Skipping %s (%s): %s",
                            label, dotted, type(exc).__name__)
                continue
            raise ImportError(
                f"Failed to register {label} from {dotted}: {exc}"
            ) from exc
    return registered


def _filter_optional_special_labels(
    labels: Optional[Sequence[str]],
) -> List[str]:
    if labels is None:
        return list(OPTIONAL_SPECIAL_AGENTS)
    allowed = {str(label) for label in labels}
    return [label for label in OPTIONAL_SPECIAL_AGENTS if label in allowed]


def _register_special_optional_agents(
    registry: AgentRegistry,
    labels: Optional[Sequence[str]],
    *,
    portfolio_value_fn: Optional[callable] = None,
    skip_on_error: bool = True,
) -> List[str]:
    registered: List[str] = []
    for label in _filter_optional_special_labels(labels):
        try:
            module = __import__("crypto_genetics", fromlist=["GeneticsAgent"])
            if label == "GeneticsRegimeRouter":
                manifest = os.environ.get("PANTEON_V2_GENETICS_ROUTER_MANIFEST")
                if not manifest and not _env_flag("PANTEON_V2_LOAD_GENETICS_ROUTER", False):
                    continue
                if not manifest:
                    raise ValueError("PANTEON_V2_GENETICS_ROUTER_MANIFEST is required")
                adapter = build_genetics_regime_router_adapter(
                    manifest,
                    getattr(module, "GeneticsAgent"),
                    results_root=_infer_genetics_results_root(manifest),
                    portfolio_value_fn=portfolio_value_fn,
                    expected_genome_size=getattr(module, "GENOME_SIZE", None),
                )
            elif label == "GeneticsRegimeAdaptiveBias":
                manifest = os.environ.get(REGIME_ADAPTIVE_BIAS_ENV)
                if not manifest:
                    continue
                adapter = build_genetics_regime_adaptive_bias_adapter(
                    manifest,
                    getattr(module, "GeneticsAgent"),
                    results_root=_infer_genetics_results_root(manifest),
                    portfolio_value_fn=portfolio_value_fn,
                    expected_genome_size=getattr(module, "GENOME_SIZE", None),
                )
            else:
                continue
            registry.register(adapter, replace=True)
            registered.append(label)
        except Exception as exc:
            if skip_on_error:
                log.warning("Skipping %s: %s", label, type(exc).__name__)
                continue
            raise ImportError(f"Failed to register {label}: {exc}") from exc
    return registered


def _apply_futures_replay_signal_fixes(label: str, instance: Any) -> bool:
    overrides = FUTURES_REPLAY_SIGNAL_FIX_OVERRIDES.get(str(label or ""))
    if not overrides:
        return False
    for attr, value in overrides.items():
        try:
            setattr(instance, attr, value)
        except Exception:
            return False
    return True


def register_all_v1_agents(
    registry: AgentRegistry,
    *,
    portfolio_value_fn: Optional[callable] = None,
    skip_on_error: bool = True,
    include_optional: Optional[bool] = None,
    optional_agent_labels: Optional[Sequence[str]] = None,
    futures_replay_signal_fixes_enabled: bool = False,
) -> List[str]:
    """Регистрирует базовые v1-агенты в registry.

    Опциональные тяжёлые агенты (Genetics) подключаются только если:
      • include_optional=True, ИЛИ
      • переменная окружения PANTEON_V2_LOAD_GENETICS=1 установлена.
    Это сделано потому что Genetics при импорте поднимают
    numba/settings/datasets и могут зависнуть на старте.
    """
    _ensure_paths()
    if include_optional is None:
        include_optional = _env_flag("PANTEON_V2_LOAD_GENETICS", False)
    registered = _register_from_list(
        registry, KNOWN_V1_AGENTS,
        portfolio_value_fn=portfolio_value_fn,
        skip_on_error=skip_on_error,
        futures_replay_signal_fixes_enabled=futures_replay_signal_fixes_enabled,
    )
    if include_optional:
        log.info("Loading optional agents (Genetics) — may take a while…")
        optional_items = _filter_optional_agent_items(optional_agent_labels)
        registered += _register_from_list(
            registry, optional_items,
            portfolio_value_fn=portfolio_value_fn,
            skip_on_error=skip_on_error,
            futures_replay_signal_fixes_enabled=futures_replay_signal_fixes_enabled,
        )
        registered += _register_manifest_genetics_agents(
            registry,
            optional_agent_labels,
            portfolio_value_fn=portfolio_value_fn,
            skip_on_error=skip_on_error,
        )
        registered += _register_special_optional_agents(
            registry,
            optional_agent_labels,
            portfolio_value_fn=portfolio_value_fn,
            skip_on_error=skip_on_error,
        )
    return registered


def register_optional_agents(
    registry: AgentRegistry,
    *,
    portfolio_value_fn: Optional[callable] = None,
    skip_on_error: bool = True,
    optional_agent_labels: Optional[Sequence[str]] = None,
) -> List[str]:
    """Явная регистрация опциональных тяжёлых агентов (Genetics)."""
    _ensure_paths()
    registered = _register_from_list(
        registry, _filter_optional_agent_items(optional_agent_labels),
        portfolio_value_fn=portfolio_value_fn,
        skip_on_error=skip_on_error,
    )
    registered += _register_manifest_genetics_agents(
        registry,
        optional_agent_labels,
        portfolio_value_fn=portfolio_value_fn,
        skip_on_error=skip_on_error,
    )
    registered += _register_special_optional_agents(
        registry,
        optional_agent_labels,
        portfolio_value_fn=portfolio_value_fn,
        skip_on_error=skip_on_error,
    )
    return registered


def _filter_optional_agent_items(
    labels: Optional[Sequence[str]],
) -> List[Tuple[str, str]]:
    legacy_manifest_labels = set(MANIFEST_GENETICS_AGENT_LABELS)
    if labels is None:
        items = list(OPTIONAL_V1_AGENTS)
        if _manifest_genetics_configured():
            return [item for item in items if item[0] not in legacy_manifest_labels]
        return items
    allowed = {str(label) for label in labels}
    items = [item for item in OPTIONAL_V1_AGENTS if item[0] in allowed]
    if _manifest_genetics_configured():
        items = [item for item in items if item[0] not in legacy_manifest_labels]
    return items


def known_labels() -> List[str]:
    return [label for label, _ in KNOWN_V1_AGENTS]


def optional_labels() -> List[str]:
    labels: List[str] = []
    for label, _ in OPTIONAL_V1_AGENTS:
        if label not in labels:
            labels.append(label)
    for label in MANIFEST_GENETICS_AGENT_LABELS:
        if label not in labels:
            labels.append(label)
    labels.extend(OPTIONAL_SPECIAL_AGENTS)
    return labels


def genetics_labels() -> List[str]:
    return optional_labels()
