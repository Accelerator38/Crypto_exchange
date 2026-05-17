"""Регистрация v1-агентов в v2 AgentRegistry.

Это место в panteon_v2/, где допустим импорт из panteon_runtime/.
Здесь мы оборачиваем v1-агенты под v2 Agent Protocol через V1AgentAdapter.

Использование:
    from panteon_v2.app.agent_bootstrap import register_all_v1_agents
    registry = AgentRegistry()
    register_all_v1_agents(registry)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..selection import AgentRegistry
from ..shadow.adapters import GeneticsV2AgentAdapter, V1AgentAdapter


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
    ("CarryFlowAgentV2",    "panteon_agents:CarryFlowAgentV2"),
    ("ResearchValidatorAgent", "panteon_agents:ResearchValidatorAgent"),
    ("VolBreakoutHunter",   "panteon_agents:VolBreakoutHunter"),
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
OPTIONAL_V1_AGENTS: List[Tuple[str, str]] = [
    ("GeneticsGenomeEnsemble", "agents_v2:GenomeEnsembleAgent"),
    ("GeneticsCore",        "crypto_genetics:GeneticsAgent"),
    ("GeneticsBullish",     "crypto_genetics:GeneticsBullishAgent"),
    ("GeneticsBearish",     "crypto_genetics:GeneticsBearishAgent"),
    ("GeneticsNeutral",     "crypto_genetics:GeneticsNeutralAgent"),
]


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


def _register_from_list(
    registry: AgentRegistry,
    items: List[Tuple[str, str]],
    *,
    portfolio_value_fn: Optional[callable] = None,
    skip_on_error: bool = True,
) -> List[str]:
    registered: List[str] = []
    for label, dotted in items:
        try:
            module_path, class_name = dotted.split(":")
            module = __import__(module_path, fromlist=[class_name])
            cls = getattr(module, class_name)
            instance = cls()
            adapter_cls = (
                GeneticsV2AgentAdapter
                if module_path == "crypto_genetics" or label.startswith("Genetics")
                else V1AgentAdapter
            )
            adapter = adapter_cls(
                label=label,
                v1_agent=instance,
                portfolio_value_fn=portfolio_value_fn,
            )
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


def register_all_v1_agents(
    registry: AgentRegistry,
    *,
    portfolio_value_fn: Optional[callable] = None,
    skip_on_error: bool = True,
    include_optional: Optional[bool] = None,
    optional_agent_labels: Optional[Sequence[str]] = None,
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
    )
    if include_optional:
        log.info("Loading optional agents (Genetics) — may take a while…")
        optional_items = _filter_optional_agent_items(optional_agent_labels)
        registered += _register_from_list(
            registry, optional_items,
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
    return _register_from_list(
        registry, _filter_optional_agent_items(optional_agent_labels),
        portfolio_value_fn=portfolio_value_fn,
        skip_on_error=skip_on_error,
    )


def _filter_optional_agent_items(
    labels: Optional[Sequence[str]],
) -> List[Tuple[str, str]]:
    if labels is None:
        return list(OPTIONAL_V1_AGENTS)
    allowed = {str(label) for label in labels}
    return [item for item in OPTIONAL_V1_AGENTS if item[0] in allowed]


def known_labels() -> List[str]:
    return [label for label, _ in KNOWN_V1_AGENTS]


def optional_labels() -> List[str]:
    return [label for label, _ in OPTIONAL_V1_AGENTS]


def genetics_labels() -> List[str]:
    return optional_labels()
