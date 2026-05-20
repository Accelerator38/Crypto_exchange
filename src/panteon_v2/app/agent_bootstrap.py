"""Регистрация v1-агентов в v2 AgentRegistry.

Это место в panteon_v2/, где допустим импорт из panteon_runtime/.
Здесь мы оборачиваем v1-агенты под v2 Agent Protocol через V1AgentAdapter.

Использование:
    from panteon_v2.app.agent_bootstrap import register_all_v1_agents
    registry = AgentRegistry()
    register_all_v1_agents(registry)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

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

OPTIONAL_SPECIAL_AGENTS: Tuple[str, ...] = ("GeneticsRegimeRouter",)


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


def _resolve_router_manifest_path(path: str | Path, *, results_root: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = results_root / resolved
    resolved = resolved.resolve()
    try:
        resolved.relative_to(results_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"genetics router path must be under {results_root}: {resolved}"
        ) from exc
    return resolved


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
    return _router_promotion_eligible(selection) and bool(
        selection.get("live_trading_eligible", False)
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
            adapter_kwargs = {
                "label": label,
                "v1_agent": instance,
                "portfolio_value_fn": portfolio_value_fn,
            }
            if adapter_cls is GeneticsV2AgentAdapter and label == "GeneticsGenomeEnsemble":
                adapter_kwargs["allowed_open_regimes"] = ("bearish", "crash")
            adapter = adapter_cls(**adapter_kwargs)
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
            if label != "GeneticsRegimeRouter":
                continue
            if not _env_flag("PANTEON_V2_LOAD_GENETICS_ROUTER", False):
                continue
            manifest = os.environ.get("PANTEON_V2_GENETICS_ROUTER_MANIFEST")
            if not manifest:
                raise ValueError("PANTEON_V2_GENETICS_ROUTER_MANIFEST is required")
            module = __import__("crypto_genetics", fromlist=["GeneticsAgent"])
            adapter = build_genetics_regime_router_adapter(
                manifest,
                getattr(module, "GeneticsAgent"),
                portfolio_value_fn=portfolio_value_fn,
                expected_genome_size=getattr(module, "GENOME_SIZE", None),
            )
            registry.register(adapter, replace=True)
            registered.append(label)
        except Exception as exc:
            if skip_on_error:
                log.warning("Skipping %s: %s", label, type(exc).__name__)
                continue
            raise ImportError(f"Failed to register {label}: {exc}") from exc
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
    if labels is None:
        return list(OPTIONAL_V1_AGENTS)
    allowed = {str(label) for label in labels}
    return [item for item in OPTIONAL_V1_AGENTS if item[0] in allowed]


def known_labels() -> List[str]:
    return [label for label, _ in KNOWN_V1_AGENTS]


def optional_labels() -> List[str]:
    return [label for label, _ in OPTIONAL_V1_AGENTS] + list(OPTIONAL_SPECIAL_AGENTS)


def genetics_labels() -> List[str]:
    return optional_labels()
