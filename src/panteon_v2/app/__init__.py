"""Application layer (Phase 9) — production wiring.

Boundary между чистой архитектурой v2 (panteon_v2/) и реальным миром
(v1 connectors, биржи, файловая система). Только здесь допустимы
импорты из panteon_runtime/ — обёрнутые в адаптерах.

Слои:
  bootstrap                 — composition root (DI всех компонентов)
  main_loop                 — production bar-loop
  exchange_adapter_template — шаблон адаптера v1-connector → v2 Exchange
  agent_bootstrap           — регистрация v1-агентов через V1AgentAdapter
  migration                 — v1 _regime_memory → v2 PerformanceMemory snapshot
  output_writer             — periodic status.json/leaderboard/dashboard
  v1_bridge_runner          — интеграция с v1-bridge как infrastructure
  startup                   — главный entry-point start_production()
  cli                       — операторская командная строка
"""

from .bootstrap import (
    PRODUCTION_PROFILES,
    ProductionPipeline,
    build_dryrun_pipeline,
    build_production_pipeline,
)
from .main_loop import (
    StepResult,
    main_loop,
)
from .migration import (
    MigrationReport,
    load_v2_snapshot,
    migrate_from_v1_memory_file,
    migrate_v1_regime_memory,
    save_v2_snapshot,
)
from .output_writer import OutputWriter, OutputWriterConfig
from .startup import start_production, resolve_exchange

__all__ = [
    # bootstrap
    "PRODUCTION_PROFILES",
    "ProductionPipeline",
    "build_dryrun_pipeline",
    "build_production_pipeline",
    # main_loop
    "StepResult",
    "main_loop",
    # migration
    "MigrationReport",
    "load_v2_snapshot",
    "migrate_from_v1_memory_file",
    "migrate_v1_regime_memory",
    "save_v2_snapshot",
    # output_writer
    "OutputWriter",
    "OutputWriterConfig",
    # startup
    "start_production",
    "resolve_exchange",
]
