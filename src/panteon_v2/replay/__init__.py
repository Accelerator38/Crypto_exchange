"""Replay layer (Phase 7) — прогон v1-логов через v2 pipeline.

Цель: проверить гарантии v2 на исторических данных без необходимости
запускать v2 в продакшн. Никаких импортов из panteon_runtime — читаем
только данные (JSON, CSV, log) из v1-сессий.

Использование:
    python -m panteon_v2.replay.cli /path/to/Results/MEXC/2026-05-05_*/
"""

from .v1_parser import (
    V1Session,
    V1Signal,
    V1Trade,
    V1AgentEntry,
    parse_session,
    parse_status_json,
    parse_leaderboard_json,
    parse_all_signals_csv,
)
from .synthesizer import synthesize_v2_state
from .validator import (
    ReplayValidationReport,
    ValidationResult,
    validate_session,
)

__all__ = [
    # parser
    "V1Session",
    "V1Signal",
    "V1Trade",
    "V1AgentEntry",
    "parse_session",
    "parse_status_json",
    "parse_leaderboard_json",
    "parse_all_signals_csv",
    # synthesizer
    "synthesize_v2_state",
    # validator
    "ReplayValidationReport",
    "ValidationResult",
    "validate_session",
]
