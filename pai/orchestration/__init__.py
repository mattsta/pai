"""
Orchestration module for Polyglot AI.

This package contains classes responsible for orchestrating the business logic
for different interaction modes (e.g., standard chat, agent loops, arenas).
It decouples the core logic from the UI implementation in `pai.pai`.
"""

from .arena import ArenaOrchestrator
from .base import BaseOrchestrator
from .default import DefaultOrchestrator
from .legacy_agent import LegacyAgentOrchestrator

__all__ = [
    "ArenaOrchestrator",
    "BaseOrchestrator",
    "DefaultOrchestrator",
    "LegacyAgentOrchestrator",
]
