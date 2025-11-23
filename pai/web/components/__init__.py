"""HTMX components for the web interface.

Each component is self-contained with single responsibility:
- ChatComponent: Message input and response streaming
- ModelsComponent: Model discovery and selection
- StatsComponent: Session statistics display
- SettingsComponent: Runtime configuration
"""

from .base import Component
from .chat import ChatComponent
from .models import ModelsComponent
from .settings import SettingsComponent
from .stats import StatsComponent

__all__ = [
    "Component",
    "ChatComponent",
    "ModelsComponent",
    "StatsComponent",
    "SettingsComponent",
]
