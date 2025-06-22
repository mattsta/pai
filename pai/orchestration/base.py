"""Base class for all orchestrators."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..pai import InteractiveUI


class BaseOrchestrator(ABC):
    """Abstract base class for all mode-specific orchestrators."""

    def __init__(self, ui: "InteractiveUI"):
        self.ui = ui
        # Make commonly used components easily accessible
        self.client = ui.client
        self.state = ui.state
        self.pt_printer = ui.pt_printer
        self.runtime_config = ui.runtime_config
        self.conversation = ui.conversation
        self.session_dir = ui.session_dir

    @abstractmethod
    async def run(self, user_input: str | None = None) -> Any:
        """Runs the orchestration logic for the specific mode."""
        pass
