"""Base class for all orchestrators."""

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from ..log_utils import save_conversation_formats
from ..models import ChatRequest, CompletionRequest, Turn

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
        self.log_dir = ui.log_dir

    @abstractmethod
    async def run(self, user_input: str | None = None) -> Any:
        """Runs the orchestration logic for the specific mode."""
        pass

    def _log_cancelled_turn(
        self, request: ChatRequest | CompletionRequest, partial_text: str
    ):
        """Saves information about a cancelled turn for debugging."""
        request_data = request.to_dict(self.client.config.model_name)
        response_data = {
            "pai_note": "This response was cancelled by the user.",
            "choices": [{"message": {"role": "assistant", "content": partial_text}}],
        }
        turn = Turn(
            request_data=request_data,
            response_data=response_data,
            assistant_message=partial_text,
            mode=self.state.mode,
            stats=self.client.stats.last_request_stats,
            endpoint_name=self.client.config.name,
        )
        self.conversation.add_turn(turn)
        try:
            turn_file = self.log_dir / f"{turn.turn_id}-turn.json"
            turn_file.write_text(json.dumps(turn.to_dict(), indent=2), encoding="utf-8")
            save_conversation_formats(
                self.conversation, self.log_dir, printer=self.pt_printer
            )
        except Exception as e:
            self.pt_printer(f"\n⚠️  Warning: Could not save cancelled session turn: {e}")
