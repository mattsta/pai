# protocols/base_adapter.py
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, NamedTuple, Union

import httpx

# MODIFIED: Import from models and pai separately to avoid circular deps.
if TYPE_CHECKING:
    from ..display import StreamingDisplay
    from ..models import ChatRequest, CompletionRequest, EndpointConfig, SessionStats
    from ..pricing import PricingService


# NEW: Define a simple data structure to pass context from the client to the adapters.
# This avoids the need for adapters to import the main client class, breaking the circular dependency.
class ProtocolContext(NamedTuple):
    http_session: httpx.AsyncClient
    display: "StreamingDisplay"
    stats: "SessionStats"
    config: "EndpointConfig"
    tools_enabled: bool
    confirmer: Callable[[str, dict], Awaitable[bool]] | None
    pricing_service: "PricingService"


# Request types are now imported from pai.models


class BaseProtocolAdapter(ABC):
    """Abstract Base Class for all protocol adapters."""

    @abstractmethod
    async def generate(
        self,
        # MODIFIED: The method now accepts the clean context object.
        context: ProtocolContext,
        request: Union["CompletionRequest", "ChatRequest"],
        verbose: bool,
        actor_name: str | None = None,
    ) -> dict[str, Any]:
        """The main generation method."""
        pass
