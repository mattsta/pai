# protocols/base_adapter.py
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, Any, Union, NamedTuple
import httpx

if TYPE_CHECKING:
    from ..pai import EndpointConfig, StreamingDisplay, TestSession


# NEW: Define a simple data structure to pass context from the client to the adapters.
# This avoids the need for adapters to import the main client class, breaking the circular dependency.
class ProtocolContext(NamedTuple):
    http_session: httpx.AsyncClient
    display: "StreamingDisplay"
    stats: "TestSession"
    config: "EndpointConfig"
    tools_enabled: bool


# Forward-declare request types for type hinting
class CompletionRequest:
    pass


class ChatRequest:
    pass


class BaseProtocolAdapter(ABC):
    """Abstract Base Class for all protocol adapters."""

    @abstractmethod
    async def generate(
        self,
        # MODIFIED: The method now accepts the clean context object.
        context: ProtocolContext,
        request: Union["CompletionRequest", "ChatRequest"],
        verbose: bool,
    ) -> Dict[str, Any]:
        """The main generation method."""
        pass
