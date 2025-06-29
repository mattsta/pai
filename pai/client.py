import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from .display import StreamingDisplay
from .models import (
    ChatRequest,
    CompletionRequest,
    EndpointConfig,
    PolyglotConfig,
    RuntimeConfig,
    SessionStats,
)
from .pricing import PricingService
from .protocols import ADAPTER_MAP
from .protocols.base_adapter import ProtocolContext


class APIError(Exception):
    pass


class PolyglotClient:
    def __init__(
        self,
        runtime_config: RuntimeConfig,
        toml_config: PolyglotConfig,
        http_session: httpx.AsyncClient,
        pricing_service: PricingService,
    ):
        self.toml_config = toml_config
        self.config = EndpointConfig()
        self.stats = SessionStats()
        self.display = StreamingDisplay(
            debug_mode=runtime_config.debug,
            rich_text_mode=runtime_config.rich_text,
            smooth_stream_mode=runtime_config.smooth_stream,
        )
        self.http_session = http_session
        self.tools_enabled = runtime_config.tools
        self.pricing_service = pricing_service
        self.switch_endpoint(runtime_config.endpoint)
        if self.config and runtime_config.model:
            self.config.model_name = runtime_config.model
        if self.config and runtime_config.timeout:
            self.config.timeout = runtime_config.timeout

    def switch_endpoint(self, name: str):
        endpoint_data = next(
            (e for e in self.toml_config.endpoints if e.name.lower() == name.lower()),
            None,
        )
        if not endpoint_data:
            self.display._print(
                f"❌ Error: Endpoint '{name}' not found in configuration file."
            )
            return
        api_key = os.getenv(endpoint_data.api_key_env)
        if not api_key:
            self.display._print(
                f"❌ Error: API key for '{name}' not found. Set {endpoint_data.api_key_env}."
            )
            return
        self.config.name = endpoint_data.name
        self.config.base_url = endpoint_data.base_url
        self.config.api_key = api_key
        self.config.chat_adapter = ADAPTER_MAP.get(endpoint_data.chat_adapter)
        self.config.completion_adapter = ADAPTER_MAP.get(
            endpoint_data.completion_adapter
        )
        # Configure the httpx client for the selected endpoint
        self.http_session.base_url = self.config.base_url
        self.http_session.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "PolyglotAI/0.1.0",
        }
        msg = f"✅ Switched to endpoint: {self.config.name}"
        self.display._print(msg)
        logging.info(msg)

    def set_model(self, model_name: str):
        """Sets the model for the current endpoint."""
        if self.config:
            self.config.model_name = model_name
            msg = f"✅ Model set to: {model_name}"
            self.display._print(msg)
            logging.info(msg)

    def set_timeout(self, timeout: int):
        """Sets the request timeout for the current endpoint."""
        if self.config:
            if timeout <= 0:
                self.display._print("❌ Timeout must be a positive integer.")
                return
            self.config.timeout = timeout
            msg = f"✅ Request timeout set to: {timeout}s"
            self.display._print(msg)
            logging.info(msg)

    async def generate(
        self,
        request: CompletionRequest | ChatRequest,
        verbose: bool,
        actor_name: str | None = None,
        confirmer: Callable[[str, dict], Awaitable[bool]] | None = None,
    ) -> dict[str, Any]:
        is_chat = isinstance(request, ChatRequest)
        adapter = (
            self.config.chat_adapter if is_chat else self.config.completion_adapter
        )
        if not adapter:
            raise APIError(
                f"Endpoint '{self.config.name}' does not support {'chat' if is_chat else 'completion'} mode."
            )
        if verbose:
            msg1 = f"\n🚀 Sending request via endpoint '{self.config.name}' using adapter '{type(adapter).__name__}'"
            msg2 = f"📝 Model: {self.config.model_name}"
            msg3 = f"🎛️ Parameters: temp={request.temperature}, max_tokens={request.max_tokens}, stream={request.stream}"
            self.display._print(msg1)
            self.display._print(msg2)
            self.display._print(msg3)
            logging.info(msg1)
            logging.info(msg2)
            logging.info(msg3)

        context = ProtocolContext(
            http_session=self.http_session,
            display=self.display,
            stats=self.stats,
            config=self.config,
            tools_enabled=self.tools_enabled,
            confirmer=confirmer,
            pricing_service=self.pricing_service,
        )
        return await adapter.generate(context, request, verbose, actor_name=actor_name)
