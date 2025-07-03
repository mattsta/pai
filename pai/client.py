import json
import logging
import os
import pathlib
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
    def _get_cache_dir(self) -> pathlib.Path:
        """Returns the appropriate cache directory for the OS."""
        cache_dir = pathlib.Path.home() / ".cache" / "pai"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def __init__(
        self,
        runtime_config: RuntimeConfig,
        toml_config: PolyglotConfig,
        http_session: httpx.AsyncClient,
        pricing_service: PricingService,
        version: str,
    ):
        self.version = version
        self.toml_config = toml_config
        self.config = EndpointConfig()
        self.stats = SessionStats()
        self.display = StreamingDisplay(
            debug_mode=runtime_config.debug,
            rich_text_mode=runtime_config.rich_text,
            smooth_stream_mode=runtime_config.smooth_stream,
        )
        self.http_session = http_session
        self._cache_dir = self._get_cache_dir()
        self._models_cache_dir = self._cache_dir / "models"
        self._model_info_cache_dir = self._cache_dir / "model_info"
        self._models_cache_dir.mkdir(exist_ok=True)
        self._model_info_cache_dir.mkdir(exist_ok=True)
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
        api_key = None
        if endpoint_data.api_key_env:
            api_key = os.getenv(endpoint_data.api_key_env)
            if not api_key:
                self.display._print(
                    f"❌ Error: API key environment variable '{endpoint_data.api_key_env}' is set in config but not found in environment."
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
        user_agent = endpoint_data.user_agent or f"PolyglotAI/{self.version}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": user_agent,
        }
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        self.http_session.headers = headers
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

    async def list_models(
        self, force_refresh: bool = False, search_term: str | None = None
    ) -> list[str]:
        """
        Fetches the list of available models from the provider, using a cache.
        The cache now stores the full model dictionary from the provider.
        An optional search term can be provided to filter the results.
        Returns an empty list on failure, printing an error message.
        """
        endpoint_name = self.config.name
        safe_endpoint_name = "".join(
            c for c in endpoint_name if c.isalnum() or c in ("-", "_")
        ).rstrip()
        cache_file = self._models_cache_dir / f"{safe_endpoint_name}.json"

        model_data_list: list[dict[str, Any]] | None = None
        if not force_refresh and cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    # Handle both old (list of strings) and new (list of dicts) formats
                    if cached_data and isinstance(cached_data[0], dict):
                        model_data_list = cached_data
                    elif cached_data:
                        # Convert old format to new format for internal consistency
                        model_data_list = [{"id": m} for m in cached_data]
            except (json.JSONDecodeError, IndexError):
                pass  # Corrupted or empty cache, will be refreshed

        if model_data_list is None:  # Cache miss or corrupted
            try:
                response = await self.http_session.get("models")
                response.raise_for_status()
                data = response.json()
                model_data_list = data.get("data", [])

                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(model_data_list, f, indent=2)
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                self.display._print(f"❌ Error fetching models: {e}")
                return []
            except (KeyError, TypeError, json.JSONDecodeError):
                self.display._print("❌ Error parsing models response. Unexpected format.")
                return []

        # Return a list of model IDs for backward compatibility with callers
        model_ids = sorted([m.get("id", "") for m in model_data_list])
        if search_term:
            return [m for m in model_ids if search_term.lower() in m.lower()]

        return model_ids

    async def get_cached_provider_model_info(
        self, model_id: str
    ) -> dict[str, Any] | None:
        """
        Loads the provider model cache and searches for a specific model's info.
        This does NOT make a network call.
        """
        endpoint_name = self.config.name
        safe_endpoint_name = "".join(
            c for c in endpoint_name if c.isalnum() or c in ("-", "_")
        ).rstrip()
        cache_file = self._models_cache_dir / f"{safe_endpoint_name}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    model_data_list = json.load(f)

                if not isinstance(model_data_list, list):
                    return None

                for model_info in model_data_list:
                    if isinstance(model_info, dict) and model_info.get("id") == model_id:
                        return model_info
            except (json.JSONDecodeError, IndexError):
                return None  # Cache is corrupt or not in the expected format.
        return None

    async def get_model_info(self, model_id: str) -> dict[str, Any] | None:
        """
        Fetches detailed model information from the Hugging Face Hub API, with caching.
        """
        # Sanitize model_id to create a safe path.
        # This handles model_ids like 'Org/ModelName' by creating subdirectories.
        cache_path = self._model_info_cache_dir / f"{model_id}.json"

        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                # Corrupted cache file, will proceed to fetch.
                pass

        try:
            url = f"https://huggingface.co/api/models/{model_id}"
            # Use a temporary client to avoid interfering with the main session's base URL and headers.
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            model_data = response.json()

            # Create parent directories if they don't exist
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(model_data, f, indent=2)

            return model_data
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                self.display._print(f"❌ Model '{model_id}' not found on Hugging Face Hub.")
            else:
                self.display._print(
                    f"❌ Error fetching model info: HTTP {e.response.status_code}"
                )
            return None
        except (httpx.RequestError, json.JSONDecodeError) as e:
            self.display._print(f"❌ Error fetching or parsing model info: {e!r}")
            return None

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
