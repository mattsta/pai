from typing import Any, Dict, Optional
import dataclasses
import toml

import pathlib
import json
from datetime import datetime, timedelta
import re
import httpx
from .models import (
    ModelPricing,
    TieredCost,
    TimeWindowPricing,
    TomlCustomPricing,
    TomlCustomModelPricing,
)

LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"
LITELLM_PRICING_CACHE_FILE = "model_prices_and_context_window.json"
CACHE_EXPIRATION_DAYS = 7

class PricingService:
    def __init__(self, url: str = LITELLM_PRICING_URL):
        self.url = url
        self._pricing_data: Optional[Dict[str, Any]] = None
        self._custom_pricing_data: Optional[TomlCustomPricing] = None
        self._provider_map: Dict[str, str] = {
            "openai": "openai",
            "anthropic": "anthropic",
            "ollama": "ollama",
            "together": "together-ai",  # LiteLLM uses together-ai
            "mistral": "mistralai", # Mistral AI is 'mistralai' in LiteLLM
            "azure": "azure", # Azure is 'azure' in LiteLLM
            # Add more mappings as needed for other providers
        }
        self._cache_dir = self._get_cache_dir()
        self._cache_file_path = self._cache_dir / LITELLM_PRICING_CACHE_FILE

    def _get_cache_dir(self) -> pathlib.Path:
        """Returns the appropriate cache directory for the OS."""
        cache_dir = pathlib.Path.home() / ".cache" / "pai"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    async def load_pricing_data(self, custom_file_path: str | None = None):
        """
        Fetches and caches pricing data from LiteLLM. Optionally loads a custom
        pricing file to override or supplement the base pricing.
        """
        # 1. Try loading from cache first if it's fresh
        if self._cache_file_path.exists():
            file_mod_time = datetime.fromtimestamp(self._cache_file_path.stat().st_mtime)
            if datetime.now() - file_mod_time < timedelta(days=CACHE_EXPIRATION_DAYS):
                try:
                    with open(self._cache_file_path, 'r', encoding='utf-8') as f:
                        self._pricing_data = json.load(f)
                    print(f"Loaded pricing data from cache: {self._cache_file_path}")
                    return
                except json.JSONDecodeError as e:
                    print(f"Warning: Corrupted pricing cache file, re-downloading: {e}")
                    self._pricing_data = None # Invalidate cache

        # 2. If cache is stale or corrupted, try fetching from URL
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.url, timeout=10.0)
                response.raise_for_status() # Raise an exception for 4xx or 5xx responses
                self._pricing_data = response.json()
                # Save to cache
                with open(self._cache_file_path, 'w', encoding='utf-8') as f:
                    json.dump(self._pricing_data, f)
                print(f"Successfully fetched and cached pricing data to {self._cache_file_path}")
        except httpx.RequestError as e:
            print(f"Warning: Could not fetch pricing data from {self.url}: {e}")
            # 3. If fetching fails, try loading from potentially old cache as fallback
            if self._cache_file_path.exists():
                try:
                    with open(self._cache_file_path, 'r', encoding='utf-8') as f:
                        self._pricing_data = json.load(f)
                    print(f"Loaded stale pricing data from cache as fallback: {self._cache_file_path}")
                except json.JSONDecodeError as e:
                    print(f"Error: Could not load pricing data from cache (corrupted fallback): {e}")
                    self._pricing_data = {}
            else:
                print("Error: No pricing data available (no network and no cache).")
                self._pricing_data = {}
        except json.JSONDecodeError as e:
            print(f"Error: Could not parse pricing data from {self.url}: {e}")
            self._pricing_data = {}

        # 4. Load custom pricing file if provided
        if custom_file_path:
            try:
                path = pathlib.Path(custom_file_path)
                with open(path, "r", encoding="utf-8") as f:
                    custom_data = toml.load(f)
                    self._custom_pricing_data = TomlCustomPricing.model_validate(
                        custom_data
                    )
                print(f"Loaded custom pricing from '{custom_file_path}'")
            except FileNotFoundError:
                print(
                    f"Warning: Custom pricing file not found at '{custom_file_path}'"
                )
            except (toml.TomlDecodeError, Exception) as e:
                print(
                    f"Warning: Could not parse custom pricing file '{custom_file_path}': {e}"
                )

    def _merge_pricing(
        self, base: ModelPricing, custom: TomlCustomModelPricing
    ) -> ModelPricing:
        """Merges custom pricing rules onto a base ModelPricing object."""
        # Create a new ModelPricing object from the base to avoid mutating it.
        merged = dataclasses.replace(base)

        if custom.input_cost is not None:
            merged.input_cost_per_token = custom.input_cost
        if custom.output_cost is not None:
            merged.output_cost_per_token = custom.output_cost

        if custom.input_tiers:
            merged.tiered_input_costs = [
                TieredCost(up_to=t.up_to, cost=t.cost) for t in custom.input_tiers
            ]
        if custom.output_tiers:
            merged.tiered_output_costs = [
                TieredCost(up_to=t.up_to, cost=t.cost) for t in custom.output_tiers
            ]

        if custom.time_windows:
            merged.time_windows = [
                TimeWindowPricing(
                    start_hour=tw.start_hour,
                    end_hour=tw.end_hour,
                    input_cost=tw.input_cost if tw.input_cost is not None else 0.0,
                    output_cost=tw.output_cost if tw.output_cost is not None else 0.0,
                    input_tiers=[
                        TieredCost(up_to=t.up_to, cost=t.cost) for t in tw.input_tiers
                    ],
                    output_tiers=[
                        TieredCost(up_to=t.up_to, cost=t.cost)
                        for t in tw.output_tiers
                    ],
                )
                for tw in custom.time_windows
            ]

        return merged

    def get_model_pricing(self, provider_name: str, model_name: str) -> ModelPricing:
        """
        Looks up pricing for a model, combining base pricing with custom overrides.
        """
        # 1. Get base pricing from the cached LiteLLM data.
        base_pricing = self._get_base_model_pricing(provider_name, model_name)

        # 2. If no custom pricing file is loaded, we're done.
        if not self._custom_pricing_data:
            return base_pricing

        # 3. Check for a matching provider and model in the custom data.
        custom_provider_pricing = self._custom_pricing_data.pricing.get(
            provider_name.lower()
        )
        if not custom_provider_pricing:
            return base_pricing

        custom_model_pricing = custom_provider_pricing.get(model_name)
        if not custom_model_pricing:
            return base_pricing

        # 4. If a custom entry is found, merge it over the base pricing.
        print(f"Applying custom pricing for '{provider_name}/{model_name}'")
        return self._merge_pricing(base_pricing, custom_model_pricing)

    def _get_base_model_pricing(
        self, provider_name: str, model_name: str
    ) -> ModelPricing:
        """Looks up the base pricing from the cached LiteLLM data."""

        Args:
            provider_name (str): The Polyglot AI endpoint name (e.g., "openai", "anthropic").
            model_name (str): The model name (e.g., "gpt-4o", "claude-3-haiku-20240307").

        Returns:
            ModelPricing: An instance of ModelPricing with 'input_cost_per_token' and 'output_cost_per_token'.
                          Returns 0.0 for both if not found.
        """
        if self._pricing_data is None:
            # Data not loaded, return default zero costs
            return ModelPricing()

        # Normalize provider name
        litellm_provider = self._provider_map.get(provider_name.lower(), provider_name.lower())

        # Normalize model name for lookup (lowercase and remove common suffixes/prefixes)
        normalized_model_name = model_name.lower()
        # Remove common suffixes/prefixes that might not be in LiteLLM keys
        normalized_model_name = re.sub(r'-(latest|preview|20\d{2}-\d{2}-\d{2}|v\d+)$|', '', normalized_model_name)
        normalized_model_name = normalized_model_name.replace('gpt-4o', 'gpt-4o') # Ensure gpt-4o is consistent

        # LiteLLM model keys are often in the format "provider/model_name" or just "model_name".
        # We'll try a few common patterns, prioritizing more specific matches.
        lookup_keys = [
            f"{litellm_provider}/{normalized_model_name}",
            normalized_model_name, # Direct model name lookup
            f"{litellm_provider}/{normalized_model_name.replace('-', '_')}", # Try replacing hyphens with underscores
            f"{litellm_provider}/{normalized_model_name.split('/')[-1]}", # For cases like 'org/model'
            model_name.lower(), # Original model name, lowercased
        ]

        for key in lookup_keys:
            model_info = self._pricing_data.get(key)
            if model_info:
                # Parse tiered costs from LiteLLM format into our new structure
                tiered_input_costs = [
                    TieredCost(up_to=t["tokens_up_to"], cost=t["cost_per_token"])
                    for t in model_info.get("tiered_input_costs", [])
                ]
                tiered_output_costs = [
                    TieredCost(up_to=t["tokens_up_to"], cost=t["cost_per_token"])
                    for t in model_info.get("tiered_output_costs", [])
                ]

                # LiteLLM format doesn't have the rich time_windows structure we support.
                # It will be populated from custom pricing files later.
                return ModelPricing(
                    input_cost_per_token=model_info.get("input_cost_per_token", 0.0),
                    output_cost_per_token=model_info.get("output_cost_per_token", 0.0),
                    tiered_input_costs=tiered_input_costs,
                    tiered_output_costs=tiered_output_costs,
                    time_windows=[],
                    input_cost_per_token_batches=model_info.get(
                        "input_cost_per_token_batches", 0.0
                    ),
                    output_cost_per_token_batches=model_info.get(
                        "output_cost_per_token_batches", 0.0
                    ),
                )

        # If no specific pricing found, return default zero costs
        return ModelPricing()
