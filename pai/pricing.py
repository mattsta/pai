from typing import Any, Dict, Optional
import dataclasses
import toml
import yaml

import pathlib
import json
from datetime import UTC, datetime, timedelta
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
            path = pathlib.Path(custom_file_path)
            if not path.is_file():
                print(
                    f"Warning: Custom pricing file not found at '{custom_file_path}'"
                )
                return

            try:
                custom_data = None
                content = path.read_text("utf-8")
                if path.suffix.lower() == ".toml":
                    custom_data = toml.loads(content)
                elif path.suffix.lower() in [".yaml", ".yml"]:
                    custom_data = yaml.safe_load(content)
                else:
                    print(
                        f"Warning: Unsupported custom pricing file format: '{path.suffix}'. Please use .toml or .yaml/.yml."
                    )
                    return

                if custom_data:
                    self._custom_pricing_data = TomlCustomPricing.model_validate(
                        custom_data
                    )
                    print(f"Loaded custom pricing from '{custom_file_path}'")

            except (toml.TomlDecodeError, yaml.YAMLError, Exception) as e:
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
                # Parse tiered costs from LiteLLM format, converting to cost per 1M tokens
                tiered_input_costs = [
                    TieredCost(
                        up_to=t["tokens_up_to"], cost=t["cost_per_token"] * 1_000_000
                    )
                    for t in model_info.get("tiered_input_costs", [])
                ]
                tiered_output_costs = [
                    TieredCost(
                        up_to=t["tokens_up_to"], cost=t["cost_per_token"] * 1_000_000
                    )
                    for t in model_info.get("tiered_output_costs", [])
                ]

                # LiteLLM format doesn't have the rich time_windows structure we support.
                # It will be populated from custom pricing files later.
                # Convert costs to be per 1M tokens for consistency.
                return ModelPricing(
                    input_cost_per_token=model_info.get("input_cost_per_token", 0.0)
                    * 1_000_000,
                    output_cost_per_token=model_info.get("output_cost_per_token", 0.0)
                    * 1_000_000,
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

    def _calculate_tiered_cost(self, tiers: list[TieredCost], total_tokens: int) -> float:
        """Calculates the cost for a token count based on a tiered structure."""
        if not tiers or total_tokens <= 0:
            return 0.0

        # Sort tiers by their 'up_to' boundary. -1 (infinity) is handled correctly.
        sorted_tiers = sorted(
            tiers, key=lambda t: t.up_to if t.up_to != -1 else float("inf")
        )

        total_cost = 0.0
        remaining_tokens = total_tokens
        last_boundary = 0

        for tier in sorted_tiers:
            tier_upper_bound = tier.up_to
            if tier_upper_bound == -1:
                # This is the last tier, it takes all remaining tokens.
                tier_upper_bound = float("inf")

            # Number of tokens this tier's price applies to
            tier_capacity = tier_upper_bound - last_boundary

            # Number of tokens to actually price in this tier for this request
            tokens_in_tier = min(remaining_tokens, tier_capacity)

            if tokens_in_tier <= 0:
                # This can happen if tiers overlap, though they shouldn't.
                # Also handles the case where remaining_tokens is now 0.
                continue

            total_cost += (tokens_in_tier / 1_000_000) * tier.cost

            remaining_tokens -= tokens_in_tier
            if remaining_tokens <= 0:
                break

            last_boundary = tier_upper_bound

        return total_cost

    def calculate_cost(
        self, model_pricing: ModelPricing, input_tokens: int, output_tokens: int
    ) -> tuple[float, float]:
        """
        Calculates the cost for a given number of input and output tokens
        based on the model's specific pricing rules (flat, tiered, time-based).
        All costs are in USD.

        Args:
            model_pricing: The ModelPricing object containing the rates.
            input_tokens: The number of input tokens.
            output_tokens: The number of output tokens.

        Returns:
            A tuple containing (input_cost, output_cost).
        """
        now_utc = datetime.now(UTC)
        current_hour = now_utc.hour

        # Default to "anytime" pricing
        active_input_cost_rate = model_pricing.input_cost_per_token
        active_output_cost_rate = model_pricing.output_cost_per_token
        active_input_tiers = model_pricing.tiered_input_costs
        active_output_tiers = model_pricing.tiered_output_costs

        # Check for time-based pricing overrides
        for window in model_pricing.time_windows:
            # Handle overnight windows (e.g., 22:00 to 06:00)
            if window.start_hour > window.end_hour:
                if current_hour >= window.start_hour or current_hour <= window.end_hour:
                    active_input_cost_rate = window.input_cost
                    active_output_cost_rate = window.output_cost
                    active_input_tiers = window.input_tiers
                    active_output_tiers = window.output_tiers
                    break
            # Handle standard daytime windows
            elif window.start_hour <= current_hour <= window.end_hour:
                active_input_cost_rate = window.input_cost
                active_output_cost_rate = window.output_cost
                active_input_tiers = window.input_tiers
                active_output_tiers = window.output_tiers
                break

        # --- Calculate Input Cost ---
        if active_input_tiers:
            input_cost = self._calculate_tiered_cost(active_input_tiers, input_tokens)
        else:
            input_cost = (input_tokens / 1_000_000) * active_input_cost_rate

        # --- Calculate Output Cost ---
        if active_output_tiers:
            output_cost = self._calculate_tiered_cost(active_output_tiers, output_tokens)
        else:
            output_cost = (output_tokens / 1_000_000) * active_output_cost_rate

        return input_cost, output_cost
