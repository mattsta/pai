import json
import httpx
from typing import Any, Dict, Optional

LITELLM_PRICING_URL = "https://raw.githubusercontent.com/BerriAI/litellm/refs/heads/main/model_prices_and_context_window.json"

class PricingService:
    def __init__(self, url: str = LITELLM_PRICING_URL):
        self.url = url
        self._pricing_data: Optional[Dict[str, Any]] = None
        self._provider_map: Dict[str, str] = {
            "openai": "openai",
            "anthropic": "anthropic",
            "ollama": "ollama",
            "together": "together-ai", # LiteLLM uses together-ai
            # Add more mappings as needed for other providers
        }

    async def load_pricing_data(self):
        """Fetches and caches the pricing data from the LiteLLM URL."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.url, timeout=10.0)
                response.raise_for_status() # Raise an exception for 4xx or 5xx responses
                self._pricing_data = response.json()
        except httpx.RequestError as e:
            print(f"Warning: Could not fetch pricing data from {self.url}: {e}")
            self._pricing_data = {}
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse pricing data from {self.url}: {e}")
            self._pricing_data = {}

    def get_model_pricing(self, provider_name: str, model_name: str) -> Dict[str, float]:
        """Looks up pricing for a given model and provider.

        Args:
            provider_name (str): The Polyglot AI endpoint name (e.g., "openai", "anthropic").
            model_name (str): The model name (e.g., "gpt-4o", "claude-3-haiku-20240307").

        Returns:
            Dict[str, float]: A dictionary with 'input_cost_per_token' and 'output_cost_per_token'.
                              Returns 0.0 for both if not found.
        """
        if self._pricing_data is None:
            # Data not loaded, return default zero costs
            return {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}

        # Map Polyglot AI provider name to LiteLLM provider name
        litellm_provider = self._provider_map.get(provider_name.lower(), provider_name.lower())

        # LiteLLM model keys are often in the format "provider/model_name"
        # or just "model_name" if the provider is implicit.
        # We'll try a few common patterns.
        lookup_keys = [
            f"{litellm_provider}/{model_name}",
            model_name, # Direct model name lookup
            f"{litellm_provider}/{model_name.replace('-', '_')}", # Try replacing hyphens with underscores
            f"{litellm_provider}/{model_name.split('/')[-1]}", # For cases like 'org/model'
        ]

        for key in lookup_keys:
            model_info = self._pricing_data.get(key)
            if model_info:
                return {
                    "input_cost_per_token": model_info.get("input_cost_per_token", 0.0),
                    "output_cost_per_token": model_info.get("output_cost_per_token", 0.0),
                }

        # If no specific pricing found, return default zero costs
        return {"input_cost_per_token": 0.0, "output_cost_per_token": 0.0}
