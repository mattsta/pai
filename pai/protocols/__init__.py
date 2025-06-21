# This file serves as a registry for all protocol adapters.

from .anthropic_adapter import AnthropicAdapter
from .legacy_completion_adapter import LegacyCompletionAdapter
from .ollama_adapter import OllamaAdapter
from .openai_chat_adapter import OpenAIChatAdapter

ADAPTER_MAP = {
    "openai_chat": OpenAIChatAdapter(),
    "legacy_completion": LegacyCompletionAdapter(),
    "anthropic": AnthropicAdapter(),
    "ollama": OllamaAdapter(),
}
