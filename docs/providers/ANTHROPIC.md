# How to Add a New Provider: An Anthropic Example

This guide walks through the process of adding support for a new AI provider to the Polyglot AI framework. We will use Anthropic's Claude models as a concrete example.

The core principle is creating a new **Protocol Adapter**. This is a Python class responsible for translating Polyglot AI's internal request format into the specific API format required by the new provider.

## Step 1: Create the Adapter File

Create a new file in the `pai/protocols/` directory. For our example, we'll call it `anthropic_adapter.py`.

```
pai/
└── protocols/
    ├── base_adapter.py
    ├── legacy_completion_adapter.py
    ├── openai_chat_adapter.py
    └── anthropic_adapter.py  <-- NEW FILE
```

## Step 2: Define the Adapter Class

Inside your new `anthropic_adapter.py` file, define a class that inherits from `BaseProtocolAdapter`. You must implement the `async def generate()` method.

```python
# pai/protocols/anthropic_adapter.py
import json
from .base_adapter import BaseProtocolAdapter, ProtocolContext, ChatRequest
from ..utils import estimate_tokens

class AnthropicAdapter(BaseProtocolAdapter):
    """Handles the Anthropic Messages API format."""

    async def generate(self, context: ProtocolContext, request: ChatRequest, verbose: bool):
        # Your implementation will go here
        # For this guide, we'll just pass. A real implementation is more complex.
        pass
```

## Step 3: Implement the `generate` Method

This is the core of the adapter. You need to handle both streaming and non-streaming requests.

### 3a. Translate the Request

Anthropic's API has its own request schema. You need to convert `pai`'s `ChatRequest` into this format. For example, Anthropic doesn't use a `system` role in the main message list; it's a top-level parameter.

```python
# Inside generate()
# Anthropic uses a top-level `system` parameter.
system_prompt = ""
messages = []
for msg in request.messages:
    if msg["role"] == "system":
        system_prompt = msg["content"]
    else:
        # Anthropic only allows alternating user/assistant roles.
        # A full implementation would need to handle merging consecutive messages.
        messages.append(msg)

url = f"{context.config.base_url}/messages"
payload = {
    "model": context.config.model_name,
    "system": system_prompt,
    "messages": messages,
    "max_tokens": request.max_tokens,
    "temperature": request.temperature,
    "stream": request.stream,
}
```

### 3b. Handle the API Call and Response Parsing

You'll use the `context.http_session` to make the call. For streaming, you must parse Server-Sent Events (SSE). Anthropic's format is slightly different from OpenAI's.

```python
# Inside generate(), after creating the payload
try:
    context.display.start_response()

    # Non-streaming implementation (simplified)
    if not request.stream:
        # Make a POST request, get the JSON response, parse it.
        # ... your non-streaming logic here ...
        pass

    # Streaming implementation
    async with context.http_session.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                data = json.loads(line[6:])
                event_type = data.get("type")
                if event_type == "content_block_delta":
                    chunk_text = data.get("delta", {}).get("text", "")
                    context.display.show_parsed_chunk(data, chunk_text)
                # ... handle other event types like message_start, message_stop ...

    # Finalize stats, add to history, and return the result dict
    request_stats = context.display.finish_response(success=True)
    # ... add tokens_sent etc. to request_stats ...
    context.stats.add_completed_request(request_stats)

    return { "text": context.display.current_response }

except Exception as e:
    # Handle errors gracefully
    context.display.finish_response(success=False)
    raise ConnectionError(f"Anthropic request failed: {e!r}")
```
*Note: This is a simplified example. A real implementation needs to handle all `event` types from the Anthropic stream (`message_start`, `content_block_start`, `ping`, etc.) for robust parsing.*

## Step 4: Register the New Adapter

Now that the adapter class is defined, you need to make `pai` aware of it.

Open `pai/pai.py` and import your new class. Then, add it to the `ADAPTER_MAP`. The key (`"anthropic"`) is what you will use in your configuration file.

```python
# In pai/pai.py

# --- Protocol Adapter Imports ---
from .protocols.base_adapter import BaseProtocolAdapter, ProtocolContext
from .protocols.openai_chat_adapter import OpenAIChatAdapter
from .protocols.legacy_completion_adapter import LegacyCompletionAdapter
from .protocols.anthropic_adapter import AnthropicAdapter # <-- IMPORT IT

# --- Global Definitions ---
session = PromptSession()
ADAPTER_MAP = {
    "openai_chat": OpenAIChatAdapter(),
    "legacy_completion": LegacyCompletionAdapter(),
    "anthropic": AnthropicAdapter(), # <-- REGISTER IT
}
```

## Step 5: Add an Endpoint to the Configuration

Finally, open your `polyglot.toml` file and add a new endpoint that uses your adapter.

```toml
# In polyglot.toml

[[endpoints]]
name = "anthropic"
base_url = "https://api.anthropic.com/v1"
api_key_env = "ANTHROPIC_API_KEY"
# Use the key you registered in ADAPTER_MAP
chat_adapter = "anthropic"
# Anthropic doesn't have a legacy /completions endpoint, so we omit it.
```

## You're Done!

That's it. You can now run `pai` with your new provider:

```bash
# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."

# Run the interactive chat
pai --chat --endpoint anthropic --model claude-3-5-sonnet-20240620
```

By following this pattern, you can integrate virtually any AI provider into the Polyglot AI framework.
