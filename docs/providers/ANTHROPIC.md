# How to Add a New Provider: The Plugin System

The Polyglot AI framework features a dynamic plugin system for adding new AI providers. This guide explains how to create your own installable Python package that `pai` can automatically discover and use. We will use a hypothetical "MyAI" provider as an example.

The core principle is to create a **Protocol Adapter** and register it as a **plugin** using Python's standard `entry_points` mechanism.

## Step 1: Create a New Python Project

First, set up a new Python package for your adapter. The structure might look like this:

```
myai-pai-adapter/
├── pyproject.toml
└── myai_adapter/
    ├── __init__.py
    └── adapter.py
```

-   `pyproject.toml`: The standard file for defining your package, its dependencies, and the crucial entry point.
-   `myai_adapter/`: The Python package containing your adapter code.

## Step 2: Define the Adapter Class

Inside `myai_adapter/adapter.py`, define a class that inherits from `pai.protocols.base_adapter.BaseProtocolAdapter`. You can depend on `pai` in your `pyproject.toml` to get access to the base classes. You must implement the `async def generate()` method.

```python
# In myai_adapter/adapter.py
from pai.models import ChatRequest
from pai.protocols.base_adapter import BaseProtocolAdapter, ProtocolContext

class MyAIAdapter(BaseProtocolAdapter):
    """Handles the MyAI API format."""
    
    async def generate(self, context: ProtocolContext, request: ChatRequest, ...):
        # Your adapter logic goes here.
        # 1. Translate the pai `ChatRequest` to your provider's format.
        # 2. Use `context.http_session` to make the API call.
        # 3. Parse the response (both streaming and non-streaming).
        # 4. Use `context.display` to show output.
        # 5. Use `context.stats.add_completed_request` to record metrics.
        # 6. Return the final result dictionary.
        
        # Refer to the built-in adapters in the main PAI repository
        # (e.g., anthropic_adapter.py) for detailed implementation examples.
        
        print("Hello from MyAI Adapter!")
        return {"text": "MyAI response"}
```

## Step 3: Register the Adapter as a Plugin

This is the most important step. In your `pyproject.toml`, you need to define an `entry_point` that tells `pai` where to find your adapter class.

```toml
# In myai-pai-adapter/pyproject.toml

[project]
name = "myai-pai-adapter"
version = "0.1.0"
dependencies = [
    "pai >= 0.1.0"  # Depend on the core pai package
]

# This section makes your adapter discoverable
[project.entry-points."polyglot_ai.protocols"]
myai = "myai_adapter.adapter:MyAIAdapter"
```

-   **`[project.entry-points."polyglot_ai.protocols"]`**: This declares that your package provides plugins for the `polyglot_ai.protocols` group. This exact group name is what `pai` scans for.
-   **`myai = ...`**: The key (`myai`) is the **name of your adapter**. This is the string you will use in `polyglot.toml` to refer to your adapter.
-   **`... = "myai_adapter.adapter:MyAIAdapter"`**: The value is the importable path to your adapter class (`package.module:ClassName`).

## Step 4: Install Your Plugin

To make your adapter available to `pai`, you must install your package into the same Python environment where `pai` is installed.

```bash
# From the root of your myai-pai-adapter/ directory
uv pip install -e .
```
Using `-e` (editable mode) is recommended during development, as it allows you to make changes to your adapter code without having to reinstall it every time.

## Step 5: Configure an Endpoint

Finally, open your `pai.toml` and add a new endpoint that uses the `name` you defined in your `entry_point`.

```toml
# In pai.toml

[[endpoints]]
name = "my_new_ai"
base_url = "https://api.myai.com/v1"
api_key_env = "MYAI_API_KEY"
# Use the key you registered in pyproject.toml
chat_adapter = "myai"
```

## You're Done!

Now, when you run `pai`, it will automatically discover and load your installed adapter.

```bash
# Check if the adapter was loaded (you should see a "✅ Loaded adapter 'myai'..." message)
uv run pai --help

# Run pai with your new provider
export MYAI_API_KEY="key-..."
uv run pai --chat --endpoint my_new_ai --model some-model
```

By following this pattern, you can integrate any AI provider into the Polyglot AI framework without ever needing to modify the core `pai` codebase.
