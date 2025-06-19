# 🪶 Polyglot AI Framework

**A Universal Command-Line Interface for Interacting with Any AI Provider.**

Polyglot AI is an interactive, provider-agnostic CLI designed for developers, researchers, and AI enthusiasts. It provides a single, unified interface to test, debug, and converse with a multitude of AI models from different providers like Featherless and OpenAI, with a plug-and-play architecture to easily add more.

![Demo Screenshot](https://i.imgur.com/g88t8D4.png) <!-- A placeholder for a future screenshot -->

### Key Features

*   **Provider Agnostic:** Seamlessly switch between different AI providers (`Featherless`, `OpenAI`, etc.) in a single session.
*   **Interactive Chat:** A rich, terminal-based chat experience with persistent command history (up/down arrows), streaming, helper commands, and a live status toolbar.
*   **Powerful Debugging:** A first-class, verbose debug mode to inspect raw API traffic, essential for development and research.
*   **Tool & Function Calling:** An integrated system for allowing models to use local Python functions to answer questions (e.g., for real-time data or actions).
*   **Agentic Looping:** The framework supports basic agentic loops where the model can use tools iteratively to solve complex problems.
*   **Automatic Session Logging:** Every interactive session is automatically saved to a timestamped folder in the `sessions/` directory. Each turn is saved as a structured JSON file, and the entire conversation is rendered into multiple browseable HTML formats.
*   **Extensible by Design:** Adding a new provider is as simple as creating a new class and registering it.

### Getting Started

#### 1. Prerequisites

*   Python 3.12+
*   See `pyproject.toml` for a full list of dependencies.

#### 2. Project Structure

Clone or download the repository. The core structure is:

```
/pai/
|-- pai.py                # The main script and orchestrator
|-- protocols/
|   |-- base_adapter.py
|   |-- openai_chat_adapter.py
|   |-- ...
|-- templates/
|   |-- conversation.html
|   |-- gptwink_format.html
|-- tools.py
/sessions/                # Auto-generated for session logs
/docs/                    # Detailed documentation
polyglot.toml             # Endpoint configuration
pyproject.toml            # Project dependencies and definition
```

#### 3. API Key Configuration

Polyglot AI loads API keys from environment variables. Set them in your shell or `.env` file:

```bash
export FEATHERLESS_API_KEY="sk-your-featherless-key"
export OPENAI_API_KEY="sk-your-openai-key"
```

#### 4. Running the Framework

**Install the package in editable mode:**
```bash
pip install -e .
```
This will install the `pai` command.

**Start in Interactive Mode (Default: OpenAI)**
```bash
pai --chat
```

**Start with a Different Endpoint (e.g., Featherless)**
```bash
pai --chat --endpoint featherless --model featherless/claude-3-opus
```

**Run a Single, Non-Interactive Prompt**
```bash
pai --endpoint openai --prompt "Explain quantum computing in one sentence."
```

**Enable Tool Calling**
```bash
pai --chat --endpoint openai --model gpt-4o --tools
```
Inside the interactive session, you can then ask: `What is the weather like in Paris?`

### Interactive Commands

Once in interactive mode, use `/` commands to control the session:

*   `/help`: Shows the list of available commands.
*   `/stats`: Displays statistics for the current session (tokens, response times, etc.).
*   `/provider <name>`: Switch the AI provider (e.g., `/provider openai`).
*   `/model <name>`: Change the model name (e.g., `/model gpt-4o-mini`).
*   `/tools`: Toggle tool-calling on or off.
*   `/debug`: Toggle raw protocol debugging on or off.
*   `/history`: View the JSON conversation history.
*   `/clear`: Clear the current conversation history.
*   `/quit`: Exit the application.

For more details on session logging, see `docs/LOGGING.md`.
```

---

### `ARCHITECTURE.md`

```markdown
# Polyglot AI Architecture

This document outlines the high-level architecture of the Polyglot AI framework. The design prioritizes modularity, extensibility, and clarity.

### High-Level Diagram

```
+---------------------------+       +---------------------------+
|      User (CLI)           |       |      TestSession          |
| (prompt_toolkit)          |       |      (Statistics)         |
+-------------^-------------+       +-------------^-------------+
              | (Input)                           | (Metrics)
              |                                   |
+-------------v-----------------------------------+-------------+
|                     main.py / FeatherlessClient                 |
|        (Orchestrator, State Management, Command Parser)       |
+-----------------------+----------------------^----------------+
| (Delegate Task)       | (Render Output)      | (Get Provider) |
|                       v                      |                |
|       +---------------+----------------+     |                |
|       |       Provider System        <-----+                |
|       |  (providers/base_provider.py)  |                      |
|       +---------------^----------------+                      |
|                       | (Implements)                          |
|      +----------------+----------------+                      |
|      |                |                |                      |
| +----v----+      +----v----+      +----v----+                 |
| |Featherless|      | OpenAI  |      | New...  | (Concrete     |
| +---------+      +---------+      +---------+  Providers)   |
|                      |                                        |
|                      +---------------------------> +----------+
|                                                    | Tool     |
|                                                    | System   |
|                                                    +----------+
+---------------------------------------------------------------+

```

### Core Components

1.  **`pai/pai.py` (The Orchestrator)**
    *   **Entrypoint:** Contains the `main()` function that parses command-line arguments using `argparse`.
    *   **Interactive Loop:** The `interactive_mode` function manages the main `while` loop, using `prompt_toolkit` for user input.
    *   **Command Parser:** Handles all `/` commands within the interactive loop, modifying session state.
    *   **`PolyglotClient` Class:** This is the central controller. It holds the session state (`TestSession`), the display handler (`StreamingDisplay`), endpoint configurations (`EndpointConfig`), and manages communication. It orchestrates the flow from user input to protocol adapter execution.

2.  **Protocol Adapter System (`pai/protocols/`)**
    *   **`base_adapter.py`:** Defines the `BaseProtocolAdapter` abstract class and the `ProtocolContext` data structure. This is the contract that all protocol adapters must adhere to. It requires a `generate()` method, ensuring a consistent interface for the `PolyglotClient`.
    *   **Concrete Adapters (`openai_chat_adapter.py`, etc.):** Each file implements a specific protocol handler. They are responsible for:
        *   Formatting the request payload for a specific API schema (e.g., OpenAI-compatible `/chat/completions`).
        *   Parsing the response stream.
        *   Handling protocol-specific features, like OpenAI's `tool_calls`.
        *   Calling back to the `ProtocolContext` to update stats and display output.

3.  **Tool System (`pai/tools.py`)**
    *   A standalone, decoupled module for defining and executing local functions.
    *   `@tool` Decorator: Registers functions into a `TOOL_REGISTRY`. It introspects the function's signature and docstring to automatically generate a JSON Schema that provider APIs (like OpenAI's) can understand.
    *   `execute_tool()`: A simple function that takes a tool name and arguments, runs the corresponding Python function, and returns the result.

4.  **Core Data Classes (within `pai/pai.py`)**
    *   **`Conversation` & `Turn`:** These dataclasses provide robust, object-oriented state management for conversations. a `Conversation` holds a list of `Turn` objects, and each `Turn` captures a single, complete request/response cycle, including all request/response data and metadata.
    *   **`TestSession`:** A dataclass for tracking all metrics of a session. It's a simple accumulator for requests, tokens, and timings.
    *   **`StreamingDisplay`:** Manages all console output during a streaming response. It crucially contains the `debug_mode` logic to print raw protocol traffic, which is vital for research and debugging new provider integrations.

### Session Persistence and Logging

A key feature of the framework is its ability to automatically log all interactive sessions for review and debugging. This process is handled by a few key components:

*   **`sessions/` directory:** When `interactive_mode` starts, it creates a unique, timestamped subdirectory within `sessions/`. This folder contains all artifacts for that specific session.
*   **`save_conversation_formats()`:** After every successful turn in `interactive_mode`, this function is called.
*   **`pai/templates/`:** This directory contains `Jinja2` templates for rendering conversation logs.
    *   `conversation.html`: A modern, styled view of the conversation.
    *   `gptwink_format.html`: A legacy-compatible format.
*   **Output Files:** `save_conversation_formats` generates:
    1.  A JSON file for each individual `Turn` (`<turn_id>-turn.json`).
    2.  An HTML file for each registered template, which is overwritten and updated after every turn.

This system ensures that no data is lost and provides multiple, easy-to-review formats for every interaction. For more details, see the `docs/LOGGING.md` file.

### Data Flow: A User Prompt with Tool Use

1.  User enters a prompt in the CLI: `What is the weather in Tokyo?`
2.  `interactive_mode()` in `pai.py` captures the input.
3.  It calls `conversation.get_messages_for_next_turn()` to construct the message history for the API call.
4.  A `ChatRequest` object is created with this history.
5.  `client.generate()` is called. It looks up the correct protocol adapter for the current endpoint (e.g., `OpenAIChatAdapter`) and calls its `.generate()` method, passing a `ProtocolContext` object.
6.  The `OpenAIChatAdapter` formats the final payload, including the message history and the JSON schemas of all registered tools from `tools.py`.
7.  It makes a streaming POST request to the provider's API via the shared `http_session` from the context.
8.  The API responds. Instead of text, it sends back a `tool_calls` object requesting to run `get_current_weather(location="Tokyo")`.
9.  The adapter parses this tool call. Instead of finishing, it calls `execute_tool("get_current_weather", ...)` from `tools.py`.
10. The tool runs and returns a JSON string: `{"location": "Tokyo", ...}`.
11. The adapter appends both the model's `tool_calls` request and the local `tool` result to its internal message list for the next iteration.
12. It **loops**, sending the *entire new history* back to the API in a second request.
13. The model, now having the weather data, generates the final text response: "The weather in Tokyo is 15°C and cloudy."
14. This text is streamed to the `StreamingDisplay`. The adapter returns the final request data, response data, and assistant text.
15. Back in `interactive_mode`, a `Turn` object is created with this data. It is added to the `Conversation` object, updating the managed history.
16. **`save_conversation_formats()` is called**, writing the new `Turn` to a JSON file and re-rendering the HTML log files in the session directory.
```
