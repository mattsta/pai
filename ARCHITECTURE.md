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
|                       pai.py / PolyglotClient                   |
|        (Orchestrator, State Management, Command Handler)      |
+-----------------------+---------------------------------------+
| (Delegate Task)       | (Render Output)      | (Pass Context) |
|      (Adapter)        v (Display)            |                |
|       +---------------+----------------+     |                |
|       |   Protocol Adapter System    <-----+                |
|       | (protocols/base_adapter.py)  |                      |
|       +---------------^----------------+                      |
|                       | (Implements)                          |
|      +----------------+----------------+                      |
|      |                |                |                      |
| +----v----+      +----v----+      +----v----+                 |
| |OpenAIChat |      | Legacy...|     | New...  | (Concrete     |
| +---------+      +---------+      +---------+  Adapters)    |
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
    *   **`InteractiveUI` and `Application`:** The `InteractiveUI` class encapsulates all logic for the text user interface. It creates and manages a persistent `prompt_toolkit.Application`. This application, not a simple `while` loop, manages the entire UI lifecycle, including the input prompt, live output window, `SearchToolbar` for history, and status toolbar. This approach is essential for a non-blocking, stable user interface. It uses `prompt-toolkit`'s `FileHistory` for history and a custom `Ctrl+C` key binding to allow for gracefully cancelling running completions without exiting the application.
    *   **`CommandHandler` (`pai/commands.py`):** A dedicated class that parses and executes all `/` commands. Each command is its own class, making the system clean and easy to extend. The `CommandHandler` is instantiated by `InteractiveUI`.
    *   **`PolyglotClient` Class:** This is the central controller. It holds the session state (`TestSession`), the display handler (`StreamingDisplay`), endpoint configurations (`EndpointConfig`), and manages communication. It is passed to the `InteractiveUI` to orchestrate the flow from user input to protocol adapter execution.

2.  **Protocol Adapter System (`pai/protocols/`)**
    *   **`base_adapter.py`:** Defines the `BaseProtocolAdapter` abstract class and the `ProtocolContext` data structure. This is the contract that all protocol adapters must adhere to. It requires a `generate()` method, ensuring a consistent interface for the `PolyglotClient`.
    *   **Concrete Adapters (`openai_chat_adapter.py`, etc.):** Each file implements a specific protocol handler. They are responsible for:
        *   Formatting the request payload for a specific API schema (e.g., OpenAI-compatible `/chat/completions`).
        *   Parsing the response stream, with robust error handling to ignore malformed data chunks without crashing. In `debug` mode, these errors are printed for diagnostics.
        *   Handling protocol-specific features, like OpenAI's `tool_calls`.
        *   Calling back to the `ProtocolContext` to update stats and display output.
    *   **Adding New Providers:** The system is designed for easy extension. For a detailed walkthrough on creating a new adapter, see the [`How to Add a New Provider`](docs/providers/ANTHROPIC.md) guide.

3.  **Tool System (`pai/tools.py`)**
    *   A highly extensible system for defining and executing local functions that the AI can call.
    *   `@tool` Decorator: Registers functions into a `TOOL_REGISTRY`. It introspects the function's signature (supporting `str`, `int`, `float`, `bool`, and `Enum`) and docstring to automatically generate a JSON Schema for the provider API.
    *   `execute_tool()`: A robust dispatcher that takes a tool name and arguments, correctly converts `Enum` types, runs the tool, and returns the result.
    *   **Dynamic Loading:** The framework can automatically discover and load custom tools from user-defined directories, making it easy to add new capabilities without modifying core code. For a detailed guide, see [`docs/TOOLS.md`](./TOOLS.md).

4.  **Core Data Models (`pai/models.py`)**
    *   **`Conversation` & `Turn`:** These dataclasses provide robust, object-oriented state management for conversations. A `Conversation` holds a list of `Turn` objects, and each `Turn` captures a single, complete request/response cycle, including all request/response data and metadata.
    *   **`TestSession` & `RequestStats`:** These dataclasses track all metrics. `TestSession` is the accumulator for the entire session (total tokens, errors etc.). `RequestStats` holds detailed metrics for a single request, including `ttft` and `finish_reason`, and is used to power the live stats in the UI toolbar.
    *   **`StreamingDisplay`:** A critical component that manages all console output. It ensures that streaming responses do not corrupt the `prompt-toolkit` interface. It uses a swappable "printer" function to either print normally (for non-interactive use) or use `prompt-toolkit`'s thread-safe method (for interactive mode). It also tracks and exposes live state like `status` ("Waiting", "Streaming", etc.) and `live_tok_per_sec` (a smoothed average over the current stream's duration) to power the real-time UI toolbar.

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
6.  A `ChatRequest` object is created, and if tools are enabled, it is populated with the tool schemas from the `TOOL_REGISTRY`. The `OpenAIChatAdapter` then ensures the model receives this payload.
7.  It makes a streaming POST request to the provider's API via the shared `http_session` from the context.
8.  The API responds. Instead of text, it sends back a `tool_calls` object requesting to run `get_current_weather(location="Tokyo")`.
9.  The adapter parses this tool call. Instead of finishing, it calls `execute_tool("get_current_weather", ...)` from `tools.py`.
10. The tool runs and returns a JSON string: `{"location": "Tokyo", ...}`.
11. The adapter appends both the model's `tool_calls` request and the local `tool` result to its internal message list for the next iteration.
12. It **loops**, sending the *entire new history* back to the API in a second API call.
13. The model, now having the weather data, generates the final text response: "The weather in Tokyo is 15°C and cloudy."
14. This text is streamed to the `StreamingDisplay`. The adapter returns the final request data, response data, and assistant text.
15. Back in `interactive_mode`, a `Turn` object is created with this data. It is added to the `Conversation` object, updating the managed history.
16. **`save_conversation_formats()` is called**, writing the new `Turn` to a JSON file and re-rendering the HTML log files in the session directory.
