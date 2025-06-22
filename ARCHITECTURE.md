# Polyglot AI Architecture

This document outlines the high-level architecture of the Polyglot AI framework. The design prioritizes modularity, extensibility, and clarity.

### High-Level Diagram

```
+---------------------------+       +---------------------------+
|      User (CLI)           |       |      SessionStats         |
| (prompt_toolkit)          |       |      (Statistics)         |
+-------------^-------------+       +-------------^-------------+
              | (Input)                           | (Metrics)
              |                                   |
+-------------v-----------------------------------+-------------+
|                         InteractiveUI                         |
|           (UI Layout, State, Command Handler)                 |
+-----------------------------+---------------------------------+
                              | (Dispatch)
+-----------------------------v---------------------------------+
|                    Orchestrator System                        |
|             (pai/orchestration/base.py)                       |
+----^-----------+----------------------^------------------^----+
     |           |                      |                  |
(Implements) (Implements)          (Implements)       (Implements)
     |           |                      |                  |
+----+----+ +----v----+           +----v----+          +----v----+
| Default | | Legacy  |           | Arena   |          | (etc)   |
+---------+ +---------+           +---------+          +---------+
     |           |                      | (Delegates Task)
     +-----------+----------------------+
                 |
                 v
+----------------+----------------+
|     PolyglotClient / Display    |
+----------------+----------------+
                 |
  (Delegate Task)| (Pass Context)
                 v
+----------------+----------------+
|    Protocol Adapter System      |
|       | (protocols/base_adapter.py)  |                      |
|       +---------------^----------------+                      |
|                       | (Implements)                          |
|      +----------------+----------------+                      |
|      |                |                |                      |
| +----v----+      +----v----+      +----v----+      |
| |OpenAIChat |      | Legacy...|     |Anthropic| (Concrete     |
| +---------+      +---------+      +---------+       Adapters)    |
|                      |                                        |
|                      +---------------------------> +----------+
|                                                    | Tool     |
|                                                    | System   |
|                                                    +----------+
+---------------------------------------------------------------+

```

### Core Components

1.  **`pai/pai.py` (The UI and App Entrypoint)**
    *   **Entrypoint:** Contains the `typer` application and `run` command, which orchestrates the setup and teardown of the application.
    *   **`InteractiveUI`:** This class encapsulates all logic for the text user interface. It creates and manages a persistent `prompt_toolkit.Application` and is responsible for UI layout, state management (`UIMode`), and command dispatch. It no longer contains business logic for generation loops.
    *   **`CommandHandler` (`pai/commands.py`):** A dedicated class that parses and executes all `/` commands. Each command is its own class, making the system clean and easy to extend. The `CommandHandler` is instantiated by `InteractiveUI`.
    *   **Orchestrator System (`pai/orchestration/`)**: A new layer responsible for the business logic of different interaction modes. `InteractiveUI` instantiates and runs the appropriate orchestrator (`DefaultOrchestrator`, `ArenaOrchestrator`, etc.) based on the current `UIMode`. This cleanly separates UI concerns from logical flow.

2.  **`pai/client.py` (The Client Controller)**
    *   **`PolyglotClient` Class:** This is the central controller. It holds the session state (`SessionStats`), the display handler (`StreamingDisplay`), endpoint configurations (`EndpointConfig`), and manages communication. It is passed to the `InteractiveUI` to orchestrate the flow from user input to protocol adapter execution.

3.  **Protocol Adapter System (`pai/protocols/`)**
    *   **`__init__.py`:** This file implements a dynamic plugin system for protocol adapters. The `load_protocol_adapters()` function uses Python's `importlib.metadata` to discover and load any installed packages that provide an entry point for `polyglot_ai.protocols`. This populates a global `ADAPTER_MAP` at runtime, making the system highly extensible.
    *   **`base_adapter.py`:** Defines the `BaseProtocolAdapter` abstract class and the `ProtocolContext` data structure. This is the contract that all protocol adapters must adhere to. It requires a `generate()` method, ensuring a consistent interface for the `PolyglotClient`.
    *   **Concrete Adapters (`openai_chat_adapter.py`, etc.):** Each built-in adapter is registered via an entry point in `pyproject.toml`, serving as the reference implementation for the plugin system. They are responsible for:
        *   Formatting the request payload for a specific API schema.
        *   Parsing the response stream, with robust error handling.
        *   Handling protocol-specific features like tool-calling.
        *   Calling back to the `ProtocolContext` to update stats and display output.
    *   **Adding New Providers:** The system is now a formal plugin architecture. To add a new provider, you create a new installable Python package that exposes its adapter class via the `polyglot_ai.protocols` entry point. For a detailed guide, see [`How to Add a New Provider`](docs/providers/ANTHROPIC.md).

4.  **Tool System (`pai/tools.py`)**
    *   A highly extensible system for defining and executing local functions that the AI can call.
    *   `@tool` Decorator: Registers functions into a `TOOL_REGISTRY`. It introspects the function's signature (supporting `str`, `int`, `float`, `bool`, and `Enum`) and docstring to automatically generate a JSON Schema for the provider API.
    *   `execute_tool()`: A robust dispatcher that takes a tool name and arguments, correctly converts `Enum` types, runs the tool, and returns the result.
    *   **Dynamic Loading:** The framework can automatically discover and load custom tools from user-defined directories, making it easy to add new capabilities without modifying core code. For a detailed guide, see [`docs/TOOLS.md`](./TOOLS.md).

5.  **Core Data Models (`pai/models.py`)**
    *   **`Conversation` & `Turn`:** These dataclasses provide robust, object-oriented state management for conversations. A `Conversation` holds a list of `Turn` objects and also tracks `session_token_count`, which is the running total of tokens for the current conversation since the last `/clear`. `Turn` objects include optional fields for `participant_name` and `model_name` to support advanced logging scenarios.
    *   **`SessionStats` & `RequestStats`:** These dataclasses track all metrics. `SessionStats` is the accumulator for the entire application lifetime (total tokens, errors etc.). `RequestStats` holds detailed metrics for a single request, including `ttft` and `finish_reason`, and is used to power the live stats in the UI toolbar.
    *   **`Arena` & `ArenaParticipant`:** To support multi-model conversations, these dataclasses model an "arena" session. The `Arena` holds the configuration for the overall session, including a dictionary of `ArenaParticipant` objects. Each participant has its own model configuration and a dedicated `Conversation` history object.
    *   **`StreamingDisplay` (`pai/display.py`):** This critical component now lives in its own file. It manages all console output and ensures that streaming responses do not corrupt the `prompt-toolkit` interface. It uses a swappable "printer" function to either print normally (for non-interactive use) or use `prompt-toolkit`'s thread-safe method (for interactive mode). It also tracks and exposes live state like `status` ("Waiting", "Streaming", etc.) and `live_tok_per_sec` (a smoothed average over the current stream's duration) to power the real-time UI toolbar.
        *   **`StreamSmoother`:** For smooth streaming mode, the `StreamingDisplay` uses a `StreamSmoother` instance. This class contains the adaptive rendering logic. It maintains a smoothed words-per-second (`wps`) printing rate that adapts slowly to the live token rate from the provider. It then uses a proportional controller to compare the current render buffer size (in seconds) to a target size. If the buffer is too large, it speeds up printing (reduces delay); if it's too small, it slows down to rebuild the buffer. This decouples the print speed from network jitter, providing a predictable, continuous output.

### Architectural Evolution: A History of Refactoring

The framework underwent a series of planned refactoring phases to arrive at its current clean and modular state. This effort focused on:

*   **State Management (`Phase 1`):** The initial implementation used several boolean flags in `InteractiveUI` to manage state. This was refactored into a more robust `UIMode` enum and a central `UIState` dataclass, simplifying state management and making it easier to extend.
*   **Orchestrator Extraction (`Phase 2`):** Business logic for different modes (e.g., arena loops, agent loops) was extracted from `InteractiveUI` into a dedicated `pai/orchestration` layer. This decoupled the UI from the application's core logic, improving testability and clarity.
*   **Decoupling (`Phase 3`):** Direct state manipulation from `Command` classes was removed. Instead, commands now call dedicated setter/toggler methods on `InteractiveUI` and `PolyglotClient`, enforcing clear API boundaries and encapsulating state.
*   **Tool System Refinement (`Phase 4`):** The `tools.py` module was improved by separating schema-generation logic from the `@tool` decorator and by transitioning from error-string-based returns to a typed exception model (`ToolNotFound`, `ToolArgumentError`), making error handling more robust.

### Multi-Model Arena

The framework supports a "Multi-Model Arena" mode where two AI models can converse with each other. This is handled by the `ArenaOrchestrator`.

*   **Configuration:** Arenas are defined in `polyglot.toml`. They consist of two or more `participants` and an optional `judge`. These are loaded into `Arena` and `ArenaParticipant` data models.
*   **Orchestration:** The `ArenaOrchestrator` manages the pausable, turn-based conversation. Crucially, it maintains a separate `Conversation` object for each participant, ensuring each model receives a valid, alternating `user`/`assistant` history from its own perspective.
*   **Judge Model:** After the primary dialogue concludes (or is cancelled), the `ArenaOrchestrator` can invoke an optional judge model. The judge is provided with the *entire unified conversation history* and a special prompt to generate a final summary and verdict.
*   **Unified Logging:** While each participant has a private conversation history for generating its next turn, all turns—including the final verdict from the judge—are added to a **single, unified `Conversation` object** managed by the `InteractiveUI`. This unified history is what gets saved to the session log, providing a complete, interleaved record of the entire multi-model session.

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
4.  A `ChatRequest` object is created with this history. If tools are enabled, the request object is populated with the tool schemas from the `TOOL_REGISTRY`.
5.  `client.generate()` is called. It looks up the correct protocol adapter for the current endpoint (e.g., `OpenAIChatAdapter`) and calls its `.generate()` method, passing a `ProtocolContext` object containing the request and other state.
6.  The adapter formats the final payload, including the message history and any tool schemas, for the provider's specific API.
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
