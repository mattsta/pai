# 🪶 Polyglot AI Framework

**A Universal Command-Line Interface for Interacting with Any AI Provider.**

Polyglot AI is an interactive, provider-agnostic CLI designed for developers, researchers, and AI enthusiasts. It provides a single, unified interface to test, debug, and converse with a multitude of AI models from different providers like Featherless and OpenAI, with a plug-and-play architecture to easily add more.

![Demo Screenshot](https://i.imgur.com/g88t8D4.png) <!-- A placeholder for a future screenshot -->

### Key Features

*   **Provider Agnostic:** Seamlessly switch between different AI providers (`Featherless`, `OpenAI`, etc.) in a single session.
*   **Interactive Chat:** A rich, terminal-based chat experience with history, streaming, and helper commands.
*   **Powerful Debugging:** A first-class, verbose debug mode to inspect raw API traffic, essential for development and research.
*   **Tool & Function Calling:** An integrated system for allowing models to use local Python functions to answer questions (e.g., for real-time data or actions).
*   **Agentic Looping:** The framework supports basic agentic loops where the model can use tools iteratively to solve complex problems.
*   **Extensible by Design:** Adding a new provider is as simple as creating a new class and registering it.

### Getting Started

#### 1. Prerequisites

*   Python 3.8+
*   `requests` and `prompt_toolkit` libraries

```bash
pip install requests prompt_toolkit
```

#### 2. Project Structure

Clone or download the repository. The structure is simple:

```/PolyglotAI/
|-- main.py              # The main script you run
|-- providers/
|   |-- __init__.py
|   |-- base_provider.py
|   |-- featherless.py
|   |-- openai.py
|-- tools.py
```

#### 3. API Key Configuration

Polyglot AI loads API keys from environment variables. Set them in your shell or `.env` file:

```bash
export FEATHERLESS_API_KEY="sk-your-featherless-key"
export OPENAI_API_KEY="sk-your-openai-key"
```

#### 4. Running the Framework

**Start in Interactive Mode (Default: Featherless)**
```bash
python main.py
```

**Start with a Different Provider (e.g., OpenAI)**
```bash
python main.py --provider openai --model gpt-4o
```

**Run a Single, Non-Interactive Prompt**
```bash
python main.py --provider openai --prompt "Explain quantum computing in one sentence."
```

**Enable Tool Calling**
```bash
python main.py --provider openai --model gpt-4o --tools
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

1.  **`main.py` (The Orchestrator)**
    *   **Entrypoint:** Contains the `main()` function that parses command-line arguments using `argparse`.
    *   **Interactive Loop:** Manages the main `while` loop for interactive mode, using `prompt_toolkit` for user input.
    *   **Command Parser:** Handles all `/` commands, modifying the session state (e.g., changing models, toggling debug mode).
    *   **`FeatherlessClient` Class:** This is the central controller. It holds the session state (`TestSession`), the display handler (`StreamingDisplay`), and the currently active `provider`. It is responsible for orchestrating the flow from user input to provider execution.

2.  **Provider System (`providers/`)**
    *   **`base_provider.py`:** Defines the `BaseProvider` abstract class. This is the contract that all providers must adhere to. It requires a `name` property and a `generate()` method, ensuring a consistent interface for the `FeatherlessClient`.
    *   **Concrete Providers (`featherless.py`, `openai.py`):** Each file implements a specific provider. They are responsible for:
        *   Knowing their endpoint URL.
        *   Formatting the request payload according to the provider's API spec.
        *   Parsing the provider's response stream (or non-streamed response).
        *   Handling provider-specific features, like OpenAI's `tool_calls`.
        *   Updating the `TestSession` with statistics after a request.

3.  **Tool System (`tools.py`)**
    *   A standalone, decoupled module for defining and executing local functions.
    *   `@tool` Decorator: Registers functions into a `TOOL_REGISTRY`. It introspects the function's signature and docstring to automatically generate a JSON Schema that provider APIs (like OpenAI's) can understand.
    *   `execute_tool()`: A simple function that takes a tool name and arguments, runs the corresponding Python function, and returns the result.

4.  **Core Classes (within `main.py`)**
    *   **`TestSession`:** A dataclass for tracking all metrics of a session. It's a simple accumulator for requests, tokens, and timings.
    *   **`StreamingDisplay`:** Manages all console output during a streaming response. It crucially contains the `debug_mode` logic to print raw protocol traffic, which is vital for research and debugging new provider integrations.

### Data Flow: A User Prompt with Tool Use

1.  User enters a prompt in the CLI: `What is the weather in Tokyo?`
2.  `interactive_mode()` in `main.py` captures the input.
3.  It appends `{"role": "user", "content": ...}` to the `messages` history.
4.  `client.generate()` is called, which delegates to the active provider's (`OpenAIProvider`) `.generate()` method.
5.  `OpenAIProvider` formats the payload, including the message history and the JSON schemas of all registered tools from `tools.py`.
6.  It makes a streaming POST request to the OpenAI API.
7.  The API responds. Instead of text, it sends back a `tool_calls` object requesting to run `get_current_weather(location="Tokyo")`.
8.  The provider parses this, and instead of finishing, it calls `execute_tool("get_current_weather", ...)` from `tools.py`.
9.  The tool runs and returns a JSON string: `{"location": "Tokyo", ...}`.
10. The provider appends both the model's `tool_calls` request and the local `tool` result to the `messages` history.
11. It **loops**, sending the *entire new history* back to the API in a second request.
12. The model, now having the weather data, generates the final text response: "The weather in Tokyo is 15°C and cloudy."
13. This text is streamed to the `StreamingDisplay`, and the final result is added to the history.
```
