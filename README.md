# 🪶 Polyglot AI Framework

**A Universal Command-Line Interface for Interacting with Any AI Provider.**

Polyglot AI is an interactive, provider-agnostic CLI designed for developers, researchers, and AI enthusiasts. It provides a single, unified interface to test, debug, and converse with a multitude of AI models from different providers like Featherless and OpenAI, with a plug-and-play architecture to easily add more.

![Demo Screenshot](https://i.imgur.com/g88t8D4.png) <!-- A placeholder for a future screenshot -->

### Key Features

*   **Universal Provider Support:** Seamlessly switch between different AI providers (`Featherless`, `OpenAI`, `Anthropic`, `Ollama`, etc.) and profiles in a single session using `/switch` and `/profile`.
*   **Advanced Interactive TUI:** A rich, terminal-based chat experience built on `prompt-toolkit`, featuring persistent and searchable command history, multiline input, and a live status toolbar that provides real-time feedback on cost, performance, and agent status.
*   **Deep Introspection & Debugging:** A first-class, verbose debug mode (`--debug`) to inspect raw API traffic, and a powerful `/stats` command to see detailed performance metrics for every request.
*   **Powerful Agentic Tool-Use:** An extensible system allowing models to use local Python functions as tools. Supports native tool-calling APIs (OpenAI, etc.) and provides a legacy agent mode for models that lack this capability.
    *   **To create tools:** See the [Tool System Guide](./docs/TOOLS.md).
    *   **For a walkthrough:** Check out the [Tool Usage Tutorial](./docs/TOOL_TUTORIAL.md).
*   **Automatic Session Logging:** Every interactive session is automatically saved to a timestamped folder in `sessions/`. Each turn is saved as structured JSON, and the entire conversation is rendered into multiple browseable HTML formats.
*   **Extensible by Design:** Add new providers via a simple plugin system. Add new tools by dropping Python files into a directory. No core code modification needed.
*   **Multi-Model Arena:** Pit models against each other in a conversational arena, with an optional judge model to provide a final verdict.

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
export OPENAI_API_KEY="sk-your-openai-key"
```

#### 4. Running the Framework

**Install the package in editable mode:**
```bash
uv sync -U
```
This will install the `pai` command.

**Start in Interactive Mode (Default: OpenAI)**
```bash
pai --chat
```

**Using a Profile**
You can define preset configurations in `polyglot.toml` and use them with the `--profile` flag. This is great for switching between common setups.
```bash
pai --profile research_haiku --chat
```

**Run a Single, Non-Interactive Prompt**
```bash
# This will stream the response by default.
pai --endpoint openai --prompt "Explain quantum computing in one sentence."

# Use --no-stream to get the full response at once.
pai --endpoint openai --prompt "Explain quantum computing in one sentence." --no-stream
```

**Using Tools**
To load and enable tools, you must start `pai` with the `--tools` flag. This gives the AI the *capability* to see and use tools.
```bash
pai --chat --endpoint openai --model gpt-4o --tools
```
For complex tasks, you can then switch into "agent mode" to give the AI a better reasoning framework for using those tools.
```
/agent
```
Now you can ask it to perform tasks that require tools: `Refactor the 'get_current_weather' function in 'pai/tools.py' to handle a 'kelvin' unit.`

### Interactive Commands

Once in interactive mode, use `/` commands to control the session:

*   `/help`: Shows the list of available commands.
*   `/stats`: Displays statistics for the current session.
*   `/switch <name>`: Switch to a different endpoint (e.g., `/switch openai`).
*   `/model <name>`: Change the model (e.g., `/model gpt-4o-mini`).
*   `/system ...`: Manage the system prompt stack (`add`, `pop`, `show`, `clear`, or replace).
*   `/prompt <name>`: Load a prompt from the `prompts/` directory and add it to the system prompt stack.
*   `/agent`: A shortcut to load the `code_editor` prompt and enable agent mode.
*   `/legacy_agent`: A shortcut to load the `legacy_agent` prompt for models without native tool-use.
*   `/tools`: Toggle tool-use on or off for the current session (Requires starting with `--tools`).
*   `/debug`: Toggle raw protocol debugging on or off.
*   `/clear`: Clear the current conversation history.
*   `/quit`: Exit the application.

For more details on session logging, see `docs/LOGGING.md`.
```

---

